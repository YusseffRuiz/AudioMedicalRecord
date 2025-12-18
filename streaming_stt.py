import datetime
import os
import time
import threading
import queue
import numpy as np
import sounddevice as sd
import torch
from dotenv import load_dotenv

from ASREngine import AsrEngine
from medical_filler import ClinicalFormFiller
from field_completer_engine import FieldCompleterEngine, FIELD_LABELS, FieldCompleterMistral
from audio_recording import AudioRecorder

# ======= Configuración =======
# ---- WHISPER -----
RATE = 16000          # Hz
CHANNELS = 1          # mono
CHUNK_SEC = 0.5       # tamaño de chunk de captura
WINDOW_SEC = 5       # ventana deslizante para inferencia
INTERVAL_SEC = 4      # cada cuánto lanzar transcripción
MAX_SECONDS = 120     # tope total de grabación (2 min)
MODEL_SIZE = "medium"  # sube a "medium" si tu CPU lo permite

# ---- VARIABLES MODELOS -----
DEVICE = 'cuda'if torch.cuda.is_available() else 'cpu'
# audio_file_path = "Audios/GrabacionPrueba_2.wav"
audio_file_path = "Audios/Grabacion_Prueba_3min.m4a"
LLM_MODEL = "../HF_Agents/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
AUDIO_SAVE_PATH = "_full_sessions"

ENV_PATH = "tokens.env"
load_dotenv(ENV_PATH)
api_key_mistral = os.getenv("MISTRAL_KEY")

# ======= Estado compartido =======
audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
stop_flag = threading.Event()
# running_flag = threading.Event()   # indica que estamos capturando (tecla presionada),
                                    # activar cuando migremos a permisos de administrador

# Buffer PCM int16 acumulado (lo mantenemos corto con ventana deslizante)
pcm_buffer = np.zeros(0, dtype=np.int16)
buffer_lock = threading.Lock()
# clinical_filler = ClinicalFormFiller()

TRANSCRIPT_LOG: list[str] = []

# Control para no imprimir duplicados
last_emitted_time = 0.0  # en segundos (según timestamps de whisper)

def finalize_session_and_save(llm_model, clinical_filler, transcript_full: list[str]):
    transcript_full = "".join(transcript_full)
    fields_final, still_missing_keys = process_transcript_with_regex_and_llm(llm_model, clinical_filler, transcript_full)


    # 3) guardar JSON de historial + transcript
    print("\n[FORM] ", clinical_filler.preview_text(), "\n", flush=True) # Forma ya llenada con el LLM.

    out_dir = "_historiales"
    os.makedirs(out_dir, exist_ok=True)
    fname = f"historial_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_dir = os.path.join(out_dir, fname)
    clinical_filler.save_json(str(out_dir), extras={
        "transcript": transcript_full,
        "input_tokens:": llm_model.input_tokens, "output_tokens:": llm_model.output_tokens})

    # 4) mensaje de faltantes (si los hay)

    if still_missing_keys:
        missing_labels = []
        seen = set()
        for key in still_missing_keys:
            label = FIELD_LABELS.get(key, key)
            if label not in seen:
                seen.add(label)
                missing_labels.append(label)
        print("\n⚠️  Faltan los siguientes campos: ", ", ".join(missing_labels))
        print("Dígalos o escríbalos manualmente, ya que no pude recuperarlos.")

    print(f"\n[OK] Historial guardado en: {out_dir}")

def process_transcript_with_regex_and_llm(llm_model, clinical_filler, transcript_full: str) -> tuple[dict, list[str]]:
    """
    Devuelve:
      - fields_final: dict con campos clínicos
      - missing_final: lista de campos que faltan al final
      - llm_text: texto bruto que devolvió el LLM (para debug)
    """

    # --- PASADA 1: REGEX directo ---
    fields_1 = clinical_filler.extract_with_regex(transcript_full)
    missing_1 = llm_model.compute_missing(fields_1)
    # print("Missing after regex: ", missing_1)

    if not missing_1:
        # No necesitamos LLM
        return fields_1, []

    # --- PASADA 2: LLM SOLO para campos faltantes ---
    llm_model.complete_fields(transcript_full, missing_1)

    fields_final = clinical_filler.snapshot()
    missing_final = llm_model.compute_missing(fields_final)

    return fields_final, missing_final


def audio_callback(indata, frames, time, status):
    # indata llega como float32 [-1..1] o int16 según driver;
    # forzamos int16 para coherencia
    if indata.dtype != np.int16:
        # convertimos a int16 (cuidado con clipping si venía como float)
        data = (indata[:, 0] * 32767.0).astype(np.int16)
    else:
        data = indata[:, 0].copy()
    audio_q.put(data)

def capture_thread():
    with sd.InputStream(
        samplerate=RATE,
        channels=CHANNELS,
        dtype='int16',          # pedimos int16
        blocksize=int(RATE*CHUNK_SEC),
        callback=audio_callback
    ):
        t0 = time.time()
        # while not stop_flag.is_set() and running_flag.is_set(): # Habilitar al implementar la retencion de tecla
        while not stop_flag.is_set():
            if time.time() - t0 >= MAX_SECONDS:
                stop_flag.set()
                break
            time.sleep(0.05)  # ceder CPU

# def consumer_thread(model: whisper.Whisper):
#     global pcm_buffer, last_emitted_time, TRANSCRIPT_LOG
#     last_infer = 0.0
#     t0 = time.time()
#
#     while not stop_flag.is_set():
#         # Vaciar la cola en el buffer
#         # drained = False
#         try:
#             chunk = audio_q.get(timeout=0.1)
#             with buffer_lock:
#                 pcm_buffer = np.concatenate([pcm_buffer, chunk])
#                 # Mantener como máximo WINDOW_SEC en buffer (ventana deslizante)
#                 max_samples = int(RATE * WINDOW_SEC)
#                 if pcm_buffer.size > max_samples:
#                     pcm_buffer = pcm_buffer[-max_samples:]
#             # drained = True
#         except queue.Empty:
#             pass
#
#         elapsed = time.time() - t0
#         if elapsed - last_infer >= INTERVAL_SEC:
#             last_infer = elapsed
#
#             # Copiar snapshot del buffer para inferir sin bloquear captura
#             with buffer_lock:
#                 snap = pcm_buffer.copy()
#
#             if snap.size == 0:
#                 # if not running_flag.is_set():  # ya se soltó la tecla y no queda nada
#                 #     break
#                 continue
#
#             # Convertir a float32 [-1,1]
#             audio_f = snap.astype(np.float32) / 32767.0
#
#             # Transcribir con timestamps para filtrar duplicados
#             # (sin timestamps también funciona, pero no sabríamos qué “nuevo” emitir)
#             try:
#                 _, _, result = model.transcribe_file(audio_f, language="es", without_timestamps=True)
#             except Exception as e:
#                 print(f"[WARN] Falló transcribir: {e}")
#                 continue
#
#             segments: List[dict] = result.get("segments", [])
#             new_text_parts = []
#
#             # Calcular la marca de tiempo de inicio del snapshot actual.
#             # Como solo mantenemos WINDOW_SEC, el tiempo 0 del snapshot es "ahora - WINDOW_SEC".
#             # Para comparar con last_emitted_time, mapeamos los tiempos relativos a absolutos:
#             #  - t_abs_segment = (tiempo_total_transcurrido - duracion_snapshot) + seg['end']
#             dur_snap = len(snap) / RATE
#             base_time_abs = elapsed - dur_snap  # tiempo absoluto de inicio del snapshot
#
#             for seg in segments:
#                 seg_end_abs = base_time_abs + float(seg.get("end", 0.0))
#                 if seg_end_abs > last_emitted_time + 0.05:
#                     new_text_parts.append(seg.get("text", ""))
#
#             if new_text_parts:
#                 # Emitir texto nuevo y actualizar la marca
#                 new_text = "".join(new_text_parts)
#                 TRANSCRIPT_LOG.append(new_text)
#                 print(new_text, flush=True)
#                 changed = clinical_filler.update(new_text)
#                 if changed:
#                     # Muestra vista rápida del estado cuando algo cambia
#                     print("\n[FORM] ", clinical_filler.preview_text(), "\n", flush=True)
#                 # Actualizamos last_emitted_time al fin del último segmento
#                 if segments:
#                     last_abs = base_time_abs + float(segments[-1].get("end", 0.0))
#                     last_emitted_time = max(last_emitted_time, last_abs)
#
#     # Al terminar, si queda algo, intentamos una última pasada
#     with buffer_lock:
#         snap = pcm_buffer.copy()
#     if snap.size:
#         audio_f = snap.astype(np.float32) / 32767.0
#         try:
#             _, _, result = model.transcribe_file(audio_f, language="es", without_timestamps=False)
#             segments = result.get("segments", [])
#             dur_snap = len(snap) / RATE
#             base_time_abs = (time.time() - (time.time() - dur_snap)) - dur_snap  # básicamente -dur_snap
#             new_text_parts = []
#             for seg in segments:
#                 seg_end_abs = base_time_abs + float(seg.get("end", 0.0))
#                 if seg_end_abs > last_emitted_time + 0.05:
#                     new_text_parts.append(seg.get("text", ""))
#             if new_text_parts:
#                 new_text = "".join(new_text_parts)
#                 TRANSCRIPT_LOG.append(new_text)
#                 print(new_text, flush=True)
#                 changed = clinical_filler.update(new_text)
#                 if changed:
#                     # Muestra vista rápida del estado cuando algo cambia
#                     print("\n[FORM] ", clinical_filler.preview_text(), "\n", flush=True)
#         except Exception:
#             pass


def record_full_session(rec_duration_sec: int = 15 * 60) -> str:
    """
    Graba toda la consulta en un WAV usando AudioRecorder.
    Devuelve la ruta al archivo WAV.
    """
    # Guardamos las sesiones completas en otra carpeta
    rec = AudioRecorder(duration=rec_duration_sec, path="_full_sessions", rate=16000, channels=1)

    print(f"\n[INFO] Se grabará una sesión completa de hasta {rec_duration_sec//60} minutos.")
    input("Presiona ENTER cuando estés listo para comenzar...")

    audio = rec.record_seconds()                 # bloqueante, dura rec.duration segundos
    wav_path = rec.save_audio(audio, sr=rec.rate)

    print(f"[OK] Sesión guardada en: {wav_path}")
    return wav_path

def transcribe_full_session(asr_engine, wav_path: str) -> str:
    print("\n[INFO] Transcribiendo sesión completa con Whisper...")
    text, conf, result = asr_engine.transcribe_file(wav_path, language="es")
    print(f"[OK] Transcripción lista. Confianza aprox.: {conf:.2f}")
    return text


def main():
    print(f"[INFO] Cargando modelo Whisper ({MODEL_SIZE})… ")
    asr = AsrEngine(model_size=MODEL_SIZE, device=DEVICE)
    clinical_filler = ClinicalFormFiller()
    audio = AudioRecorder(path=AUDIO_SAVE_PATH, rate=RATE, channels=CHANNELS)
    engine_num = int(input("Escribe 1 si quieres el engine local, Escribe 2, si quieres un proveedor externo\n"))
    # value = input("Modelo Listo. \n Escribe [1] si va a ser live streaming. \n Escribe [2] si vas a grabar la sesion completa.\n").strip()
    value = str(input("Escribe 2 para grabar la sesion \nEscribe 3 para usar un audio pre grabado.\n"))
    wav_path = None
    if value == "1":
        print("Comienza el streaming… (Usa cntrl+C para salir)")
        print("Presiona ENTER para comenzar a grabar y usa cntrl+C para finalizar la grabacion.")
        t_cap = threading.Thread(target=capture_thread, daemon=True)
        # t_con = threading.Thread(target=consumer_thread, args=(asr,), daemon=True)
        t_cap.start()
        # t_con.start()
        try:
            while not stop_flag.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            stop_flag.set()
            print("\n\n[FIN] Streaming detenido.")

        t_cap.join()
            # t_con.join()
        transcript_local = TRANSCRIPT_LOG
    elif value == "2":  #Grabar la sesion completa
        # wav_path = record_full_session(rec_duration_sec=15 * 60)  # grabacion por varios minutos completos
        audio_file = audio.record_until_stop()
        wav_path = audio.save_audio(audio_file)
    else:  # Uso de Audio pre grabado
        wav_path = audio_file_path
    t_0_transcript = time.time()
    transcript_local = transcribe_full_session(asr, wav_path)
    t_1_transcript = time.time()
    t_transcript = t_1_transcript - t_0_transcript

    print("[INFO] Inicializando analisis via LLM")
    if engine_num == 1:
        llm_filler = FieldCompleterEngine(medical_filler=clinical_filler, max_tokens_per_chunk=1000, device=DEVICE)
        llm_filler.initialize(model_name=LLM_MODEL)
    else:
        llm_filler = FieldCompleterMistral(medical_filler=clinical_filler, max_tokens_per_chunk=1000)
        llm_filler.initialize(api_key=api_key_mistral)

    t_0_process = time.time()
    finalize_session_and_save(llm_filler, clinical_filler, transcript_local)
    t_1_process = time.time()
    t_process = t_1_process - t_0_process
    print("[INFO] Analisis Finalizado, Favor de SIEMPRE revisar, corroborar y corregir los apartados:"
          "\nAlergias\nDiagnóstico\nReceta.")
    print(f"[INFO] Tokens de entrada: {llm_filler.input_tokens}, tokens de salida: {llm_filler.output_tokens}")
    print(f"[INFO] Time to transcript: {t_transcript} seconds")
    print(f"[INFO] Time to fill medical history: {t_process} seconds")


def main_external_engine():
    print(f"Cargando modelo Whisper ({MODEL_SIZE})… ")
    asr = AsrEngine(model_size=MODEL_SIZE, device=DEVICE)
    clinical_filler = ClinicalFormFiller()
    wav_path = audio_file_path
    t_0_transcript = time.time()
    transcript_local = transcribe_full_session(asr, wav_path)
    t_1_transcript = time.time()
    t_transcript = t_1_transcript - t_0_transcript

    print("[INFO] Inicializando analisis via LLM")
    llm_filler = FieldCompleterMistral(medical_filler=clinical_filler, max_tokens_per_chunk=1000)
    llm_filler.initialize(api_key=api_key_mistral)

    t_0_process = time.time()
    finalize_session_and_save(llm_filler, clinical_filler, transcript_local)
    t_1_process = time.time()
    t_process = t_1_process - t_0_process
    print("[INFO] Analisis Finalizado, Favor de SIEMPRE revisar, corroborar y corregir los apartados:"
          "\nAlergias\nDiagnóstico\nReceta.")
    print(f"[INFO] Time to transcript: {t_transcript} seconds")
    print(f"[INFO] Time to fill medical history: {t_process} seconds")

def prueba_llm():
    TRANSCRIPT_LOG =[' nuevamente comenzamos una', ' Comenzamos un nuevo modelo para paciente de nombre Adan cuya edad es de 33 años', ' 33 años peso de 83 kilogramos', ' altura es de uno 76 metros', ' Su tension arterial es de 120 sobre 80 y su frecuencia cardíaca es de 80.', 'la glucosa se encuentra en 40 y la temperatura corporea', ' la temperatura corporal en 36.5 grados centígrados su y', ' IMC se encuentra en rango normal de 24 IMC y el oxigen...', ' y el oxígeno en la sangre de 86%.']
    print("Inicializando analisis via LLM")
    clinical_filler = ClinicalFormFiller()
    llm_filler = FieldCompleterMistral(medical_filler=clinical_filler, max_tokens_per_chunk=1000)
    print("Texto a transcribir:")
    print(str(TRANSCRIPT_LOG))
    finalize_session_and_save(llm_filler, clinical_filler, TRANSCRIPT_LOG)
    print("Analisis Finalizado, Favor de SIEMPRE revisar, corroborar y corregir los apartados:"
          "\nAlergias\nDiagnóstico\nReceta.")


if __name__ == "__main__":
    main()
    # prueba_llm()
    # main_external_engine()
