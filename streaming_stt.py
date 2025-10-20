import time
import threading
import queue
import numpy as np
import sounddevice as sd
import whisper
from typing import List


from ASREngine import AsrEngine
from medical_filler import ClinicalFormFiller, REQUIRED_MEDICAL_FIELDS
from field_completer_engine import FieldCompleterEngine

# ======= Configuración =======
RATE = 16000          # Hz
CHANNELS = 1          # mono
CHUNK_SEC = 0.5       # tamaño de chunk de captura
WINDOW_SEC = 5       # ventana deslizante para inferencia
INTERVAL_SEC = 4      # cada cuánto lanzar transcripción
MAX_SECONDS = 120     # tope total de grabación (2 min)
MODEL_SIZE = "small"  # sube a "medium" si tu CPU lo permite

# ====== Prompt para el LLM ======
EXTRACTION_PROMPT = (
    "Eres un extractor clínico estricto en español que tiene que llenar los siguientes campos SIN cambiar el nombre:"
    "- Edad (entero, años)\n"
    "- Peso (float, kg),"
    "- Talla (si escuchas algo como 76 m, probablemente sea 1.76 metros) (float, m),"
    "- Tension Arterial (en forma X,Y) (Si escuchas: 'frecuencia cardiaca' pero el valor lo da como en formato '120 sobre 80, entonces es tension"
    "- Frecuencia Cardiaca (entero)"
    "- Frecuencia Respiratoria (entero)"
    "- SpO2 (entero)"
    "- Temperatura (float)"
    "- Glucosa (float)\n"
    "- Alergias (string corto)"
    "- Diagnostico (string)"
    "- Receta (string)\n\n"
    "Reglas: Traduce a los campos mencionados.\n"
    "Con el contexto que se te ofrezca, tienes que popular la información de dichos campos de la manera en que se indica en cada uno\n"
    "Valores numéricos con punto decimal. Unidades ya normalizadas. \n"
    "Ejemplos:\n"
    "Texto: 'tensión ciento veinte ochenta, frecuencia 78, frecuencia respiratoria 60, saturación 97 por ciento, altura 1 76'\n"
    "{\"Tension Arterial\":120, 80\"\"Frecuencia Cardiaca:78, \"Frecuencia Respiratoria\":68, \"SpO2\":97, \"Talla\":1.76}\n\n"
    "Texto: 'talla uno setenta y dos, peso 82 kilogramos'\n"
    "{\"Talla\":1.72, \"Peso\":82.0}\n\n"
    "Texto: 'Tension ciento diez sobre setenta'\n"
    "{\"Tension Arterial\":110, 70}\n\n"
)


# ======= Estado compartido =======
audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
stop_flag = threading.Event()
# running_flag = threading.Event()   # indica que estamos capturando (tecla presionada),
                                    # activar cuando migremos a permisos de administrador

# Buffer PCM int16 acumulado (lo mantenemos corto con ventana deslizante)
pcm_buffer = np.zeros(0, dtype=np.int16)
buffer_lock = threading.Lock()
clinical_filler = ClinicalFormFiller()

TRANSCRIPT_LOG: list[str] = []

# Control para no imprimir duplicados
last_emitted_time = 0.0  # en segundos (según timestamps de whisper)

def finalize_session_and_save(llm_model, transcript_full: list[str]):
    transcript_full = str(transcript_full)
    # 1) estado actual por regex
    current = clinical_filler.snapshot()

    # 2) completar con LLM si faltan campos
    llm_model.complete_fields(transcript_full, current)

    # 3) guardar JSON de historial + transcript
    print("\n[FORM] ", clinical_filler.preview_text(), "\n", flush=True)

    # out_dir = "_historiales"
    # os.makedirs(out_dir, exist_ok=True)
    # fname = f"historial_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    # out_dir = os.path.join(out_dir, fname)
    # clinical_filler.save_json(str(out_dir), extras={"transcript": transcript_full})

    # 4) mensaje de faltantes (si los hay)
    still_missing = [k for k in REQUIRED_MEDICAL_FIELDS if getattr(clinical_filler.state, k, None) in (None, "", 0) and k not in {"imc","tam_map"}]
    if still_missing:
        print("\n⚠️  Faltan los siguientes campos: ", ", ".join(still_missing))
        print("Dígalos o escríbalos manualmente, ya que no pude recuperarlos.")

    # print(f"\n[OK] Historial guardado en: {out_dir}")


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

def consumer_thread(model: whisper.Whisper):
    global pcm_buffer, last_emitted_time, TRANSCRIPT_LOG
    last_infer = 0.0
    t0 = time.time()

    while not stop_flag.is_set():
        # Vaciar la cola en el buffer
        # drained = False
        try:
            chunk = audio_q.get(timeout=0.1)
            with buffer_lock:
                pcm_buffer = np.concatenate([pcm_buffer, chunk])
                # Mantener como máximo WINDOW_SEC en buffer (ventana deslizante)
                max_samples = int(RATE * WINDOW_SEC)
                if pcm_buffer.size > max_samples:
                    pcm_buffer = pcm_buffer[-max_samples:]
            # drained = True
        except queue.Empty:
            pass

        elapsed = time.time() - t0
        if elapsed - last_infer >= INTERVAL_SEC:
            last_infer = elapsed

            # Copiar snapshot del buffer para inferir sin bloquear captura
            with buffer_lock:
                snap = pcm_buffer.copy()

            if snap.size == 0:
                # if not running_flag.is_set():  # ya se soltó la tecla y no queda nada
                #     break
                continue

            # Convertir a float32 [-1,1]
            audio_f = snap.astype(np.float32) / 32767.0

            # Transcribir con timestamps para filtrar duplicados
            # (sin timestamps también funciona, pero no sabríamos qué “nuevo” emitir)
            try:
                _, _, result = model.transcribe_file(audio_f, language="es", without_timestamps=True)
            except Exception as e:
                print(f"[WARN] Falló transcribir: {e}")
                continue

            segments: List[dict] = result.get("segments", [])
            new_text_parts = []

            # Calcular la marca de tiempo de inicio del snapshot actual.
            # Como solo mantenemos WINDOW_SEC, el tiempo 0 del snapshot es "ahora - WINDOW_SEC".
            # Para comparar con last_emitted_time, mapeamos los tiempos relativos a absolutos:
            #  - t_abs_segment = (tiempo_total_transcurrido - duracion_snapshot) + seg['end']
            dur_snap = len(snap) / RATE
            base_time_abs = elapsed - dur_snap  # tiempo absoluto de inicio del snapshot

            for seg in segments:
                seg_end_abs = base_time_abs + float(seg.get("end", 0.0))
                if seg_end_abs > last_emitted_time + 0.05:
                    new_text_parts.append(seg.get("text", ""))

            if new_text_parts:
                # Emitir texto nuevo y actualizar la marca
                new_text = "".join(new_text_parts)
                TRANSCRIPT_LOG.append(new_text)
                print(new_text, flush=True)
                changed = clinical_filler.update(new_text)
                if changed:
                    # Muestra vista rápida del estado cuando algo cambia
                    print("\n[FORM] ", clinical_filler.preview_text(), "\n", flush=True)
                # Actualizamos last_emitted_time al fin del último segmento
                if segments:
                    last_abs = base_time_abs + float(segments[-1].get("end", 0.0))
                    last_emitted_time = max(last_emitted_time, last_abs)

    # Al terminar, si queda algo, intentamos una última pasada
    with buffer_lock:
        snap = pcm_buffer.copy()
    if snap.size:
        audio_f = snap.astype(np.float32) / 32767.0
        try:
            _, _, result = model.transcribe_file(audio_f, language="es", without_timestamps=False)
            segments = result.get("segments", [])
            dur_snap = len(snap) / RATE
            base_time_abs = (time.time() - (time.time() - dur_snap)) - dur_snap  # básicamente -dur_snap
            new_text_parts = []
            for seg in segments:
                seg_end_abs = base_time_abs + float(seg.get("end", 0.0))
                if seg_end_abs > last_emitted_time + 0.05:
                    new_text_parts.append(seg.get("text", ""))
            if new_text_parts:
                new_text = "".join(new_text_parts)
                TRANSCRIPT_LOG.append(new_text)
                print(new_text, flush=True)
                changed = clinical_filler.update(new_text)
                if changed:
                    # Muestra vista rápida del estado cuando algo cambia
                    print("\n[FORM] ", clinical_filler.preview_text(), "\n", flush=True)
        except Exception:
            pass

def main():
    print(f"Cargando modelo Whisper ({MODEL_SIZE})… ")
    asr = AsrEngine(model_size=MODEL_SIZE, device="cpu")
    print("Modelo listo. Comienza el streaming… (Ctrl+C para salir, máx 2 min)")

    while True:
        print("Presiona ENTER para comenzar a grabar y cntrl+C para terminar.")
        fin = input("\nComienza a dictar ")
        if fin == "Terminar":
            break
        t_cap = threading.Thread(target=capture_thread, daemon=True)
        t_con = threading.Thread(target=consumer_thread, args=(asr,), daemon=True)
        t_cap.start()
        t_con.start()
        try:
            while not stop_flag.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            stop_flag.set()

        t_cap.join()
        t_con.join()
        print("\n\n[FIN] Streaming detenido.")
        print("Inicializando analisis via LLM")
        llm_model_name = "../llama-2-7b-chat.Q4_K_M.gguf"
        llm_filler = FieldCompleterEngine(llm_model_name, medical_filler=clinical_filler)
        print("Texto a transcribir:")
        print(str(TRANSCRIPT_LOG))
        finalize_session_and_save(llm_filler, TRANSCRIPT_LOG)
        print("Analisis Finalizado")
        print("\n[ESTADO FINAL] ", clinical_filler.preview_text())


def prueba_llm():
    TRANSCRIPT_LOG =[' nuevamente comenzamos una', ' Comenzamos un nuevo modelo para paciente de nombre Adan cuya edad es de 33 años', ' 33 años peso de 83 kilogramos', ' altura es de uno 76 metros', ' Y su frecuencia cardíaca es de 120 sobre 8.', ' 220 sobre 80 la glucosa se encuentra en 40 y la temperatura corporea', ' la temperatura corporal en 26.5 grados centígrados su y', ' IMC se encuentra en rango normal de 24 IMC y el oxigen...', ' y el oxígeno en la sangre de 86%.']
    print("Inicializando analisis via LLM")
    llm_model_name = "../llama-2-7b-chat.Q4_K_M.gguf"
    llm_filler = FieldCompleterEngine(llm_model_name, medical_filler=clinical_filler)
    print("Texto a transcribir:")
    print(str(TRANSCRIPT_LOG))
    finalize_session_and_save(llm_filler, TRANSCRIPT_LOG)
    print("Analisis Finalizado")
    print("\n[ESTADO FINAL] ", clinical_filler.preview_text())


if __name__ == "__main__":
    # main()
    prueba_llm()
