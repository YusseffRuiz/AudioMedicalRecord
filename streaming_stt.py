import time
import threading
import queue
import numpy as np
import sounddevice as sd
import whisper
from typing import List
from ASREngine import AsrEngine
from medical_filler import ClinicalFormFiller

# ======= Configuración =======
RATE = 16000          # Hz
CHANNELS = 1          # mono
CHUNK_SEC = 0.5       # tamaño de chunk de captura
WINDOW_SEC = 5       # ventana deslizante para inferencia
INTERVAL_SEC = 4      # cada cuánto lanzar transcripción
MAX_SECONDS = 120     # tope total de grabación (2 min)
MODEL_SIZE = "small"  # sube a "medium" si tu CPU lo permite

# ======= Estado compartido =======
audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
stop_flag = threading.Event()
# running_flag = threading.Event()   # indica que estamos capturando (tecla presionada),
                                    # activar cuando migremos a permisos de administrador

# Buffer PCM int16 acumulado (lo mantenemos corto con ventana deslizante)
pcm_buffer = np.zeros(0, dtype=np.int16)
buffer_lock = threading.Lock()
clinical_filler = ClinicalFormFiller()

# Control para no imprimir duplicados
last_emitted_time = 0.0  # en segundos (según timestamps de whisper)

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
    global pcm_buffer, last_emitted_time
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
        input("\nENTER para grabar ")
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
        print("\n[ESTADO FINAL] ", clinical_filler.preview_text())

if __name__ == "__main__":
    main()
