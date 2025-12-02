from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pathlib import Path
import tempfile
import time
import os

from .ASREngine import AsrEngine
from .medical_filler import ClinicalFormFiller
from .field_completer_engine import FieldCompleterEngine, FIELD_LABELS



app = FastAPI(title="Clinical Capture API", version="0.1.0")

# Config mínima vía variables de entorno
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
# LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "models/llama-2-7b-chat.Q4_K_M.gguf")
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

# Instancias globales
asr_engine = AsrEngine(model_size=WHISPER_MODEL)
clinical_filler = ClinicalFormFiller()


@app.get("/healthz")
def healthz():
    return {"status": "ok"}

def process_transcript_with_regex_and_llm(llm_model, form_filler, transcript_full: str) -> tuple[dict, list[str]]:
    """
    Devuelve:
      - fields_final: dict con campos clínicos
      - missing_final: lista de campos que faltan al final
      - llm_text: texto bruto que devolvió el LLM (para debug)
    """

    # --- PASADA 1: REGEX directo ---
    fields_1 = form_filler.extract_with_regex(transcript_full)
    missing_1 = llm_model.compute_missing(fields_1)

    if not missing_1:
        # No necesitamos LLM
        return fields_1, []

    # --- PASADA 2: LLM SOLO para campos faltantes ---
    llm_model.complete_fields(transcript_full, missing_1)

    fields_final = form_filler.snapshot()
    print(fields_final)
    missing_final = llm_model.compute_missing(fields_final)

    return fields_final, missing_final


@app.post("/api/v1/process_audio_session")
async def process_audio_session(
    file: UploadFile = File(...),
    patient_id: str | None = Form(None),
    session_id: str | None = Form(None),
    language: str = Form("es"),
):

    # 1) Guardar audio temporalmente
    suffix = Path(file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    print("Audio file found")
    # 2) Transcribir con Whisper
    t0 = time.time()
    text, conf, result = asr_engine.transcribe_file(str(tmp_path), language=language)
    t1 = time.time()
    # 3) Extraer campos clínicos usando tu filler (regex + guardrails)
    #    IMPORTANTE: limpia el estado primero, para no contaminar entre multiples requests
    clinical_filler.reset_state()
    llm_filler = FieldCompleterEngine(LLM_MODEL_PATH, medical_filler=clinical_filler, max_tokens_per_chunk=300)
    print("LLM Created")
    transcript_full = "".join(text)
    t2 = time.time()
    fields_final, still_missing_keys = process_transcript_with_regex_and_llm(llm_filler, clinical_filler, transcript_full)
    t3 = time.time()

    # 4) Determinar campos faltantes (ignorando derivados)
    missing_labels = []
    if still_missing_keys:
        seen = set()
        for key in still_missing_keys:
            label = FIELD_LABELS.get(key, key)
            if label not in seen:
                seen.add(label)
                missing_labels.append(label)


    dt_ms = int((time.time() - t0) * 1000)

    resp = {
        "patient_id": patient_id,
        "session_id": session_id,
        # "transcript": text,  # Para el desarrollo final, eliminar, solo sirve para el debugging
        "clinical_fields": fields_final,
        "missing_fields": ", ".join(missing_labels),
        "meta": {
            "whisper_model": WHISPER_MODEL,
            "asr_time": int((t1 - t0)*1000),
            "llm_time": int((t3 - t2)*1000),
            "processing_ms": dt_ms,
        },
    }

    # Limpieza
    try:
        tmp_path.unlink()
    except Exception:
        pass


    output = JSONResponse(resp)
    return output