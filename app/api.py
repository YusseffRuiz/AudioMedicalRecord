import datetime
import sys
import uuid
import logging


from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, LiteralString
from pathlib import Path
import tempfile
import time
import os

from .ASREngine import AsrEngine
from .medical_filler import ClinicalFormFiller
from .field_completer_engine import FieldCompleterEngine, FIELD_LABELS, FieldCompleterMistral
from .utils import health


# ----------------- Modelos Pydantic de respuesta -----------------
class ErrorAudio(BaseModel):
    whisper_engine: Optional[str] = None
    attempt: Optional[str] = None
    filename: Optional[str] = None
    stage: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


class ErrorPayload(BaseModel):
    type: str                  # p.ej. "validation_error", "model_error", "transcript_error"
    message: str               # mensaje entendible
    detail: Optional[str] = None  # detalle técnico más específico
    context: Optional[ErrorAudio] = None

class HISTORYApiError(Exception):
    def __init__(
        self,
        *,
        type: str,
        message: str,
        detail: Optional[str] = None,
        context: Optional[dict] = None,
        status_code: int = 400,
    ):
        self.type = type
        self.message = message
        self.detail = detail
        self.context = context or {}
        self.status_code = status_code
        super().__init__(message)

class HISTORYData(BaseModel):
    patient_id: Optional[str] = None
    session_id: Optional[str] = None
    transcript: Optional[str] = None
    clinical_fields: Optional[Dict] = None
    missing_fields: Optional[str] = None
    recommendations: Optional[str] = None # En desarrollo
    date: Optional[str] = None

class HISTORYMeta(BaseModel):
    whisper_model: Optional[str] = None
    asr_time: str
    llm_time: str
    processing_ms: str
    warnings: List[str] = []


class HISTORYOKResponse(BaseModel):
    status: str = "ok"
    data: HISTORYData
    meta: HISTORYMeta


class HISTORYErrorDetail(BaseModel):
    type: str
    message: str
    suggestion: Optional[str] = None


class HISTORYErrorResponse(BaseModel):
    status: str = "error"
    error: HISTORYErrorDetail

# --------------------------------
app = FastAPI(title="Clinical Capture API", version="0.1.0")

# Config mínima vía variables de entorno
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
# LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "models/llama-2-7b-chat.Q4_K_M.gguf")
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

# Carga de llave
BASE_DIR = Path(__file__).resolve().parent.parent   # sube dos carpetas
ENV_PATH = BASE_DIR / "tokens.env"
load_dotenv(ENV_PATH)
api_key = os.getenv("MISTRAL_API_KEY")

# Instancias globales
asr_engine = AsrEngine(model_size=WHISPER_MODEL)
clinical_filler = ClinicalFormFiller()
llm_filler = FieldCompleterMistral(medical_filler=clinical_filler, max_tokens_per_chunk=1000)
llm_filler.initialize(api_key=api_key)

# -----------------Error Handling ---------------------
logger = logging.getLogger("history_api")

logging.basicConfig(
    level=logging.INFO,  # ⬅️ IMPORTANTE
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],
)

@app.exception_handler(HISTORYApiError)
async def history_api_error_request(request: Request, exc: HISTORYApiError):
    payload = ErrorPayload(
        type=exc.type,
        message=exc.message,
        detail=exc.detail,
        context=ErrorAudio(**exc.context) if exc.context else None,
    )

    # Log estructurado
    logger.error(
        "HISTORYApiError",
        extra={
            "error_type": exc.type,
            "message": exc.message,
            "detail": exc.detail,
            "context": exc.context,
            "path": str(request.url),
        },
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={"error": payload.dict()},
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    # Aquí atrapamos lo que no controlamos
    logger.exception("Unhandled exception in HISTORY API", extra={"path": str(request.url)})

    payload = ErrorPayload(
        type="internal_error",
        message="Ocurrió un error inesperado procesando el audio.",
        detail=str(exc),
        context=ErrorAudio(
            extra={"path": str(request.url)}
        ),
    )

    return JSONResponse(
        status_code=500,
        content={"error": payload.dict()},
    )

# ----------------- Endpoint principal -----------------


@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/readyz")
async def readyz():
    components = {
        "asr_engine": health.check_asr(asr_engine),
        "llm_model": health.check_mistral(llm_filler),
    }

    all_ok = all(c["ok"] for c in components.values())

    status = "ok" if all_ok else "degraded"

    payload = {
        "status": status,
        "components": components,
    }

    status_code = 200 if all_ok else 503
    return JSONResponse(content=payload, status_code=status_code)

@app.post(
    "/api/v1/process_audio_session",
    response_model=HISTORYOKResponse,
    responses={
        400: {"model": HISTORYErrorResponse},
        415: {"model": HISTORYErrorResponse},
        422: {"model": HISTORYErrorResponse},
        500: {"model": HISTORYErrorResponse},
    },
)

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
    missing_final = llm_model.compute_missing(fields_final)

    return fields_final, missing_final


@app.post("/api/v1/process_audio_session")
async def process_audio_session(
    file: UploadFile = File(...),
    patient_id: str | None = Form(None),
    language: str = Form("es"),
):
    session_id = str(uuid.uuid4().hex)[:4]  # Caben 65,536 requests

    # 0.5) Validar tipo de archivo
    allowed_types = {
        "audio/m4a",
        "audio/mp4",
        "audio/wav",
    }
    if file.content_type not in allowed_types:
        raise HISTORYApiError(
            status_code=415,
            type="unsupported_media_type",
            message=f"Formato no soportado: {file.content_type}. Use m4a, mp4, wav.",
        )

    # 1) Guardar audio temporalmente
    suffix = Path(file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    # 2) Transcribir con Whisper
    t_0_transcript = time.time()
    text, conf, result = asr_engine.transcribe_file(str(tmp_path), language=language)
    t_1_transcript = time.time()

    if text is None:
        raise HISTORYApiError(
            type="transcription_error",
            message="No se pudo extraer texto del audio.",
            detail="El motor de transcripción retornó el audio vacío, probablemente no se grabó correctamente",
            context={
                "whisper_model": WHISPER_MODEL,
                "filename": file.filename,
                "stage": "transcription",
            },
            status_code=422,
        )

    # 3) Extraer campos clínicos usando tu filler (regex + guardrails)
    #    IMPORTANTE: limpia el estado primero, para no contaminar entre multiples requests
    clinical_filler.reset_state()

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

    dt_transcript_s = int(t_1_transcript - t_0_transcript)
    dt_llm_s = int(t3 - t2)
    dt_total = time.time() - t_0_transcript

    # Limpieza
    try:
        tmp_path.unlink()
    except Exception:
        pass

    data = HISTORYData(
        patient_id=patient_id,
        session_id=session_id,
        # transcript= text,  # Solo sirve para el debugging
        clinical_fields=fields_final,
        missing_fields=", ".join(missing_labels),
        recommendations="⚠️ En desarrollo...",
        date=str(datetime.datetime.now()),
    )

    meta = HISTORYMeta(
        whisper_model=WHISPER_MODEL,
        asr_time=f"{dt_transcript_s} s",
        llm_time=f"{dt_llm_s} s",
        processing_ms=f"{dt_total} s",
        warnings=[],
    )

    response = HISTORYOKResponse(status="ok", data=data, meta=meta)
    return JSONResponse(content=response.model_dump())


    output = JSONResponse(resp)
    return output