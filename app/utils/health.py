import os
from typing import Dict, Any


def check_mistral(mistral_engine) -> Dict[str, Any]:
    try:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            return {"ok": False, "detail": "MISTRAL_API_KEY not set"}

        if mistral_engine is None:
            return {"ok": False, "detail": "mistral engine is None"}

        # Versión ligera: solo revisar que el cliente está construido
        return {"ok": True, "detail": "initialized"}
    except Exception as e:
        return {"ok": False, "detail": f"exception in mistral: {e}"}


def check_asr(asr_engine) -> Dict[str, Any]:
    try:
        ok = hasattr(asr_engine, "model") and asr_engine.model is not None
        return {"ok": bool(ok), "detail": "loaded" if ok else "model is None"}
    except Exception as e:
        return {"ok": False, "detail": f"exception: {e}"}
