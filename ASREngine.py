from typing import List, Dict, Tuple
import whisper
import numpy as np


# Patrones clínicos mínimos de ejemplo (puedes ampliarlos luego)
MED_PATTERNS = [
(r"(\bno\b\s+(fiebre|dolor|alergias?))", "negacion"),
(r"(\b\d{2,3}\/\d{2,3}\b)", "presion_arterial"),
(r"(\b\d{2,3}\.?\d?\s?°?C\b)", "temperatura"),
]


class AsrEngine:
    def __init__(self, model_size: str = "small", device: str = "cpu") -> None:
        """
        model_size: "tiny" | "base" | "small" | "medium" | "large-v3"
        device: "cpu" | "cuda"
        """
        # compute_type "int8_float16" funciona bien en CPU modernas; en GPU puedes usar "float16"
        self.model = whisper.load_model(model_size, device=device)


    def transcribe_file(self, audio: str, language: str = "es", fp16=False, without_timestamps=True
                        ) -> Tuple[str, float, dict]:

        # audio es un array
        result = self.model.transcribe(
            audio,
            language=language,
            fp16=fp16,
            without_timestamps=without_timestamps,
        )

        text = result.get("text", "")
        confidence = 0.90 if text else 0.0 # placeholder; faster-whisper no expone prob estable
        return text, confidence, result