import math
import os
# import whisper
from faster_whisper import WhisperModel


# Fuerza a CTranslate2 a ignorar CUDA incluso si está compilado con soporte GPU
os.environ["CT2_FORCE_CPU"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "4")  # opcional para no volarte todos los cores

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
        # self.model = whisper.load_model(model_size, device=device)
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")


    def transcribe_file(self, audio: str, language: str = "es", fp16=False, without_timestamps=True
                        ):

        # audio es un array
        # result = self.model.transcribe(
        #     audio,
        #     language=language,
        #     fp16=fp16,
        #     without_timestamps=without_timestamps,
        # )
        segments, info = self.model.transcribe(audio,
            language=language,
            without_timestamps=without_timestamps, beam_size=5, best_of=1,)


        # Materializar el generador
        segments_list = list(segments)
        # Concatenar texto
        text = "".join(seg.text.strip() for seg in segments_list)

        # Confianza aproximada a partir de avg_logprob
        logps = [seg.avg_logprob for seg in segments if seg.avg_logprob is not None]

        if logps:
            avg_logp = sum(logps) / len(logps)
            # pasar de log-probabilidad a algo tipo [0,1]
            confidence = float(math.exp(avg_logp))
        else:
            confidence = 0.0

        # print(text)
        # print(confidence)
        # text = result.get("text", "")
        # confidence = 0.90 if text else 0.0 # placeholder; faster-whisper no expone prob estable
        return text, confidence, None