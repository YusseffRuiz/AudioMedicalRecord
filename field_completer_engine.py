# Uso: complemento para terminar de poblar campos faltantes con un LLM, al FINAL de la transcripción.
# - Separa el transcript en chunks ~800 tokens (configurable).
# - Envía cada chunk a un LLM con un prompt de extracción JSON estricto.
# - Valida rangos y fusiona resultados.
import os
from typing import Callable, Dict, Any, List, Tuple
import tiktoken
import torch
from langchain_community.llms.ctransformers import CTransformers

# ---------------- Configuración de campos y rangos plausibles ----------------
REQUIRED_FIELDS = [
    "edad", "peso_kg", "talla_m", "imc", "ta_sis", "ta_dia", "tam_map",
    "fc_lpm", "fr_rpm", "spo2_pct", "temp_c", "gluc_mgdl", "alergias",
    # Posterior al llenado de signos, el flujo incluye:
    "diagnostico", "receta"
]

FIELD_META = {
    "edad":       {"label": "Edad (años)"},
    "peso_kg":    {"label": "Peso (kg)"},
    "talla_m":    {"label": "Talla (m)"},
    "imc":        {"label": "IMC"},
    "ta_sis":     {"label": "Tensión arterial sistólica (mmHg)"},
    "ta_dia":     {"label": "Tensión arterial diastólica (mmHg)"},
    "tam_map":    {"label": "Tensión arterial media (TAM, mmHg)"},
    "fc_lpm":     {"label": "Frecuencia cardíaca (lpm)"},
    "fr_rpm":     {"label": "Frecuencia respiratoria (rpm)"},
    "spo2_pct":   {"label": "SpO₂ (%)"},
    "temp_c":     {"label": "Temperatura (°C)"},
    "gluc_mgdl":  {"label": "Glucosa (mg/dL)"},
    "alergias":   {"label": "Alergias"},
    "diagnostico":{"label": "Diagnóstico"},
    "receta":     {"label": "Receta"},
}

RANGES = {
    "edad": (0, 120),
    "peso_kg": (1, 400),
    "talla_m": (0.5, 2.5),
    "imc": (8, 80),
    "ta_sis": (40, 260),
    "ta_dia": (30, 160),
    "tam_map": (40, 170),
    "fc_lpm": (20, 220),
    "fr_rpm": (5, 80),
    "spo2_pct": (50, 100),
    "temp_c": (30.0, 45.0),
    "gluc_mgdl": (20, 600)
}

# ---------------- Tokenizer helpers ----------------

def _try_tiktoken_encoding():
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None

_enc = _try_tiktoken_encoding()

def count_tokens(text: str) -> int:
    """Cuenta tokens usando tiktoken si está disponible; si no, aproxima por palabras."""
    if _enc is not None:
        try:
            return len(_enc.encode(text))
        except Exception:
            pass
    # fallback: aprox 1 token ≈ 0.75 palabras en español
    return int(len(text.split()) / 0.75) + 1


def chunk_text(text: str, max_tokens: int = 800, overlap_tokens: int = 50) -> List[str]:
    """Divide el texto en chunks de ~max_tokens con un pequeño overlap para contexto."""
    if not text:
        return []
    # Con tiktoken
    ids = _enc.encode(text)
    chunks = []
    i = 0
    while i < len(ids):
        part = ids[i:i+max_tokens]
        if not part:
            break
        chunks.append(_enc.decode(part))
        i += max_tokens - overlap_tokens
    return chunks


# ---------------- LLMFieldCompleter ----------------

class FieldCompleterEngine:
    """
    Completa campos faltantes con apoyo de un LLM de forma segura:
    - Separa transcript en chunks (tokens).
    - Pide SOLO JSON de campos presentes.
    - Valida y fusiona resultados sin sobreescribir valores válidos existentes.
    """
    def __init__(self,
                 model_name,
                 max_tokens_per_chunk: int = 800,
                 overlap_tokens: int = 50,
                 medical_filler = None,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.max_tokens = max_tokens_per_chunk
        self.overlap_tokens = overlap_tokens
        self.initial_prompt = None
        self.medical_filler = medical_filler


        # Verificar GPU
        cuda_available = torch.cuda.is_available()
        print("CUDA available:", cuda_available)

        print("Initializing Model ...")
        gpu_layers = 0
        if cuda_available:
            gpu_layers = 16
            config = {'max_new_tokens': 256, 'context_length': 1100, 'temperature': 0.35, "gpu_layers": gpu_layers,
                      "threads": os.cpu_count()}
        else:
            config = {'max_new_tokens': 256, 'context_length': 1100, 'temperature': 0.35}


        self.llm_model = CTransformers(
            model=model_name,
            model_type="llama",
            config=config,
            verbose=False,
            device=device,
            gpu_layers=gpu_layers,
        )
        print("Module Created!")

    def initialize(self, initial_prompt):
        self.initial_prompt = initial_prompt

    def build_llama2_prompt(self, context: str) -> str:
        # Plantilla oficial LLaMA-2 chat
        return (
            f"[INST] <<SYS>>\n{self.initial_prompt}\n<</SYS>>\n\n"
            f"# CONTEXTO\n{context}\n\n"
            f"# PREGUNTA\n{'Extrae los campos desde el contexto'
                           'Si no hay ninguno, devuelve {}.'}\n"
            "[/INST]"
        )

    @staticmethod
    def _in_range(key: str, value) -> bool:
        # Verifica que los valores obtenidos si esten en rango real
        if key not in RANGES:
            return True
        lo, hi = RANGES[key]
        v = float(value)
        return lo <= v <= hi


    def _extract_from_chunk(self, chunk_text_str: str):
        prompt = self.build_llama2_prompt(chunk_text_str)
        error_cnt = 0
        success = False
        while not success:
            raw = self.llm_model.invoke(prompt)
            error_cnt += 1
            if raw is not None:
                success = True
            if error_cnt >= 10:
                success = True
                print("LLM Failed")
        print("Respuesta LLM: ")
        print(raw)
        print("#################")
        lines = raw.strip().splitlines()
        for line in lines:
            print(line)
            self.medical_filler.update(line)

    def complete_fields(self, transcript: str, current_state: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Devuelve (updates, missing_after):
        - updates: dict con nuevos campos válidos a aplicar sobre current_state
        - missing_after: lista de keys que siguieron faltando tras LLM
        """
        # Determinar faltantes iniciales
        missing = [k for k in REQUIRED_FIELDS if current_state.get(k) in (None, "", 0) and k not in {"imc", "tam_map"}]
        if not missing:
            return {}, []

        chunks = chunk_text(transcript, max_tokens=self.max_tokens, overlap_tokens=self.overlap_tokens)

        for ch in chunks:
            self._extract_from_chunk(ch)