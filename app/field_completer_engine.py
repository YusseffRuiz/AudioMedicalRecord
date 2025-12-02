# Uso: complemento para terminar de poblar campos faltantes con un LLM, al FINAL de la transcripción.
# - Separa el transcript en chunks ~800 tokens (configurable).
# - Envía cada chunk a un LLM con un prompt de extracción JSON estricto.
# - Valida rangos y fusiona resultados.
import os
from typing import Dict, Any, List
import tiktoken
# import torch
from langchain_community.llms.ctransformers import CTransformers

# ---------------- Configuración de campos y rangos plausibles ----------------
REQUIRED_FIELDS = [
    "edad", "peso_kg", "talla_m", "ta_sis", "ta_dia",
    "fc_lpm", "fr_rpm", "spo2_pct", "temp_c", "gluc_mgdl", "alergias",
    # Posterior al llenado de signos, el flujo incluye:
    "diagnostico", "receta"
]

FIELD_META = {
    "edad":       {
                    "label": "Edad",
                    "desc": "edad del paciente en años (entero)",
                    "example": "La edad del paciente es de 33 años."
                    },
    "peso_kg":    {"label": "Peso",
                    "desc": "peso en kilogramos (float, con punto decimal si hace falta)",
                    "example": "El peso es de 83.0 kilogramos.",
                   },
    "talla_m":    {"label": "Talla",
                    "desc": "talla/altura en metros (float, por ejemplo 1.76)",
                    "example": "La talla es de 1.76 metros.",
                   },
    "t_a":        {"label": "Tensión arterial",
                    "desc": "tensión arterial en mmHg (entero en forma X,Y, mmHg)",
                    "example": "La presión arterial es de 120 sobre 80",
                  },

    "fc_lpm":     {"label": "Frecuencia cardíaca",
                    "desc": "frecuencia cardiaca en latidos por minuto (entero)",
                    "example": "La frecuencia cardiaca es de 80 latidos por minuto.",
                   },
    "fr_rpm":     {"label": "Frecuencia respiratoria",
                    "desc": "frecuencia respiratoria en respiraciones por minuto (entero)",
                    "example": "La frecuencia respiratoria es de 16 respiraciones por minuto.",
                   },
    "spo2_pct":   {"label": "SpO₂",
                    "desc": "saturación de oxígeno en porcentaje (entero, %)",
                    "example": "La saturación de oxígeno es de 97%.",
                   },
    "temp_c":     {"label": "Temperatura",
                    "desc": "temperatura corporal en °C (float)",
                    "example": "La temperatura corporal de 36.7 grados centígrados.",
                   },
    "gluc_mgdl":  {"label": "Glucosa",
                    "desc": "glucosa en sangre en mg/dL (float)",
                    "example": "La glucosa capilar en ayunas es de 90 miligramos por decilitro.",
                   },
    "alergias":   {"label": "Alergias",
                    "desc": "alergias relevantes en un texto corto",
                    "example": "Las alergias reportadas son: alergia a penicilina.",
                   },
    "diagnostico":{"label": "Diagnóstico",
                    "desc": "diagnóstico clínico principal en una frase",
                    "example": "El diagnóstico es cefalea tensional aguda.",
                   },
    "receta":     {"label": "Receta",
                    "desc": "tratamiento o receta indicada en pocas frases",
                    "example": "La receta indicada es ibuprofeno 400 mg cada 8 horas por 3 días.",
                   },
}

FIELD_LABELS = { ## Solo para mostrar en formato humano.
    "edad": "Edad",
    "peso_kg": "Peso",
    "talla_m": "Talla",
    "ta_sis": "Tensión arterial",
    "ta_dia": "Tensión arterial",
    "fc_lpm": "Frecuencia cardiaca",
    "fr_rpm": "Frecuencia respiratoria",
    "spo2_pct": "SpO2",
    "temp_c": "Temperatura",
    "gluc_mgdl": "Glucosa",
    "alergias": "Alergias",
    "diagnostico": "Diagnóstico",
    "receta": "Receta",
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
                 # device="cuda" if torch.cuda.is_available() else "cpu"):
                 device="cpu"):
        self.max_tokens = max_tokens_per_chunk
        self.overlap_tokens = overlap_tokens
        self.initial_prompt = None
        self.medical_filler = medical_filler


        # Verificar GPU solo si hay GPU, descomentar.
        # cuda_available = torch.cuda.is_available()
        cuda_available = False
        # print("CUDA available:", cuda_available)

        # print("Initializing Model ...")
        gpu_layers = 0
        config = {'max_new_tokens': 140, 'context_length': 1900, 'temperature': 0.35, "gpu_layers": gpu_layers,
                  "threads": os.cpu_count()}
        if cuda_available:
            gpu_layers = 16
            config = {'max_new_tokens': 140, 'context_length': 1900, 'temperature': 0.35, "gpu_layers": gpu_layers,
                      "threads": os.cpu_count()}

        # self.llm_model = None
        self.llm_model = CTransformers(
            model=model_name,
            model_type="llama",
            config=config,
            verbose=False,
        )
        print("Module Created!")

    def initialize(self, initial_prompt):
        # Se usa cuando se necesita crear un prompt inicial.
        self.initial_prompt = initial_prompt

    def build_llama2_prompt(self, context: str) -> str:
        # Plantilla oficial LLaMA-2 chat, uso basico para crear solo un prompt inicial
        return (
            f"[INST] <<SYS>>\n{self.initial_prompt}\n<</SYS>>\n\n"
            f"# CONTEXTO\n{context}\n\n"
            "# PREGUNTA\nExtrae los campos desde el contexto en el formato especificado. Si no hay ninguno, devuelve {{}}.\n[/INST]"
        )

    @staticmethod
    def _in_range(key: str, value) -> bool:
        # Verifica que los valores obtenidos si esten en rango real
        if key not in RANGES:
            return True
        lo, hi = RANGES[key]
        v = float(value)
        return lo <= v <= hi


    def _extract_from_chunk(self, chunk_text_str: str, missing_fields=None):

        # prompt = self.build_llama2_prompt(chunk_text_str)
        prompt = self.build_prompt_for_missing_fields(chunk_text_str, missing_fields=missing_fields)
        # print("Tokens context: ", count_tokens(chunk_text_str))
        # print("Tokens prompt: ", count_tokens(prompt))
        error_cnt = 0
        success = False
        raw = None
        while not success:
            try:
                raw = self.llm_model.invoke(prompt)
            except Exception as e:
                print("[LLM ERROR]", repr(e))
                error_cnt += 1
            if raw is not None:
                success = True
            if error_cnt >= 5:
                success = True
                print("LLM Failed")
        # Habilitar para debugging de la respuesta del LLM
        # print("Respuesta LLM: ")
        # print(raw)
        lines = raw.strip().splitlines()
        for line in lines:
            self.medical_filler.update(line)

    def complete_fields(self, transcript: str, missing_fields: list[str]):
        """
        Construye un prompt para el LLM usando SOLO los campos que faltan.
        """
        if not missing_fields:
            return ""

        chunks = chunk_text(transcript, max_tokens=self.max_tokens, overlap_tokens=self.overlap_tokens)
        for ch in chunks:
            # print(ch) # solo para debugging de analisis del texto transcrito
            self._extract_from_chunk(ch, missing_fields)

    @staticmethod
    def build_prompt_for_missing_fields(transcript: str, missing_fields: list[str]) -> str:
        """
        Construye un prompt LLaMA2-style usando SOLO los campos faltantes.
        - transcript: texto completo de la consulta.
        - missing_fields: lista de keys internas que faltan (ej: ["edad", "peso_kg", "diagnostico"]).

        El modelo debe responder con frases simples, una por campo, para
        que luego regex pueda extraer los datos.
        """
        if not missing_fields:
            # Idealmente no se llamaría a esta función en ese caso,
            # pero por seguridad regresamos algo corto.
            return (
                "[INST] Solo responde con el texto: \"Sin campos faltantes\". [/INST]"
            )

        # 1) Construir descripción dinámica de campos faltantes
        lines_desc = []

        for key in missing_fields:
            meta = FIELD_META.get(key)
            if not meta:
                continue
            # lines_desc.append(f"- {meta['label']} : {meta['desc']} - Ejemplo: {meta['example']}")
            lines_desc.append(f"- {meta['label']}")

        campos_descripcion = "\n".join(lines_desc)

        # 2) Instrucciones base (puedes ajustar para aligerar tokens si hace falta)
        sys_instructions = (
            "Eres un extractor clínico muy estricto en español.\n"
            "Tu única tarea es leer la transcripción de una consulta médica y escribir "
            "solo los campos clínicos que se te pidan, en el formato EXACTO:\n"
            "Campo: valor\n\n"
            "Reglas OBLIGATORIAS:\n"
            "1) Responde solo con una línea por campo faltante.\n"
            "2) Cada línea debe seguir exactamente el formato: 'Campo: valor'.\n"
            "3) NO uses viñetas, NO uses comillas, NO expliques nada, NO agregues texto antes ni después.\n"
            "4) Usa exactamente los nombres de campo indicados (misma ortografía y mayúsculas).\n"
            "5) No escribas saludos, despedidas, preguntas ni texto de conversación.\n"
            "6) Si un campo no se puede deducir con certeza del contexto, NO lo escribas.\n"
            "7) Para campos numéricos usa números árabes y punto decimal cuando aplique (ej. 36.5).\n"
            "8) Para diagnóstico y receta usa una sola oración corta, sin copiar diálogos textuales.\n"
            "9) No incluyas frases como '¿Con esto terminamos?', '¿Alguna duda?', 'Muchas gracias', etc.\n"
            "10) No repitas la misma información en más de un campo.\n\n"
            "Campos faltantes (solo estos te interesan):\n"
            f"{campos_descripcion}\n"
        )

        # 3) Ejemplos orientados a regex
        ejemplos = (
            "Ejemplos de formato EXACTO (no agregues nada más):\n"
            "Edad: 33\n"
            "Peso: 82.0 kg\n"
            "Talla: 1.76 m\n"
            "Tensión arterial: 120, 80 mmHg\n"
            "Frecuencia cardíaca: 80 lpm\n"
            "Frecuencia respiratoria: 16 rpm\n"
            "SpO₂: 97 %\n"
            "Temperatura: 36.7 grados\n"
            "Glucosa: 90 mg/dL\n"
            "Alergias: ninguna\n"
            "Diagnóstico: cefalea tensional aguda\n"
            "Receta: ibuprofeno 400 mg cada 8 horas por 3 días\n\n"
            "Si en el texto no hay información suficiente para algún campo, NO escribas esa línea.\n"
        )

        # 4) Construir el INST con contexto
        campos_str = ", ".join(
            FIELD_META[k]["label"] for k in missing_fields if k in FIELD_META
        )

        prompt = (
            "[INST] <<SYS>>\n"
            f"{sys_instructions}\n"
            f"{ejemplos}\n"
            "<</SYS>>\n"
            "# CONTEXTO CLÍNICO\n"
            f"{transcript}\n"
            "# TAREA\n"
            f"Con base en el contexto clínico anterior, escribe SOLO las líneas de los campos faltantes: {campos_str}.\n"
            "Respeta estrictamente el formato 'Campo: valor' para cada línea. "
            "No escribas nada fuera de esas líneas.\n"
            "[/INST]"
        )
        # print("Tokens: ", count_tokens(prompt))

        return prompt

    @staticmethod
    def compute_missing(fields: dict) -> list[str]:
        """
        Calcula qué campos siguen vacíos después de una pasada de regex.
        Excluimos campos derivados como imc y tam_map.
        """
        return [
            k for k in REQUIRED_FIELDS
            if fields.get(k) in (None, "", 0) and k not in {"imc", "tam_map"}
        ]