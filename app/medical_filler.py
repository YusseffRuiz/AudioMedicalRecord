import datetime
import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict
import re


@dataclass
class ClinicalFields:
    # Demográficos
    edad: Optional[int] = None
    # Signos vitales / medidas
    peso_kg: Optional[float] = None
    talla_m: Optional[float] = None
    imc: Optional[float] = None
    ta_sis: Optional[int] = None
    ta_dia: Optional[int] = None
    tam_map: Optional[float] = None
    fc_lpm: Optional[int] = None
    fr_rpm: Optional[int] = None
    spo2_pct: Optional[int] = None
    temp_c: Optional[float] = None
    gluc_mgdl: Optional[float] = None
    alergias: Optional[str] = None
    # Flujo posterior
    diagnostico: Optional[str] = None
    receta: Optional[str] = None

class ClinicalFormFiller:
    """
    Extrae campos clínicos desde frases en español (dictado libre)
    y mantiene un estado acumulado. Idempotente: puedes llamar update()
    con texto incremental y sólo rellenará lo que falte o actualizará coherentemente.
    """
    # Patrones (sensibles a español clínico común)
    # Nota: se prioriza robustez/legibilidad; ajusta a tus frases reales luego.
    _re_edad = re.compile(r"\b(?:edad\s*[:\-]?\s*)?(\d{1,3})\s*(?:años?|año)\b", re.I)
    _re_peso = re.compile(r"\b(?:peso\s*[:\-]?\s*)?(\d{1,3}(?:[.,]\d)?)\s*(?:kg|kilogramos?)\b", re.I)
    _re_talla_m = re.compile(r"\b(?:(?:talla|altura|estatura)\s*[:\-]?\s*)?(?P<val>\d(?:[.,]\d{1,3})?)\s*(m|metros|mt)\b",re.I)
    _re_talla_cm = re.compile(r"\b(?:(?:talla|altura|estatura)\s*[:\-]?\s*)?(?P<val>\d(?:[.,]\d{1,3})?)\s*cm\b",re.I)
    _re_ta = re.compile(
        r"\b(?:T/?A|TA|[tT]ensión\s*[aA]rterial)?\s*"  # 
        r"(?:[:\-]?\s*)?"  # separador 
        r"(?:es\s+de\s+|de\s+|en\s+)?"  # conector opcional: es de / de / en
        r"(\d{2,3})\s*"  # sistólica
        r"(?:[/\s-]|sobre)\s*"  # separador: /, espacio o 'sobre'
        r"(\d{2,3})\b",  # diastólica
        re.I
    )
    _re_fc = re.compile(r"\b(?:FC|frecuencia\s*card[ií]aca)\s*(?:[:\-]?\s*)?(\d{1,3})\s*(?:lpm|bpm)?\b",
    re.I)
    _re_fr = re.compile(r"\b(?:FR|frecuencia\s*respiratoria)\s*(?:[:\-]?\s*)?(\d{1,2})\s*(?:rpm|respiraciones\s*por\s*minuto)?\b",
    re.I)
    _re_spo2 = re.compile(
        r"\b(?:SpO₂|SpO2|Sp02|saturaci(?:o|ó)n(?:\s*de\s*ox[ií]geno)?"  # SpO2 / saturación / saturación de oxígeno
        r"|ox[ií]geno(?:\s*en\s*la\s*sangre)?)\s*"  # oxígeno / oxígeno en sangre
        r"(?:[:\-]?\s*)?"  # separador opcional
        r"(\d{2,3})\s*"  # valor 2–3 dígitos
        r"(?:%|por\s*ciento)?\b",  # % o 'por ciento'
        re.I
    )
    _re_temp = re.compile(
        r"\b(?:temp(?:eratura)?(?:\s*corporal)?)\s*"  # temp / temperatura / temperatura corporal
        r"(?:[:\-|]?\s*)?"  # separador
        r"(\d{2,3}(?:[.,]\d+)?)\s*"  # número 2–3 dígitos + opcional decimal
        r"(?:°\s*)?"  # símbolo ° opcional
        r"(?:C|cent(?:í|i)grados?|centigrados)?\b",  # C / centígrados / centigrados (opc.)
        re.I
    )
    _re_gluc = re.compile(r"\b(?:gluc(?:osa)?|glucemia)\s*(?:[:\-]?\s*)?(?:es\s+de\s+|de\s+|en\s+)?(\d{1,3}(?:[.,]\d+)?)\s*(?:mg/?dL|mgdL)?\b",
    re.I)
    _re_alerg_none = re.compile(r"\b(no\s+(alergias?|al(?:e|é)rgico))\b", re.I)
    _re_alerg = re.compile(
        r"\b(?:alerg(?:ias?)?|al(?:e|é)rgico(?:a)?)\s*(?:a|:)?\s*[:\-]?\s*(?P<val>.+)$"
        , re.I)
    _re_diag = re.compile(
        r"\b(?:diagn[oó]stico(?:\s+principal)?|impresi[oó]n\s+diagn[oó]stica)\s*[:\-]?\s*(?P<val>.+)$",
        re.I
    )
    _re_receta = re.compile(
        r"\b(?:[Rr]eceta|[Mm]edicamento?:s)\s*[:\-]?\s*(?P<val>.+)$",
        re.I
    )
    NO_CAND_RE = r"\b(no\s+se\s+especific[oó]|no\s+se\s+inform[oó]|no\s+se\s+menciona)\b"

    # Diagnóstico y receta se capturarán con métodos explícitos
    def __init__(self):
        self.state = ClinicalFields()

    def update(self, text: str, reg_flag : bool =True) -> Dict[str, object]:
        """Procesa texto incremental y actualiza el estado. Devuelve sólo los campos que cambiaron."""
        changed: Dict[str, object] = {}
        s = text.strip()

        # EDAD
        m = self._re_edad.search(s)
        if m:
            edad = int(m.group(1))
            if 0 < edad < 120 and self.state.edad != edad:
                self.state.edad = edad
                changed["edad"] = edad

        # PESO
        m = self._re_peso.search(s)
        if m:
            kg = float(m.group(1))
            if 1 <= kg <= 400 and self.state.peso_kg != kg:
                self.state.peso_kg = kg
                changed["peso_kg"] = kg

        # TALLA
        m = self._re_talla_m.search(s)
        talla_m = None
        if m:
            talla_m = float(m.group("val"))
        else:
            m = self._re_talla_cm.search(s)
            if m:
                cm = float(m.group("val"))
                talla_m = cm / 100.0

        if talla_m and 0.5 <= talla_m <= 2.5 and self.state.talla_m != talla_m:
            self.state.talla_m = talla_m
            changed["talla_m"] = talla_m

        # IMC (si hay peso y talla)
        if self.state.peso_kg and self.state.talla_m:
            imc_calc = round(self.state.peso_kg / (self.state.talla_m ** 2), 1)
            if self.state.imc != imc_calc:
                self.state.imc = imc_calc
                changed["imc"] = imc_calc

        # TA
        m = self._re_ta.search(s)
        if m:
            sis, dia = int(m.group(1)), int(m.group(2))
            if 60 <= sis <= 260 and 30 <= dia <= 160:
                if self.state.ta_sis != sis:
                    self.state.ta_sis = sis
                    changed["ta_sis"] = sis
                if self.state.ta_dia != dia:
                    self.state.ta_dia = dia
                    changed["ta_dia"] = dia
                # TAM (MAP)
                tam = round((sis + 2 * dia) / 3.0)
                if self.state.tam_map != tam:
                    self.state.tam_map = tam
                    changed["tam_map"] = tam

        # FC
        m = self._re_fc.search(s)
        if m:
            fc = int(m.group(1))
            if 20 <= fc <= 250 and self.state.fc_lpm != fc:
                self.state.fc_lpm = fc
                changed["fc_lpm"] = fc

        # FR
        m = self._re_fr.search(s)
        if m:
            fr = int(m.group(1))
            if 5 <= fr <= 80 and self.state.fr_rpm != fr:
                self.state.fr_rpm = fr
                changed["fr_rpm"] = fr

        # SpO2
        m = self._re_spo2.search(s)
        if m:
            spo2 = int(m.group(1))
            if 50 <= spo2 <= 100 and self.state.spo2_pct != spo2:
                self.state.spo2_pct = spo2
                changed["spo2_pct"] = spo2

        # Temperatura
        m = self._re_temp.search(s)
        if m:
            temp = float(m.group(1))
            # Filtra valores plausibles humanos
            if 30.0 <= temp <= 45.0 and self.state.temp_c != temp:
                self.state.temp_c = temp
                changed["temp_c"] = temp

        # Glucosa
        m = self._re_gluc.search(s)
        if m:
            gluc = int(m.group(1))
            if 20.0 <= gluc <= 600.0 and self.state.gluc_mgdl != gluc:
                self.state.gluc_mgdl = gluc
                changed["gluc_mgdl"] = gluc

        if not reg_flag:
            # Diagnostico
            m = self._re_diag.search(s)
            if m:
                cand = m.group("val").strip(" .,-\"")
                # Si ya hay algo, une de forma única
                if cand and not re.search(self.NO_CAND_RE, cand, re.I):
                    if not self.state.diagnostico:
                        self.state.diagnostico = cand
                        changed["diagnostico"] = cand
                    elif cand.lower() not in self.state.diagnostico.lower():
                        self.state.diagnostico = f"{self.state.diagnostico}, {cand}"
                        changed["diagnostico"] = self.state.diagnostico
            # Receta
            m = self._re_receta.search(s)
            if m:
                cand = m.group("val").strip(" .,-\"")
                # Si ya hay algo, une de forma única
                if cand and not re.search(self.NO_CAND_RE, cand, re.I):
                    if not self.state.receta:
                        self.state.receta = cand
                        changed["receta"] = cand
                    elif cand.lower() not in self.state.receta.lower():
                        self.state.receta = f"{self.state.receta}, {cand}"
                        changed["receta"] = self.state.receta

            # Alergias
            if self._re_alerg_none.search(s):
                if self.state.alergias != "Ninguna":
                    self.state.alergias = "Ninguna"
                    changed["alergias"] = "Ninguna"
            else:
                m = self._re_alerg.search(s)
                if m:
                    cand = m.group("val").strip(" .,-\"")
                    # Si ya hay algo, une de forma única
                    if cand and not re.search(self.NO_CAND_RE, cand, re.I):
                        if not self.state.alergias:
                            self.state.alergias = cand
                            changed["alergias"] = cand
                        elif cand.lower() not in self.state.alergias.lower():
                            self.state.alergias = f"{self.state.alergias}, {cand}"
                            changed["alergias"] = self.state.alergias

        return changed

    # Métodos explícitos para diagnóstico y receta (se llenarán cuando el médico lo indique)
    def set_diagnostico(self, texto: str):
        self.state.diagnostico = texto.strip()

    def set_receta(self, texto: str):
        self.state.receta = texto.strip()

    def snapshot(self) -> Dict[str, object]:
        """Estado completo (dict)."""
        return asdict(self.state)

    def preview_text(self) -> str:
        """Representación legible (para consola)."""
        s = self.state
        lines = [
            f"Edad: {s.edad or ''}",
            f"Peso: {s.peso_kg or ''} kg",
            f"Talla: {s.talla_m or ''} m",
            f"IMC: {s.imc or ''}",
            f"T/A: {s.ta_sis or ''}/{s.ta_dia or ''} mmHg",
            f"TAM: {s.tam_map or ''} mmHg",
            f"FC: {s.fc_lpm or ''} lpm",
            f"FR: {s.fr_rpm or ''} rpm",
            f"SpO2: {s.spo2_pct or ''} %",
            f"Temp: {s.temp_c or ''} °C",
            f"Gluc: {s.gluc_mgdl or ''} mg/dL",
            f"Alergias: {s.alergias or ''}",
            f"Diagnóstico: {s.diagnostico or ''}",
            f"Receta: {s.receta or ''}",
        ]
        return " | ".join(lines)

    def save_json(self, path: str, extras: dict | None = None):
        """Guarda el estado actual a un JSON (con timestamp y extras opcionales)."""
        payload = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "fields": self.snapshot(),  # lo que ya tienes normalizado
        }
        if extras:
            payload["extras"] = extras
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def reset_state(self):\
        self.state = ClinicalFields()

    def extract_with_regex(self, text: str) -> dict:
        # 1) procesar línea por línea (útil si el texto viene pseudo-estructurado)
        for line in text.splitlines():
            self.update(line)

        # 2) pasada global
        self.update(text)

        return self.snapshot()