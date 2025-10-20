import datetime
import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple, List
import re

REQUIRED_MEDICAL_FIELDS = {
    "edad",
    "peso",
    "talla",
    "imc",
    "ta",
    "tam",
    "fc",
    "fr",
    "spo2"
    "temperatura",
    "glucosa",
    "alergias",
    "diagnostico",
    "receta",

}

def _norm_num(s: str) -> float:
    # Permite decimales con coma o punto
    s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return float("nan")

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
    _re_talla_m = re.compile(r"\b(?:(talla|altura|estatura)\s*[:\-]?\s*)?(\d(?:[.,]\d{1,2})?)\s*m\b", re.I)
    _re_talla_cm = re.compile(r"\b(?:talla\s*[:\-]?\s*)?(\d{2,3})\s*cm\b", re.I)
    _re_ta = re.compile(r"\b(?:T/?A|TA|tensión\s*arterial)\s*[:\-]?\s*(\d{2,3})\s*[\/\s]\s*(\d{2,3})\b", re.I)
    _re_fc = re.compile(r"\b(?:FC|frecuencia\s*card(?:i|í)aca)\s*[:\-]?\s*(\d{2,3})\s*(?:lpm|bpm)?\b", re.I)
    _re_fr = re.compile(r"\b(?:FR|frecuencia\s*respiratoria)\s*[:\-]?\s*(\d{1,2})\s*(?:rpm)?\b", re.I)
    _re_spo2 = re.compile(r"\b(?:SpO2|saturaci(?:o|ó)n)\s*[:\-]?\s*(\d{2,3})\s*%\b", re.I)
    _re_temp = re.compile(r"\b(?:temp|temperatura)\s*[:\-]?\s*(\d{2,3}(?:[.,]\d)?)\s*(?:°?\s*C|°?C|C)?\b", re.I)
    _re_gluc = re.compile(r"\b(?:gluc(?:osa)?)\s*[:\-]?\s*(\d{1,3}(?:[.,]\d)?)\s*(?:mg/?dL|mgdL)?\b", re.I)
    _re_alerg_none = re.compile(r"\b(no\s+(alergias?|al(?:e|é)rgico))\b", re.I)
    _re_alerg = re.compile(r"\b(?:alerg(?:ias?)?|al(?:e|é)rgico(?:a)?)\s*(?:a|:)?\s*([A-Za-zÁÉÍÓÚÑÜáéíóúñ0-9 ,\-]+)", re.I)

    # Diagnóstico y receta se capturarán con métodos explícitos
    def __init__(self):
        self.state = ClinicalFields()

    def update(self, text: str) -> Dict[str, object]:
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
            kg = _norm_num(m.group(1))
            if 1 <= kg <= 400 and self.state.peso_kg != kg:
                self.state.peso_kg = kg
                changed["peso_kg"] = kg

        # TALLA
        m = self._re_talla_m.search(s)
        talla_m = None
        if m:
            talla_m = _norm_num(m.group(1))
        else:
            m = self._re_talla_cm.search(s)
            if m:
                cm = _norm_num(m.group(1))
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
            temp = _norm_num(m.group(1))
            # Filtra valores plausibles humanos
            if 30.0 <= temp <= 45.0 and self.state.temp_c != temp:
                self.state.temp_c = temp
                changed["temp_c"] = temp

        # Glucosa
        m = self._re_gluc.search(s)
        if m:
            gluc = _norm_num(m.group(1))
            if 20.0 <= gluc <= 600.0 and self.state.gluc_mgdl != gluc:
                self.state.gluc_mgdl = gluc
                changed["gluc_mgdl"] = gluc

        # Alergias
        if self._re_alerg_none.search(s):
            if self.state.alergias != "Ninguna":
                self.state.alergias = "Ninguna"
                changed["alergias"] = "Ninguna"
        else:
            m = self._re_alerg.search(s)
            if m:
                cand = m.group(1).strip(" .,-")
                # Si ya hay algo, une de forma única
                if cand:
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
