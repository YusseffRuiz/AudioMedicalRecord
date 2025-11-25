## 📄 **README — Sistema de Captura Clínica con STT + LLM + Regex**

Este proyecto implementa un módulo híbrido de captura automática de datos clínicos, pensado para integrarse con un sistema de historial clínico en una página web.
El objetivo es automatizar la obtención de datos a partir de dictado médico en español, usando:

Grabación de audio (streaming o sesión completa)

Transcripción con Whisper

Normalización / reescritura con LLM local (LLaMA)

Extracción clínica robusta con regex médicas + validación de rangos

Construcción final de un JSON clínico listo para autollenado

## 🚀 Arquitectura General

            ┌──────────────────────────────┐                        
            │       Estación Médica        │                        
            │        PC + Micrófono        │                        
            │         Llamada API          │                        
            │                              │                        
            └──────────────┬───────────────┘                        
                           │    Audio se almacena                   
                           ▼    temporalmente en la PC              
         ┌────────────────────────────────────┐                     
         │           AWS Server Call          │                     
         │         Audio Medical API          │                     
         │        Fast API + Unicorn          │                     
         └─────────────────┬──────────────────┘                     
                           │                                        
                           │  Entrada de Audio                      
                           │  patient_id, session_id                
                           │                                        
                           ▼                                        
     ┌──────────────────────────────────────────────┐               
     │             Process Audio API                │               
     │                                              │               
     │         ASR-Engine (Fast-Whisper en CPU)     │               
     │                (Open Source)                 │               
     │                                              │               
     │            [Transcripción de audio a texto]  │               
     │                                              │               
     │       Regex Obtiene los datos que puede      │               
     │                                              │               
     │                                              │               
     │        LLM basado en LLAMA extrae            │               
     │           campos restantes                   │               
     │                                              │               
     │      Con los datos del LLM, llenar campos    │               
     │             Medical Clinical Field           │               
     │                                              │               
     └──────────────────────┬───────────────────────┘               
                            │    Salida: JSON con datos             
                            │                                       
                            │                                       
                            ▼                                       
    ┌────────────────────────────────────────────────────────────────┐
    │   Respuesta HTTP JSON                                          │
    │   {                                                            │
    │     "patient_id": ...,                                         │
    │     "session_id": ...,                                         │
    │     "transcript": transcript_full,                             │
    │     "clinical_fields": fields_final,                           │
    │     "missing_fields": missing_final,                           │
    │     "meta": {                                                  │
    │        "whisper_model": "medium",                              │
    │        "device": "cpu",                                        │
    │        "asr_confidence": ...,                                  │
    │        "processing_ms": ...                                    │
    │     }                                                          │
    │   }                                                            │
    └────────────────────────────────────────────────────────────────┘




**1. Captura de Audio**

Existen dos modos:

* Modo 1 — Live Streaming (Push-to-Talk)

Captura continua por fragmentos (0.5 s).

Se implementa mediante callback (audio_callback) y ventana deslizante.

Whisper transcribe segmentos nuevos y se acumulan en TRANSCRIPT_LOG.

* Modo 2 — Sesión Completa

Con AudioRecorder se graba toda la consulta (5–15 min).

Audio completo → .wav → Whisper para transcripción final.

Útil cuando el médico prefiere no interactuar durante la consulta.

**2. Transcripción — ASREngine.py**

Wrapper simple sobre Whisper:

Carga modelos tiny/base/small/medium/large-v3.

Soporta arrays de audio o rutas .wav.

Retorna: texto, confianza_estimada, diccionario_raw.

Se usa para ambos modos (streaming y sesión completa).

**3. Procesamiento del lenguaje**

* LLM local — FieldCompleterEngine.py

* Usa modelos tipo LLaMA vía CTransformers.
 
* Divide el transcript completo en chunks de ~600–800 tokens.

* Cada chunk se envía al LLM con un prompt clínico.
 
* El LLM produce texto organizado, no JSON.

* El texto es alimentado al ClinicalFormFiller.
 
* Puede completar campos faltantes basándose en el contexto chunk por chunk.
 
IMPORTANTE:
El JSON clínico NO lo genera el LLM.
Lo generamos nosotros con el extractor de regex.

4. Extracción de Datos Médicos — medical_filler.py
* Recibe texto (línea por línea o completo).
 
* Identifica valores mediante regex clínicas diseñadas para dictado real:


    Edad
    
    Peso (kg)
    
    Talla (m o cm → conversión a metros)
    
    Tensión arterial (incluye “sobre”)
    
    Frecuencia cardíaca
    
    Frecuencia respiratoria
    
    SpO₂
    
    Temperatura
    
    Glucosa
    
    Alergias

* Realiza validación fisiológica:


    Ej: Temp between 30–45 °C, TA entre 60–260/30–160, SpO₂ 50–100…

* Calcula derivados:


    IMC = peso / talla²
    
    TAM = (sis + 2·dia) / 3

* Mantiene el estado en un dataclass: ClinicalFields.

* Permite snapshot (dict) y previsualización legible (preview_text()).
 
* El resultado se guarda como JSON.

🧠 5. Flujo final de análisis

* Whisper → obtiene texto crudo.
* LLM (FieldCompleterEngine) → reescribe texto clínico de forma organizada.
* ClinicalFormFiller → extrae valores usando regex + validación.
* Se genera un JSON con:
  * Campos clínicos validados
  * Transcript original
  * Timestamp
  * Advertencia de faltantes:


    Por ejemplo: "Faltan: talla, FR, alergias"

El médico debe llenar esos campos manualmente.


## **🧪 Flujos disponibles**

1. A. Live Streaming
   1. Push-to-talk con ENTER
   2. Fin con Ctrl+C
   3. Uso ideal para consultas cortas o instrucciones rápidas

2. B. Sesión Completa
   1. En streaming_stt.main() existe el menú:
   2. Escribe [1] para grabar sesión completa
   3. Escribe [2] para live streaming


Sesión completa:
1.  Graba toda la consulta
2. Transcribe
3. Envía a LLM
4. Regex → JSON final

📦 Salida del sistema

Cada consulta genera en _historiales/ un archivo:

historial_YYYYMMDD_HHMMSS.json


Ejemplo:

    {
      "timestamp": "2025-02-11T15:42:10",
      "fields": {
        "edad": 33,
        "peso_kg": 83,
        "talla_m": 1.76,
        "imc": 26.8,
        "ta_sis": 120,
        "ta_dia": 80,
        "tam_map": 93,
        "fc_lpm": 80,
        "fr_rpm": 16,
        "spo2_pct": 86,
        "temp_c": 36.5,
        "gluc_mgdl": 40,
        "alergias": "Ninguna",
        "diagnostico": null,
        "receta": null
      },
      "extras": {
        "transcript": "texto dictado…"
      }
    }

## **Próximos pasos:**

* Afinar prompt del LLM con ejemplos reales
* Integrar con navegador (autollenado con Selenium/Playwright/Extension)
* Probar con médicos reales
* Ajustar regex según estilo de dictado real


**🙋 ¿Dudas o contribuciones?**

Este proyecto está diseñado para crecer hacia:

Sistemas FHIR

Integración con dispositivos médicos

Alertas clínicas

Diagnóstico asistido

Si vas a contribuir, revisa primero el flujo del streaming y el ClinicalFormFiller, ya que son los módulos más importantes.

