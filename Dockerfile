FROM python:3.11-slim
LABEL authors="AdanYDR"

# 1. Dependencias de sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


# 2. Dependencias Python
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN python -m pip install --default-timeout=120 --no-cache-dir -r requirements.txt

# 3. Código
COPY app/ ./app
COPY models/ ./models

# 4. Config por defecto
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    WHISPER_MODEL=medium \
    DEVICE=cpu \
    CT2_FORCE_CPU=1 \
    OMP_NUM_THREADS=4

# 5. Puerto expuesto
EXPOSE 8000

# 6. Comando de arranque
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]