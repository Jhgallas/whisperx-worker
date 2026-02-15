FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/cache/torch
ENV HF_HOME=/cache/huggingface

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    bash ca-certificates curl git ffmpeg libsndfile1 \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3.10-distutils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create venv with Python 3.10
RUN python3.10 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install pinned Python dependencies (proven working combination)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.0.0 \
    torchaudio==2.0.0 \
    -f https://download.pytorch.org/whl/cu118/torch_stable.html

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    faster-whisper==0.10.1 \
    ctranslate2==4.4.0 \
    pyannote.audio==3.1.1 \
    huggingface_hub==0.23.0 \
    runpod==1.7.0 \
    requests

# Install whisperx from pinned git tag
RUN pip install --no-cache-dir "whisperx @ git+https://github.com/m-bain/whisperX.git@v3.1.6"

# Pre-download all models during build to avoid runtime downloads
RUN mkdir -p /cache/torch /cache/huggingface

# 1. Download whisper large-v2 model
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('large-v2', device='cpu', compute_type='int8')"

# 2. Download VAD model
RUN python -c "from whisperx.vad import load_vad_model; load_vad_model('cpu')"

# 3. Download English alignment model
RUN python -c "import whisperx; whisperx.load_align_model(language_code='en', device='cpu')"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
