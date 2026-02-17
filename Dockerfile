FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/cache/torch
ENV HF_HOME=/cache/huggingface

WORKDIR /app

# System dependencies (Ubuntu 22.04 has Python 3.10 built-in)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    ffmpeg libsndfile1 git ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install pinned Python dependencies
RUN pip install --no-cache-dir \
    torch==2.0.0+cu118 \
    torchaudio==2.0.1+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    faster-whisper==1.1.0 \
    ctranslate2==4.5.0 \
    pyannote.audio==3.1.1 \
    huggingface_hub==0.23.0 \
    runpod==1.7.0 \
    requests

# Install whisperx from git main (supports large-v3-turbo)
RUN pip install --no-cache-dir --no-deps "whisperx @ git+https://github.com/m-bain/whisperX.git"

# Pre-download all models during build to avoid runtime downloads
RUN mkdir -p /cache/torch /cache/huggingface

# 1. Download whisper large-v3-turbo model
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('large-v3-turbo', device='cpu', compute_type='int8')"

# 2. Download VAD model
RUN python3 -c "from whisperx.vad import load_vad_model; load_vad_model('cpu')"

# 3. Download alignment models for primary languages
RUN python3 -c "import whisperx; whisperx.load_align_model(language_code='en', device='cpu')"
RUN python3 -c "import whisperx; whisperx.load_align_model(language_code='pt', device='cpu')"

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
