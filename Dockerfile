FROM runpod/pytorch:2.0.0-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/cache/torch
ENV HF_HOME=/cache/huggingface

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install pinned Python dependencies (proven working combination)
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    faster-whisper==0.10.1 \
    ctranslate2==4.4.0 \
    pyannote.audio==3.1.1 \
    huggingface_hub==0.23.0 \
    runpod==1.7.0 \
    requests

# Install whisperx from pinned git tag
RUN pip install --no-cache-dir --no-deps "whisperx @ git+https://github.com/m-bain/whisperX.git@v3.1.6"

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
