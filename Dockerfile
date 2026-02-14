FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
# Pin numpy<2.0 to avoid np.NaN removal issue in pyannote.audio 3.1.1
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    runpod==1.7.0 \
    whisperx==3.1.1 \
    pyannote.audio==3.1.1 \
    "huggingface_hub>=0.23" \
    requests

# Pre-download the WhisperX model to avoid runtime download issues
RUN python -c "from faster_whisper.utils import download_model; download_model('large-v2')"

WORKDIR /app
COPY handler.py .

CMD ["python", "-u", "handler.py"]
