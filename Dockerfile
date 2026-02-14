FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip install --no-cache-dir \
    runpod==1.7.0 \
    whisperx==3.1.1 \
    pyannote.audio==3.1.1 \
    requests

WORKDIR /app
COPY handler.py .

CMD ["python", "-u", "handler.py"]
