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

# Pre-download the English alignment model (wav2vec2 from torchaudio)
# This avoids runtime 301 redirect issues from torch.hub
RUN python -c "\
import numpy as np; \
np.NaN = np.nan if not hasattr(np, 'NaN') else np.NaN; \
import whisperx; \
whisperx.load_align_model(language_code='en', device='cpu')"

WORKDIR /app
COPY handler.py .

CMD ["python", "-u", "handler.py"]
