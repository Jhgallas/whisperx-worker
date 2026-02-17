"""
RunPod Serverless Handler for WhisperX Transcription with Speaker Diarization.

Uses large-v3-turbo model for optimal speed/accuracy/cost balance.
Supports auto language detection for bilingual audio.
Models are pre-downloaded in Docker image; loaded lazily on first job.
"""

import os
import sys
import tempfile
import traceback
import gc
import json

import requests
import torch
import runpod

print("[init] Handler starting...", flush=True)
print(f"[init] Python: {sys.version}", flush=True)
print(f"[init] CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"[init] GPU: {torch.cuda.get_device_name(0)}", flush=True)
    props = torch.cuda.get_device_properties(0)
    vram = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
    print(f"[init] VRAM: {vram / 1e9:.1f} GB", flush=True)

try:
    import whisperx
    print(f"[init] WhisperX loaded OK", flush=True)
except Exception as e:
    print(f"[init] ERROR importing whisperx: {e}", flush=True)
    traceback.print_exc()

import numpy as np
print(f"[init] numpy: {np.__version__}", flush=True)

# Global model cache
_model = None
_device = None
_compute_type = None

MODEL_NAME = "large-v3-turbo"


def get_model():
    """Load WhisperX model on first call, cache for reuse."""
    global _model, _device, _compute_type
    if _model is None:
        import whisperx
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        # int8_float16: quantized weights (fast) with float16 activations (accurate)
        # Best balance for RTX 4090 — faster than float16 with negligible quality loss
        _compute_type = "int8_float16" if _device == "cuda" else "int8"
        print(f"[model] Loading WhisperX {MODEL_NAME} on {_device} ({_compute_type})...", flush=True)
        _model = whisperx.load_model(MODEL_NAME, _device, compute_type=_compute_type)
        print(f"[model] Model loaded successfully.", flush=True)
    return _model, _device, _compute_type


def download_audio(url, dest_path):
    """Download audio file from URL to a local path."""
    print(f"[job] Downloading audio from: {url[:100]}...", flush=True)
    response = requests.get(url, stream=True, timeout=600)
    response.raise_for_status()

    total = 0
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            total += len(chunk)

    print(f"[job] Downloaded {total / (1024*1024):.1f} MB", flush=True)


def handler(job):
    """Process a transcription job."""
    import whisperx

    print(f"[job] === NEW JOB RECEIVED ===", flush=True)

    try:
        job_input = job["input"]

        audio_url = job_input.get("audio_url") or job_input.get("audio_file")
        language = job_input.get("language", None)
        # Treat "auto" as None so whisper auto-detects
        if language == "auto":
            language = None
        batch_size = job_input.get("batch_size", 16)
        diarization = job_input.get("diarization", True)
        hf_token = job_input.get("huggingface_access_token") or os.environ.get("HF_TOKEN")
        min_speakers = job_input.get("min_speakers")
        max_speakers = job_input.get("max_speakers")

        print(f"[job] language={language or 'auto-detect'}, batch_size={batch_size}, diarization={diarization}", flush=True)

        # Step 1: Load model
        print("[job] Step 1: Loading model...", flush=True)
        model, device, compute_type = get_model()
        print("[job] Step 1: Done.", flush=True)

        # Step 2: Download audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            print("[job] Step 2: Downloading audio...", flush=True)
            download_audio(audio_url, tmp_path)
            print("[job] Step 2: Done.", flush=True)

            # Step 3: Load audio
            print("[job] Step 3: Loading audio...", flush=True)
            audio = whisperx.load_audio(tmp_path)
            print(f"[job] Step 3: Done. {len(audio)/16000:.1f}s of audio.", flush=True)

            # Step 4: Transcribe
            print("[job] Step 4: Transcribing...", flush=True)
            result = model.transcribe(audio, batch_size=batch_size, language=language)
            detected_language = result.get("language", language or "en")
            print(f"[job] Step 4: Done. {len(result.get('segments', []))} segments, lang={detected_language}.", flush=True)

            # Step 5: Align
            print("[job] Step 5: Aligning...", flush=True)
            try:
                align_model, align_metadata = whisperx.load_align_model(
                    language_code=detected_language, device=device)
                result = whisperx.align(
                    result["segments"], align_model, align_metadata,
                    audio, device, return_char_alignments=False)
                del align_model, align_metadata
            except Exception as e:
                # Alignment model may not exist for all languages — continue without alignment
                print(f"[job] Step 5: Alignment failed ({e}), continuing without alignment.", flush=True)
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            print("[job] Step 5: Done.", flush=True)

            # Step 6: Diarization
            if diarization and hf_token:
                print("[job] Step 6: Diarizing...", flush=True)
                diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=hf_token, device=device)
                diarize_kwargs = {}
                if min_speakers is not None:
                    diarize_kwargs["min_speakers"] = min_speakers
                if max_speakers is not None:
                    diarize_kwargs["max_speakers"] = max_speakers
                diarize_segments = diarize_model(audio, **diarize_kwargs)
                result = whisperx.assign_word_speakers(diarize_segments, result)
                del diarize_model, diarize_segments
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                print("[job] Step 6: Done.", flush=True)
            elif diarization:
                print("[job] WARNING: Diarization requested but no HF token. Skipping.", flush=True)

            # Build output
            segments = result.get("segments", [])
            output = {
                "segments": segments,
                "detected_language": detected_language,
            }

            # Verify JSON-serializable
            try:
                json.dumps(output)
            except (TypeError, ValueError):
                output = json.loads(json.dumps(output, default=str))

            print(f"[job] Complete! {len(segments)} segments.", flush=True)
            return output

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[job] ERROR: {error_msg}", flush=True)
        traceback.print_exc()
        return {"error": error_msg}


print("[init] Handler registered. Ready for jobs.", flush=True)
runpod.serverless.start({"handler": handler})
