"""
RunPod Serverless Handler for WhisperX Transcription with Speaker Diarization.

Models are loaded lazily on first job to avoid startup crashes.
"""

import os
import sys
import tempfile
import traceback
import gc
import json

# Fix np.NaN removal in NumPy 2.0 (needed by pyannote.audio 3.1.1)
import numpy as np
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

import requests
import torch
import runpod

print("[init] Handler starting...", flush=True)
print(f"[init] Python: {sys.version}", flush=True)
print(f"[init] CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"[init] GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[init] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

# Verify whisperx can be imported
try:
    import whisperx
    print(f"[init] WhisperX version: {whisperx.__version__}", flush=True)
except Exception as e:
    print(f"[init] ERROR importing whisperx: {e}", flush=True)
    traceback.print_exc()

# Global model cache - loaded lazily on first job
_model = None
_device = None
_compute_type = None


def get_model():
    """Load WhisperX model on first call, cache for reuse."""
    global _model, _device, _compute_type
    if _model is None:
        import whisperx
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _compute_type = "float16" if _device == "cuda" else "int8"
        print(f"[model] Loading WhisperX large-v2 on {_device} ({_compute_type})...", flush=True)
        _model = whisperx.load_model("large-v2", _device, compute_type=_compute_type)
        print(f"[model] WhisperX model loaded successfully.", flush=True)
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

    size_mb = total / (1024 * 1024)
    print(f"[job] Downloaded {size_mb:.1f} MB", flush=True)
    return dest_path


def handler(job):
    """Process a transcription job."""
    import whisperx

    print(f"[job] === NEW JOB RECEIVED ===", flush=True)
    print(f"[job] Job keys: {list(job.keys())}", flush=True)

    try:
        job_input = job["input"]
        print(f"[job] Input keys: {list(job_input.keys())}", flush=True)

        # Parse input parameters
        audio_url = job_input["audio_file"]
        language = job_input.get("language", None)
        batch_size = job_input.get("batch_size", 16)
        diarization = job_input.get("diarization", True)
        hf_token = job_input.get("huggingface_access_token") or os.environ.get("HF_TOKEN")
        min_speakers = job_input.get("min_speakers")
        max_speakers = job_input.get("max_speakers")

        print(f"[job] audio_url: {audio_url[:100]}...", flush=True)
        print(f"[job] language={language}, batch_size={batch_size}, diarization={diarization}", flush=True)
        print(f"[job] hf_token present: {bool(hf_token)}", flush=True)

        # Load model (cached after first call)
        print("[job] Step 1: Loading model...", flush=True)
        model, device, compute_type = get_model()
        print("[job] Step 1: Model ready.", flush=True)

        # Download audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            print("[job] Step 2: Downloading audio...", flush=True)
            download_audio(audio_url, tmp_path)
            print("[job] Step 2: Download complete.", flush=True)

            # Load audio
            print("[job] Step 3: Loading audio into memory...", flush=True)
            audio = whisperx.load_audio(tmp_path)
            print(f"[job] Step 3: Audio loaded. Length: {len(audio)} samples ({len(audio)/16000:.1f}s)", flush=True)

            # Transcribe
            print(f"[job] Step 4: Transcribing (batch_size={batch_size}, language={language})...", flush=True)
            result = model.transcribe(
                audio,
                batch_size=batch_size,
                language=language,
            )
            print(f"[job] Step 4: Transcription complete. {len(result.get('segments', []))} segments.", flush=True)

            detected_language = result.get("language", language or "en")
            print(f"[job] Detected language: {detected_language}", flush=True)

            # Align output for word-level timestamps
            print("[job] Step 5: Aligning output...", flush=True)
            align_model, align_metadata = whisperx.load_align_model(
                language_code=detected_language,
                device=device,
            )
            result = whisperx.align(
                result["segments"],
                align_model,
                align_metadata,
                audio,
                device,
                return_char_alignments=False,
            )
            print("[job] Step 5: Alignment complete.", flush=True)

            # Free alignment model memory
            del align_model, align_metadata
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            # Speaker diarization
            if diarization and hf_token:
                print("[job] Step 6: Running speaker diarization...", flush=True)
                diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=hf_token,
                    device=device,
                )

                diarize_kwargs = {}
                if min_speakers is not None:
                    diarize_kwargs["min_speakers"] = min_speakers
                if max_speakers is not None:
                    diarize_kwargs["max_speakers"] = max_speakers

                diarize_segments = diarize_model(audio, **diarize_kwargs)
                result = whisperx.assign_word_speakers(diarize_segments, result)

                # Free diarization model memory
                del diarize_model, diarize_segments
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()

                print(f"[job] Step 6: Diarization complete.", flush=True)
            elif diarization and not hf_token:
                print("[job] WARNING: Diarization requested but no HF token. Skipping.", flush=True)
            else:
                print("[job] Diarization not requested. Skipping.", flush=True)

            # Build output
            segments = result.get("segments", [])
            print(f"[job] Done! {len(segments)} segments produced.", flush=True)

            # Ensure all segment data is JSON-serializable
            clean_segments = []
            for seg in segments:
                clean_seg = {}
                for k, v in seg.items():
                    if isinstance(v, float) and (v != v):  # NaN check
                        clean_seg[k] = None
                    elif isinstance(v, list):
                        clean_words = []
                        for w in v:
                            if isinstance(w, dict):
                                clean_w = {}
                                for wk, wv in w.items():
                                    if isinstance(wv, float) and (wv != wv):
                                        clean_w[wk] = None
                                    else:
                                        clean_w[wk] = wv
                                clean_words.append(clean_w)
                            else:
                                clean_words.append(w)
                        clean_seg[k] = clean_words
                    else:
                        clean_seg[k] = v
                clean_segments.append(clean_seg)

            output = {
                "segments": clean_segments,
                "detected_language": detected_language,
            }

            # Verify it's serializable before returning
            try:
                json.dumps(output)
                print("[job] Output verified as JSON-serializable.", flush=True)
            except (TypeError, ValueError) as e:
                print(f"[job] WARNING: Output not JSON-serializable: {e}", flush=True)
                # Fallback: convert everything to strings
                output = json.loads(json.dumps(output, default=str))

            return output

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[job] ERROR: {error_msg}", flush=True)
        traceback.print_exc()
        return {"error": error_msg}


print("[init] Handler registered. Ready for jobs.", flush=True)
runpod.serverless.start({"handler": handler})
