import base64
import binascii
import os
import tempfile
import time
from typing import Any, Dict, Optional

import librosa
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


MODEL_NAME = os.getenv("KUBEFLOW_MODEL_NAME", "whisper-base")
MODEL_VERSION = os.getenv("KUBEFLOW_MODEL_VERSION", "v1").replace(".", "")
MODEL_PATH = os.getenv(
    "MODELPATH",
    os.getenv("KUBEFLOW_MODEL_PATH", "/whisper/models/whisper-base"),
)
MAX_AUDIO_MB = int(os.getenv("MAX_AUDIO_MB", "100"))
MAX_AUDIO_SECONDS = int(os.getenv("MAX_AUDIO_SECONDS", "600"))

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE.startswith("cuda") else torch.float32

processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    torch_dtype=TORCH_DTYPE,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)
model.to(DEVICE)
model.eval()

asr = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=TORCH_DTYPE,
    device=0 if DEVICE.startswith("cuda") else -1,
    chunk_length_s=30,
)

app = FastAPI(title="Whisper speech recognition service")


def _decode_audio(audio_base64: str) -> bytes:
    if not audio_base64:
        raise ValueError("audio is required")
    if "," in audio_base64 and audio_base64.lstrip().startswith("data:"):
        audio_base64 = audio_base64.split(",", 1)[1]
    try:
        content = base64.b64decode(audio_base64, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("audio must be valid base64 data") from exc
    if len(content) > MAX_AUDIO_MB * 1024 * 1024:
        raise ValueError(f"audio exceeds {MAX_AUDIO_MB} MB")
    return content


def transcribe_bytes(
    content: bytes,
    language: str = "auto",
    task: str = "transcribe",
    timestamps: bool = True,
) -> Dict[str, Any]:
    temp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        audio, _ = librosa.load(temp_path, sr=16000, mono=True)
        duration = len(audio) / 16000.0
        if duration <= 0:
            raise ValueError("audio is empty")
        if duration > MAX_AUDIO_SECONDS:
            raise ValueError(f"audio exceeds {MAX_AUDIO_SECONDS} seconds")

        generate_kwargs: Dict[str, Any] = {"task": task}
        if language and language != "auto":
            generate_kwargs["language"] = language

        result = asr(
            audio,
            return_timestamps=bool(timestamps),
            generate_kwargs=generate_kwargs,
        )
        chunks = result.get("chunks") or []
        segments = []
        for chunk in chunks:
            timestamp = chunk.get("timestamp") or (None, None)
            segments.append({
                "start": timestamp[0],
                "end": timestamp[1],
                "text": (chunk.get("text") or "").strip(),
            })

        return {
            "text": (result.get("text") or "").strip(),
            "language": language or "auto",
            "duration_seconds": round(duration, 3),
            "segments": segments,
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "device": DEVICE,
    }


@app.post("/demo/predict")
@app.post(f"/v1/models/{MODEL_NAME}/versions/{MODEL_VERSION}/predict")
async def predict(request: Request):
    data = await request.json()
    try:
        content = _decode_audio(data.get("audio", ""))
        language = data.get("language") or "auto"
        task = data.get("task") or "transcribe"
        if task not in {"transcribe", "translate"}:
            raise ValueError("task must be transcribe or translate")
        timestamps = data.get("timestamps", True)
        if isinstance(timestamps, str):
            timestamps = timestamps.lower() not in {"false", "0", "no"}
        started_at = time.time()
        # Whisper inference is CPU/GPU bound. Run it outside the ASGI event loop so
        # health checks can still respond while a long audio file is transcribed.
        result = await run_in_threadpool(
            transcribe_bytes, content, language, task, bool(timestamps)
        )
        result["inference_seconds"] = round(time.time() - started_at, 3)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/")
async def index():
    return {
        "service": "whisper",
        "health": "/health",
        "predict": f"/v1/models/{MODEL_NAME}/versions/{MODEL_VERSION}/predict",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
