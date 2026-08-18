import base64
import os
import sys

import requests


host = os.getenv("WHISPER_SERVICE_URL", "http://127.0.0.1:8000")
audio_path = sys.argv[1] if len(sys.argv) > 1 else os.getenv("WHISPER_AUDIO", "")
if not audio_path:
    raise SystemExit("Usage: python client.py /path/to/audio.wav")

url = host.rstrip("/") + "/v1/models/whisper-base/versions/v1/predict"

with open(audio_path, "rb") as audio_file:
    payload = {
        "audio": base64.b64encode(audio_file.read()).decode("utf-8"),
        "language": "auto",
        "task": "transcribe",
        "timestamps": True,
    }

response = requests.post(url, json=payload, timeout=300)
response.raise_for_status()
print(response.json())
