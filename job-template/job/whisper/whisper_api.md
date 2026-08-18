# Whisper API

## 请求

```http
POST /v1/models/whisper-base/versions/v1/predict
Content-Type: application/json
```

```json
{
  "audio": "base64 audio data",
  "language": "auto",
  "task": "transcribe",
  "timestamps": true
}
```

## 响应

```json
{
  "text": "识别文本",
  "language": "auto",
  "duration_seconds": 3.2,
  "segments": [
    {"start": 0.0, "end": 3.2, "text": "识别文本"}
  ],
  "inference_seconds": 0.8
}
```
