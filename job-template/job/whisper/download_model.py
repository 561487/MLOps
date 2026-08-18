import os

from modelscope import snapshot_download
from transformers import AutoProcessor, WhisperForConditionalGeneration


source = os.getenv("WHISPER_MODEL_SOURCE", "openai-mirror/whisper-base")
target = os.getenv("WHISPER_MODEL_TARGET", "/whisper/models/whisper-base")

os.makedirs(target, exist_ok=True)
model_dir = snapshot_download(
    source,
    local_dir=target,
    allow_file_pattern=[
        "*.json",
        "*.txt",
        "*.model",
        "*.safetensors",
    ],
)

# 在镜像构建阶段验证模型文件可被 Transformers 离线加载，避免把不完整镜像推入仓库。
AutoProcessor.from_pretrained(model_dir, local_files_only=True)
WhisperForConditionalGeneration.from_pretrained(
    model_dir,
    local_files_only=True,
    use_safetensors=True,
)
print(f"Whisper model downloaded from ModelScope to {model_dir}")
