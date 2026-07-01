FROM python:3.11-slim

# 安装 torch（cu128 兼容 CUDA 13.2，pip 自动解析最合适版本）
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128

# 先安装最新 transformers（确保支持 Qwen3.5），再装 llmcompressor（避免被锁死在 4.57.6）
RUN pip install --no-cache-dir transformers datasets accelerate
RUN pip install --no-cache-dir llmcompressor==0.11.0

COPY launcher.py /app/launcher.py
WORKDIR /app

ENTRYPOINT ["python3", "/app/launcher.py"]
