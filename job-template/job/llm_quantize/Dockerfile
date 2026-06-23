FROM python:3.11-slim

# 安装 PyTorch（LLM Compressor 需要 torch，Linux x86_64 默认带 CUDA）
RUN pip install --no-cache-dir torch

# LLM Compressor + 依赖
RUN pip install --no-cache-dir llmcompressor && \
    pip install --no-cache-dir datasets accelerate transformers

COPY launcher.py /app/launcher.py
WORKDIR /app

ENTRYPOINT ["python3", "/app/launcher.py"]
