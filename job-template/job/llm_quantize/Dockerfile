FROM python:3.11-slim

# 安装 torch（cu128 兼容 CUDA 13.2，pip 自动解析最合适版本）
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128

# 安装量化依赖（pip 自动解析兼容版本组合）
RUN pip install --no-cache-dir llmcompressor==0.11.0 transformers datasets accelerate

COPY launcher.py /app/launcher.py
WORKDIR /app

ENTRYPOINT ["python3", "/app/launcher.py"]
