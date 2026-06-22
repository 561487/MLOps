FROM python:3.9-slim

# CUDA 版 PyTorch（自带 CUDA 运行时）
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# 量化依赖
RUN pip install --no-cache-dir gptqmodel && \
    pip install --no-cache-dir autoawq bitsandbytes datasets accelerate transformers

COPY launcher.py /app/launcher.py
WORKDIR /app

ENTRYPOINT ["python3", "/app/launcher.py"]
