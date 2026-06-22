FROM python:3.9-slim

# 安装 CUDA 版 PyTorch（使用最新 CUDA 12.4，与 K8s 驱动兼容）
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

# gptqmodel 需要 torch >= 2.7.1，cu124 提供的版本满足要求
RUN pip install --no-cache-dir gptqmodel && \
    pip install --no-cache-dir autoawq bitsandbytes datasets accelerate transformers

COPY launcher.py /app/launcher.py
WORKDIR /app

ENTRYPOINT ["python3", "/app/launcher.py"]
