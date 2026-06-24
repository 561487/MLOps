FROM python:3.11-slim

# 升级 pip
RUN pip install --upgrade pip

# 安装 torch（Linux x86_64 默认带 CUDA）
RUN pip install --no-cache-dir torch

# 量化依赖（gptqmodel 需要 torch >= 2.7.1，用最新 torch 满足要求）
RUN pip install --no-cache-dir gptqmodel && \
    pip install --no-cache-dir autoawq bitsandbytes datasets accelerate transformers

COPY launcher.py /app/launcher.py
WORKDIR /app

ENTRYPOINT ["python3", "/app/launcher.py"]
