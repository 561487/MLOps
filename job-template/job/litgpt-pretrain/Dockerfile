FROM python:3.11-slim-bookworm
LABEL maintainer="cube-studio"
LABEL description="LitGPT Pre-training — Cube Studio Job Template"

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_ENDPOINT=https://hf-mirror.com
ENV PYTHONUNBUFFERED=1

# 1. System deps
RUN sed -i 's|http://deb.debian.org|https://mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update -o Acquire::Retries=3 --fix-missing && \
    apt-get install -y --no-install-recommends git wget curl ca-certificates build-essential && \
    rm -rf /var/lib/apt/lists/*

# 2. pip mirrors
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip config set global.trusted-host mirrors.aliyun.com && \
    pip install --upgrade pip setuptools wheel

# 3. 合到一起装，pip 一次性协调版本，不会拆散 torch+torchvision 兼容组
RUN pip install \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://mirrors.aliyun.com/pypi/simple/ \
    torch torchvision torchaudio litgpt swanlab litdata

# 5. Copy code
COPY job-template/job/litgpt-pretrain/launcher.py /app/
COPY job-template/job/litgpt-pretrain/start.py /app/
COPY job-template/common/ /app/common/

WORKDIR /app
ENV PYTHONPATH=/app

ENTRYPOINT ["python3", "/app/launcher.py"]
