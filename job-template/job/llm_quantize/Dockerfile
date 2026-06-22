FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 修复 apt 源（删除坏的签名，使用 http）
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get clean && \
    apt-get update -y 2>/dev/null || true && \
    apt-get install -y python3 python3-pip && \
    ln -sf /usr/bin/python3 /usr/bin/python

# 升级 pip
RUN python3 -m pip install --upgrade pip

# CUDA 版 PyTorch
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu121

# 量化依赖
RUN pip3 install gptqmodel --no-build-isolation && \
    pip3 install autoawq bitsandbytes datasets accelerate transformers

COPY launcher.py /app/launcher.py
WORKDIR /app

ENTRYPOINT ["python3", "/app/launcher.py"]
