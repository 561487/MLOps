FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip && \
    ln -sf /usr/bin/python3 /usr/bin/python

# 安装 PyTorch
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu121

# 安装剪枝依赖
RUN pip3 install torch-pruning --upgrade && \
    pip3 install transformers[sentencepiece] datasets accelerate

COPY launcher.py /app/launcher.py
WORKDIR /app

ENTRYPOINT ["python3", "/app/launcher.py"]
