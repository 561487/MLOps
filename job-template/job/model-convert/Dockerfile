FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
LABEL maintainer="cube-studio"
LABEL description="Model Format Conversion — PyTorch/HF → ONNX / TensorRT / TorchScript / GGUF"

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_ENDPOINT=https://hf-mirror.com

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3 /usr/bin/python

RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir onnx onnxruntime optimum transformers safetensors
RUN pip3 install --no-cache-dir gguf sentencepiece protobuf
RUN pip3 install --no-cache-dir tensorrt --index-url https://pypi.nvidia.com || \
    echo "[WARN] tensorrt 需要 GPU 驱动, ONNX→TensorRT 转换仅在 GPU 节点可用"

COPY job-template/job/model-convert/ /app/
RUN chmod +x /app/launcher.py

WORKDIR /app
ENTRYPOINT ["python3", "/app/launcher.py"]
