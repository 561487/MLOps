FROM python:3.10-slim
LABEL maintainer="cube-studio"
LABEL description="Model Format Conversion — PyTorch/HF → ONNX → TensorRT"

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_ENDPOINT=https://hf-mirror.com

RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --no-cache-dir \
    onnx onnxruntime optimum transformers safetensors

COPY job-template/job/model-convert/ /app/
RUN chmod +x /app/launcher.py

WORKDIR /app
ENTRYPOINT ["python3", "/app/launcher.py"]
