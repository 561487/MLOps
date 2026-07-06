FROM python:3.10-slim
LABEL maintainer="cube-studio"
LABEL description="Model Format Conversion — PyTorch/HF → ONNX / TensorRT / TorchScript / GGUF"

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_ENDPOINT=https://hf-mirror.com

RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
RUN pip3 install --no-cache-dir onnx onnxruntime optimum transformers safetensors
RUN pip3 install --no-cache-dir gguf sentencepiece protobuf
# tensorrt 需要 CUDA 环境, GPU 节点上手动: pip install tensorrt --index-url https://pypi.nvidia.com

COPY job-template/job/model-convert/ /app/
RUN chmod +x /app/launcher.py

WORKDIR /app
ENTRYPOINT ["python3", "/app/launcher.py"]
