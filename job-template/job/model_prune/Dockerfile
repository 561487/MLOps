FROM python:3.11-slim

# 升级 pip
RUN pip install --upgrade pip

# CUDA 版 PyTorch
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# 剪枝依赖
RUN pip install --no-cache-dir torch-pruning transformers accelerate datasets

COPY launcher.py /app/launcher.py
WORKDIR /app

ENTRYPOINT ["python3", "/app/launcher.py"]
