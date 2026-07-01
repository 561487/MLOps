FROM python:3.11

# 完整版 python 自带 gcc 编译工具，无需额外安装

# 安装 torch + torchvision（cu128 频道，兼容 CUDA 13.2）
RUN pip --default-timeout=600 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 安装 gptqmodel + 依赖
RUN pip install --no-cache-dir gptqmodel autoawq bitsandbytes datasets accelerate transformers

COPY launcher.py /app/launcher.py
WORKDIR /app

ENTRYPOINT ["python3", "/app/launcher.py"]
