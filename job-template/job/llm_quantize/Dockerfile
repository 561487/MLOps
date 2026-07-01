FROM python:3.11-slim

# 安装编译工具（gptqmodel 依赖 pypcre 需要 gcc 编译 C 扩展）
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# 安装 torch（cu128 最新稳定版，兼容 CUDA 13.2，加长超时避免网络波动）
RUN pip --default-timeout=600 install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# 安装 gptqmodel + 依赖（不锁版本，pip 自动解析最合适版本）
RUN pip install --no-cache-dir gptqmodel autoawq bitsandbytes datasets accelerate transformers

COPY launcher.py /app/launcher.py
WORKDIR /app

ENTRYPOINT ["python3", "/app/launcher.py"]
