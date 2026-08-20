FROM 10.121.177.20:8082/mlops/llama_factory:20250712

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive

# ── 切换默认 python3 到 3.13（llamafactory>=0.9.5 要求 Python>=3.11）──
RUN rm -f /usr/local/bin/python3 && ln -sf /usr/bin/python3.13 /usr/local/bin/python3

# ── 为 Python 3.13 安装 pip（使用 get-pip.py，绕过 PEP 668）──
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3 - --break-system-packages

# ── 一次性安装所有依赖（使用清华镜像加速，合并为单层减少镜像体积）──
# torch 用 --no-deps 跳过冗余 nvidia-* 包(1.5GB)，系统已有 CUDA 12.8 库
RUN python3 -m pip install --no-cache-dir --break-system-packages \
    --no-deps torch==2.8.0 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    python3 -m pip install --no-cache-dir --break-system-packages \
    filelock jinja2 fsspec sympy networkx numpy pillow \
    nvidia-cuda-runtime-cu12 \
    "llamafactory>=0.9.5" \
    "transformers>=4.57" \
    "peft>=0.18" \
    "swanlab>=0.8.4" \
    psutil pyyaml email-validator \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# ── 复制代码 ──
COPY job/llama_factory/ /app/
COPY common/ /app/common/
WORKDIR /app
ENV PYTHONPATH=/app

ENTRYPOINT ["python3", "launcher.py"]
