# job-template/job/model-evaluate-opencompass/Dockerfile
# 构建方式: 在项目根目录执行
#   docker build -t 10.121.177.20:8082/model-evaluate-opencompass:latest \
#       -f job-template/job/model-evaluate-opencompass/Dockerfile .
#
# 基础镜像: CUDA 12.1 + Ubuntu 22.04（支持 GPU 推理）
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

LABEL maintainer="cube-studio"
LABEL description="OpenCompass LLM Evaluation Tool - Cube Studio Job Template"

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8

# ── 运行时环境变量（OpenCompass 会自动识别） ──
ENV HF_ENDPOINT=https://hf-mirror.com
ENV DATASET_SOURCE=ModelScope

# ── 1. 系统依赖（使用清华 apt 镜像源） ──
RUN sed -i 's|http://archive.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    apt-get update -o Acquire::Retries=3 --fix-missing && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    python3 \
    python3-pip \
    python3-venv \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── 2. 配置 PyPI 国内镜像源 ──
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install --no-cache-dir --upgrade pip setuptools wheel

# ── 3. 安装 PyTorch（CUDA 12.1）──
# 关键：--index-url 显式指向 CUDA 版 PyTorch 源，必须放在包名后面才生效
# --extra-index-url 指向清华源作为依赖（numpy 等）的下载源
RUN pip3 install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1

# ── 4. 安装 OpenCompass ──
# 从国内 PyPI 镜像安装，无需 GitHub
# modelscope: 数据集下载源（对应 DATASET_SOURCE=ModelScope）
# transformers<5.0: 与 PyTorch 2.5.1 兼容（5.x 要求 torch>=2.6 以加载 .bin 模型）
RUN pip3 install --no-cache-dir \
    opencompass \
    Evaluate \
    modelscope \
    "transformers>=4.44,<5.0"

# ── 5. 查找 OpenCompass 安装路径，建立统一入口目录 ──
RUN OPENCOMPASS_DIR=$(python3 -c "import opencompass; import os; print(os.path.dirname(opencompass.__file__))") && \
    echo "OpenCompass installed at: $OPENCOMPASS_DIR" && \
    mkdir -p /opt/opencompass && \
    # 如果 pip 包包含 run.py，则链接；否则内置的 CLI 也能用
    if [ -f "$OPENCOMPASS_DIR/../run.py" ]; then \
        ln -sf "$OPENCOMPASS_DIR/../run.py" /opt/opencompass/run.py; \
    else \
        echo "[WARN] run.py 未找到，将仅使用 opencompass CLI: $OPENCOMPASS_DIR"; \
    fi && \
    if [ -d "$OPENCOMPASS_DIR/configs" ]; then \
        ln -sf "$OPENCOMPASS_DIR/configs" /opt/opencompass/configs; \
    else \
        echo "[WARN] configs 目录未找到: $OPENCOMPASS_DIR/configs"; \
    fi

# ── 6. 复制入口脚本 ──
COPY job-template/job/model-evaluate-opencompass/src/ /app/
RUN chmod +x /app/*.py

# ── 7. 安装 sitecustomize.py 到 site-packages（Python 启动时自动加载） ──
# 为 MsDataset.load 注入 trust_remote_code=True，在 opencompass 子进程中也生效
RUN SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])") && \
    cp /app/sitecustomize.py "$SITE_PACKAGES/sitecustomize.py" && \
    echo "sitecustomize.py installed to: $SITE_PACKAGES/sitecustomize.py"

WORKDIR /opt/opencompass
ENV PYTHONPATH=/app

ENTRYPOINT ["python3", "/app/run_evaluation.py"]
