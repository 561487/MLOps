# job-template/job/model-distillation/Dockerfile
# 基于已有镜像修复：CUDA PyTorch + EasyDistill（仅训练依赖，不含 vllm）
FROM 10.121.177.20:8082/mlops/model-distillation:20260630

LABEL maintainer="cube-studio"
LABEL description="Model Distillation Node - EasyDistill (CUDA 12.1, train-only) - v20260701"

# ── 替换为 CUDA 版 PyTorch（先卸载 CPU 版以释放空间）──
RUN pip3 uninstall -y torch && \
    pip3 install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1

# ── 复制 EasyDistill 源码 ──
ARG CACHEBUST=0
COPY job-template/job/model-distillation/easydistill_src /opt/easydistill/

# ── 安装 EasyDistill 框架（--no-deps 跳过 vllm，只装训练需要的）──
RUN pip3 install --no-cache-dir --no-deps /opt/easydistill/ && \
    pip3 install --no-cache-dir \
    transformers==4.51.1 \
    trl==0.17.0 \
    tokenizers==0.21.1 \
    datasets \
    accelerate \
    jinja2 \
    jsonlines \
    modelscope

# ── 复制项目文件 ──
COPY job-template/job/model-distillation/launcher.py /app/
COPY job-template/job/model-distillation/chat_template/ /app/chat_template/
COPY job-template/job/model-distillation/accelerate_config.yaml /app/

# ── 创建 EasyDistill CLI 需要的 configs 目录 + 单卡 accelerate 配置 ──
RUN EASY_DISTILL_DIR=$(python3 -c "import easydistill; import os; print(os.path.dirname(easydistill.__file__))") && \
    SITE_PACKAGES=$(dirname ${EASY_DISTILL_DIR}) && \
    mkdir -p ${SITE_PACKAGES}/configs/accelerate_config && \
    cp /app/accelerate_config.yaml ${SITE_PACKAGES}/configs/accelerate_config/muti_gpu.yaml

WORKDIR /app
ENV PYTHONPATH=/app

ENTRYPOINT ["python3", "launcher.py"]