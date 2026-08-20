# job-template/job/model-distillation/Dockerfile
# 优化版：所有依赖在 v5 基础镜像中，只保留会变化的 COPY 层
#
# rebuild 耗时：
#   只改 launcher.py / chat_template → ~5 秒
#   改 easydistill_src               → ~30 秒
FROM modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.9.0-vllm0.13.0-modelscope1.33.0-swift3.12.5

LABEL maintainer="cube-studio"
LABEL description="Model Distillation Node - EasyDistill (CUDA 12.8, Qwen3.5 text)"

# ── SwanLab 监控依赖 ──
ARG TRANSFORMERS_VERSION="5.9.0"
ARG TRL_VERSION="0.27.0"
RUN pip3 uninstall -y ms-swift vllm lmdeploy evalscope peft || true
RUN pip3 install --no-cache-dir --timeout 120 --retries 5 --upgrade \
    "transformers==${TRANSFORMERS_VERSION}" "trl==${TRL_VERSION}" \
    "accelerate>=1.10,<2" "datasets>=4,<5" \
    "qwen-vl-utils>=0.0.14" "swanlab>=0.8.4"

# Qwen3.5 Gated DeltaNet optional CUDA kernels are intentionally not compiled
# in the base image: Transformers provides a portable PyTorch fallback.
RUN python3 -c "import torch, torchvision, transformers; print(torch.__version__, torchvision.__version__, transformers.__version__)"

# ── EasyDistill 源码（偶尔变）──
COPY job-template/job/model-distillation/easydistill_src /opt/easydistill/
RUN pip3 install --no-cache-dir --no-deps /opt/easydistill/

# ── 项目文件（最常变）──
COPY job-template/job/model-distillation/launcher.py /app/
COPY job-template/job/model-distillation/chat_template/ /app/chat_template/
COPY job-template/job/model-distillation/accelerate_config.yaml /app/
COPY job-template/common/ /app/common/

# ── 运行时配置 ──
RUN EASY_DISTILL_DIR=$(python3 -c "import easydistill; import os; print(os.path.dirname(easydistill.__file__))") && \
    SITE_PACKAGES=$(dirname ${EASY_DISTILL_DIR}) && \
    mkdir -p ${SITE_PACKAGES}/configs/accelerate_config && \
    cp /app/accelerate_config.yaml ${SITE_PACKAGES}/configs/accelerate_config/muti_gpu.yaml

WORKDIR /app
ENV PYTHONPATH=/app

ENTRYPOINT ["python3", "launcher.py"]
