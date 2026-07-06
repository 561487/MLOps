# job-template/job/model-distillation/Dockerfile
# 优化版：所有依赖在 v5 基础镜像中，只保留会变化的 COPY 层
#
# rebuild 耗时：
#   只改 launcher.py / chat_template → ~5 秒
#   改 easydistill_src               → ~30 秒
FROM 10.121.177.20:8082/mlops/model-distillation:20260704-v5

LABEL maintainer="cube-studio"
LABEL description="Model Distillation Node - EasyDistill (CUDA 12.1, train-only)"

# ── EasyDistill 源码（偶尔变）──
COPY job-template/job/model-distillation/easydistill_src /opt/easydistill/
RUN pip3 install --no-cache-dir --no-deps /opt/easydistill/

# ── 项目文件（最常变）──
COPY job-template/job/model-distillation/launcher.py /app/
COPY job-template/job/model-distillation/chat_template/ /app/chat_template/
COPY job-template/job/model-distillation/accelerate_config.yaml /app/

# ── 运行时配置 ──
RUN EASY_DISTILL_DIR=$(python3 -c "import easydistill; import os; print(os.path.dirname(easydistill.__file__))") && \
    SITE_PACKAGES=$(dirname ${EASY_DISTILL_DIR}) && \
    mkdir -p ${SITE_PACKAGES}/configs/accelerate_config && \
    cp /app/accelerate_config.yaml ${SITE_PACKAGES}/configs/accelerate_config/muti_gpu.yaml

WORKDIR /app
ENV PYTHONPATH=/app

ENTRYPOINT ["python3", "launcher.py"]
