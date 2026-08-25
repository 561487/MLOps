FROM 10.121.177.20:8082/mlops/llm-offline-predict:20260714

# ============================================================
# 增量构建：基于上一版镜像，只覆盖更新的文件，构建速度快
# ============================================================

# 添加诊断工具（可选，nvidia-smi 已内置）
RUN pip install --quiet py3nvml 2>/dev/null || true

# 覆盖更新文件
COPY predict_model.py /app/predict_model.py
COPY py_rabbit.py /app/py_rabbit.py
COPY predict.py /app/predict.py

# 诊断脚本
RUN echo '#!/bin/bash' > /app/diagnose.sh && \
    echo 'echo "===== DIAGNOSTIC INFO ====="' >> /app/diagnose.sh && \
    echo 'echo "[1] Hostname: $(hostname)"' >> /app/diagnose.sh && \
    echo 'echo "[2] Date: $(date)"' >> /app/diagnose.sh && \
    echo 'echo "[3] Python: $(python3 --version)"' >> /app/diagnose.sh && \
    echo 'echo "[4] GPU:"' >> /app/diagnose.sh && \
    echo 'nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv 2>/dev/null || echo "  nvidia-smi not available"' >> /app/diagnose.sh && \
    echo 'echo "[5] Memory:"' >> /app/diagnose.sh && \
    echo 'free -h' >> /app/diagnose.sh && \
    echo 'echo "[6] Disk (model path):"' >> /app/diagnose.sh && \
    echo 'df -h /mnt/storage/models-share-volume/ 2>/dev/null || echo "  not mounted"' >> /app/diagnose.sh && \
    echo 'echo "[7] Env vars:"' >> /app/diagnose.sh && \
    echo 'env | grep -E "MODEL_PATH|INPUT_FILE|OUTPUT_FILE|MAX_NEW_TOKENS|TEMPERATURE|TOP_K|TOP_P|BACKEND|RABBIT_HOST|VC_TASK_INDEX|VC_TASK_NUM|GPU|CUDA" | sort' >> /app/diagnose.sh && \
    echo 'echo "[8] pip packages:"' >> /app/diagnose.sh && \
    echo 'pip list 2>/dev/null | grep -iE "torch|vllm|transformers|pika|cuda"' >> /app/diagnose.sh && \
    echo 'echo "===== END DIAGNOSTIC ====="' >> /app/diagnose.sh && \
    chmod +x /app/diagnose.sh

WORKDIR /app
