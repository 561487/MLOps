FROM 10.121.177.20:8082/mlops/offline-predict-launcher:20260705_v2

# ============================================================
# 增量构建：基于上一版镜像，只覆盖更新文件，构建速度快
# ============================================================

# 覆盖更新文件
COPY model_offline_predict/* /app/
COPY llm_offline_predict/image_tags.conf /app/
COPY pkgs /app/job/pkgs

WORKDIR /app
ENV PYTHONPATH=/app:$PYTHONPATH

ENTRYPOINT ["python3", "launcher-rabbitmq.py"]
