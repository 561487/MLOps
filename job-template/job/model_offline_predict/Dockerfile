FROM 10.121.177.20:8082/mlops/offline-predict-launcher:20260714

# ============================================================
# 增量构建：基于上一版镜像，只覆盖更新文件，构建速度快
# ============================================================

# 覆盖更新文件
COPY model_offline_predict/* /app/
COPY pkgs /app/job/pkgs

WORKDIR /app
ENV PYTHONPATH=/app:$PYTHONPATH
# worker 镜像 tag 直接固化在镜像里（部署时可用环境变量覆盖）
ENV LLM_OFFLINE_PREDICT=10.121.177.20:8082/mlops/model-offline-predict:1.0.0-20260825-r1

ENTRYPOINT ["python3", "launcher-rabbitmq.py"]
