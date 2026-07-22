FROM 10.121.177.20:8082/mlops/llama_factory:20250712

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive

# ── SwanLab 监控依赖 ──
RUN pip3 install --no-cache-dir "swanlab>=0.8.4" psutil

# ── 复制代码 ──
COPY job/llama_factory/ /app/
COPY common/ /app/common/
WORKDIR /app
ENV PYTHONPATH=/app

ENTRYPOINT ["python3", "launcher.py"]
