FROM 10.121.177.20:8082/mlops/llama_factory:20250617

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive

# 直接复制代码（依赖已在基础镜像中）
COPY . /app/
WORKDIR /app
ENV PYTHONPATH=/app

ENTRYPOINT ["python3", "launcher.py"]