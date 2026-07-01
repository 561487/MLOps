FROM 10.121.177.20:8082/mlops/gptqmodel:2.5.0

# 安装 torchvision 0.19（无 _meta_registrations，避免算子注册冲突）
RUN pip install --no-cache-dir torchvision==0.19.1+cu124 --no-deps \
    --extra-index-url https://download.pytorch.org/whl/cu124

COPY launcher.py /app/launcher.py
WORKDIR /app

ENTRYPOINT ["python3", "/app/launcher.py"]
