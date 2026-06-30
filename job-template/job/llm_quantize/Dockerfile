FROM 10.121.177.20:8082/mlops/gptqmodel:2.5.0

# 安装 torchvision 并升级 transformers/huggingface-hub（已有 torch，跳过依赖避免重复安装 CUDA 包）
RUN pip install --no-cache-dir torchvision --no-deps && \
    pip install --no-cache-dir --upgrade transformers huggingface-hub --no-deps

COPY launcher.py /app/launcher.py
WORKDIR /app

ENTRYPOINT ["python3", "/app/launcher.py"]
