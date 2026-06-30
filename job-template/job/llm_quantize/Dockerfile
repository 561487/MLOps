FROM 10.121.177.20:8082/mlops/gptqmodel:2.5.0

# 升级 transformers 并安装 torchvision（Qwen3.5 多模态需要）
RUN pip install --no-cache-dir --upgrade transformers torchvision

COPY launcher.py /app/launcher.py
WORKDIR /app

ENTRYPOINT ["python3", "/app/launcher.py"]
