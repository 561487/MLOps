FROM 10.121.177.20:8082/mlops/gptqmodel:2.5.0

# 安装 torchvision（匹配 torch 2.5.1）并升级 transformers/huggingface-hub
RUN pip install --no-cache-dir torchvision==0.20.1 --no-deps && \
    pip install --no-cache-dir --upgrade transformers huggingface-hub --no-deps

COPY launcher.py /app/launcher.py
WORKDIR /app

ENTRYPOINT ["python3", "/app/launcher.py"]
