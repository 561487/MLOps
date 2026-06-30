FROM 10.121.177.20:8082/mlops/gptqmodel:2.5.0

# 安装与 torch 2.5.1+cu124 匹配的 torchvision（不要升级 transformers，4.57.6 已支持 Qwen3.5）
RUN pip install --no-cache-dir torchvision==0.20.1+cu124 --no-deps \
    --extra-index-url https://download.pytorch.org/whl/cu124

COPY launcher.py /app/launcher.py
WORKDIR /app

ENTRYPOINT ["python3", "/app/launcher.py"]
