FROM 10.121.177.20:8082/mlops/gptqmodel:2.5.0

# 额外安装剪枝依赖（gptqmodel 镜像已有 torch/transformers/datasets）
RUN pip install --no-cache-dir torch-pruning

COPY launcher.py /app/launcher.py
WORKDIR /app

ENTRYPOINT ["python3", "/app/launcher.py"]
