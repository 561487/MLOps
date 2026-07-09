FROM 10.121.177.20:8082/mlops/gptqmodel:3.1.1

# 安装剪枝依赖（基础镜像已有 torch/transformers/datasets）
RUN pip install --no-cache-dir torch-pruning

COPY launcher.py /app/launcher.py
WORKDIR /app

ENTRYPOINT ["python3", "/app/launcher.py"]
