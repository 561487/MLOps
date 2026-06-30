FROM 10.121.177.20:8082/mlops/gptqmodel:2.5.0

# 升级 transformers 以支持 Qwen3.5
RUN pip install --no-cache-dir --upgrade transformers

COPY launcher.py /app/launcher.py
WORKDIR /app

ENTRYPOINT ["python3", "/app/launcher.py"]
