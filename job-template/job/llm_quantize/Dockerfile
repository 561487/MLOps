FROM python:3.11-slim
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple torch
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple llmcompressor && \
    pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple datasets accelerate transformers
COPY launcher.py /app/launcher.py
WORKDIR /app
ENTRYPOINT ["python3", "/app/launcher.py"]
