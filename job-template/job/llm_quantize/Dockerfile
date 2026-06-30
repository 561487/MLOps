FROM python:3.11-slim
RUN pip install --upgrade pip
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir transformers datasets accelerate && \
    pip install --no-cache-dir llmcompressor==0.11.0
COPY launcher.py /app/launcher.py
WORKDIR /app
ENTRYPOINT ["python3", "/app/launcher.py"]
