FROM python:3.9-slim

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

COPY job-template/job/dataset-convert/ /app/
WORKDIR /app
ENV PYTHONPATH=/app
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3", "launcher.py"]
