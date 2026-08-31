FROM python:3.9-slim

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
RUN pip install --no-cache-dir pyarrow

COPY job-template/job/datasetpreprocess/ /app/
WORKDIR /app
ENV PYTHONPATH=/app

ENTRYPOINT ["python3", "launcher.py"]
