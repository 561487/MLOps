FROM python:3.9-slim

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive

COPY job-template/job/datasetmergesplit/ /app/
WORKDIR /app
ENV PYTHONPATH=/app

ENTRYPOINT ["python3", "launcher.py"]
