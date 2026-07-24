FROM python:3.10-slim

ENV TZ=Asia/Shanghai \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple \
    && pip install \
        "modelscope>=1.31,<2" \
        "pyarrow>=14,<20" \
        "ijson>=3.2,<4"

COPY job/serverops-data-curation/ /app/

WORKDIR /app
ENV PYTHONPATH=/app

ENTRYPOINT ["python3", "launcher.py"]
