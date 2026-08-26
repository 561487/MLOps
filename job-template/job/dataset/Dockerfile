FROM python:3.9

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
RUN pip install pysnooper psutil requests huggingface_hub hf_transfer
RUN sed -i 's|http://deb.debian.org/debian|https://mirrors.aliyun.com/debian|g; s|http://deb.debian.org/debian-security|https://mirrors.aliyun.com/debian-security|g' /etc/apt/sources.list.d/debian.sources
RUN apt-get update && apt-get install -y git-lfs && git lfs install && rm -rf /var/lib/apt/lists/*
RUN pip install modelscope pandas pyarrow
COPY job-template/job/dataset/ /app/

WORKDIR /app
ENV PYTHONPATH=/app:${PYTHONPATH:-}

ENTRYPOINT ["python3", "launcher.py"]

