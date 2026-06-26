FROM python:3.9

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt update -y && apt install -y --no-install-recommends \
    vim wget curl git net-tools \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip
RUN /usr/local/bin/python -m pip install --upgrade pip

# 安装 Python 依赖
COPY job/feature-process/requirement.txt /tmp/requirement.txt
RUN pip install -r /tmp/requirement.txt

# 复制代码
COPY job/feature-process/ /app/

WORKDIR /app
ENV PYTHONPATH=/app:${PYTHONPATH:-}

ENTRYPOINT ["python3", "launcher.py"]
