FROM python:3.9

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive

# 设置 pip 全局镜像源
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/

RUN /usr/local/bin/python -m pip install --upgrade pip && \
    pip install opencv-python-headless Pillow numpy pandas

COPY . /app/

WORKDIR /app
ENV PYTHONPATH=/app


