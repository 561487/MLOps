# RTX 5090 D (Blackwell, sm_120) 训练基础镜像
# CUDA 12.8+ 才能编进 sm_120 kernel；torch 2.11.0+cu128 已包含 sm_120
#
# 构建：bash build.sh

ARG FROM_IMAGES=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
FROM ${FROM_IMAGES}

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=zh_CN.UTF-8
ENV LC_ALL=zh_CN.UTF-8
ENV LANGUAGE=zh_CN.UTF-8

COPY sources.list-ubuntu22.04.amd64 /etc/apt/sources.list

# 规避镜像站同步中的 Hash Sum mismatch（by-hash / pipeline / 脏 lists）
RUN printf '%s\n' \
        'Acquire::http::Pipeline-Depth "0";' \
        'Acquire::http::No-Cache "true";' \
        'Acquire::BrokenProxy "true";' \
        'Acquire::By-Hash "no";' \
        'APT::Get::Fix-Missing "true";' \
      > /etc/apt/apt.conf.d/99fix-hash \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common vim apt-transport-https gnupg2 \
        rsync jq wget git dnsutils iputils-ping net-tools curl \
        locales zip unzip ca-certificates \
        python3.11 python3.11-dev python3.11-venv \
        fonts-wqy-microhei fonts-wqy-zenhei xfonts-wqy \
    && locale-gen zh_CN zh_CN.UTF-8 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && wget -q https://bootstrap.pypa.io/get-pip.py \
    && python get-pip.py --ignore-installed \
    && rm -f get-pip.py \
    && ln -sf /usr/local/bin/pip /usr/bin/pip \
    && echo "alias ll='ls -alF'" >> ~/.bashrc \
    && echo "alias la='ls -A'" >> ~/.bashrc \
    && echo "alias vi='vim'" >> ~/.bashrc \
    && rm -rf /var/lib/apt/lists/*

# 必须走 pytorch cu128 源；阿里云/PyPI 默认 wheel 没有 sm_120
RUN pip install --no-cache-dir \
        torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
        --index-url https://download.pytorch.org/whl/cu128

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple \
    && pip install --no-cache-dir \
        pysnooper psutil requests numpy pyinstaller argparse \
        pika Pillow tensorboardX
