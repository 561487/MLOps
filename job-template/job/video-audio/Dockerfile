# 这个下面包含了rayproject/ray:nightly  和  rayproject/ray:nightly-aarch64
FROM ccr.ccs.tencentyun.com/cube-studio/ray:nightly
USER root
ARG TARGETARCH
ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive

# 阿里云源近期索引校验失败；构建只用清华二进制源，不含 deb-src / proposed
RUN set -eux; \
    if [ "${TARGETARCH}" = "arm64" ]; then \
      MIRROR="https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports"; \
    else \
      MIRROR="https://mirrors.tuna.tsinghua.edu.cn/ubuntu"; \
    fi; \
    printf '%s\n' \
      "deb ${MIRROR} jammy main restricted universe multiverse" \
      "deb ${MIRROR} jammy-updates main restricted universe multiverse" \
      "deb ${MIRROR} jammy-backports main restricted universe multiverse" \
      "deb ${MIRROR} jammy-security main restricted universe multiverse" \
      > /etc/apt/sources.list; \
    printf '%s\n' \
      'Acquire::http::Pipeline-Depth "0";' \
      'Acquire::http::No-Cache "true";' \
      'Acquire::BrokenProxy "true";' \
      'Acquire::Retries "3";' \
      > /etc/apt/apt.conf.d/99fixbadproxy; \
    rm -rf /var/lib/apt/lists/*; \
    apt-get update -y; \
    apt-get install -y --no-install-recommends \
      vim gnupg2 rsync jq wget git dnsutils iputils-ping net-tools curl \
      mysql-client locales zip software-properties-common; \
    apt-get install -y --no-install-recommends \
      python3-dev gcc automake autoconf libtool make ffmpeg build-essential; \
    rm -rf /var/lib/apt/lists/*

# 安装pip库
RUN pip install kubernetes==25.3.0 pysnooper psutil requests numpy  pyinstaller argparse cython

ENV RAY_CLIENT_SERVER_MAX_THREADS=1000

# 安装当前代码
USER root
COPY job/video-audio/ /app/
COPY job/pkgs /app/job/pkgs

ENV PYTHONPATH=/app

WORKDIR /app

ENTRYPOINT ["python", "start_download.py"]
