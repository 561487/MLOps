FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

ARG TARGETARCH=amd64

RUN sed -i \
      -e 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' \
      -e 's|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' \
      /etc/apt/sources.list \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get -o Acquire::Retries=5 -o Acquire::http::No-Cache=true update \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      libaio-dev \
      wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/stern/stern/releases/download/v1.25.0/stern_1.25.0_linux_${TARGETARCH}.tar.gz \
    && tar -zxvf stern_1.25.0_linux_${TARGETARCH}.tar.gz \
    && rm stern_1.25.0_linux_${TARGETARCH}.tar.gz \
    && chmod +x stern \
    && mv stern /usr/bin/stern

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir \
      accelerate==0.34.2 \
      argparse \
      datasets==2.21.0 \
      deepspeed==0.14.4 \
      kubernetes==25.3.0 \
      peft==0.12.0 \
      psutil \
      pysnooper \
      requests \
      sentencepiece \
      transformers==4.44.2 \
    && python -c 'import accelerate, datasets, deepspeed, peft, torch, transformers; assert torch.__version__.startswith("2.3.1"); assert deepspeed.__version__ == "0.14.4"; assert transformers.__version__ == "4.44.2"' \
    && mkdir -p /root/.triton/autotune

COPY job/deepspeed/ /app/
COPY job/pkgs /app/job/pkgs
WORKDIR /app
ENV PYTHONPATH=/app

ENTRYPOINT ["python3", "launcher.py"]
