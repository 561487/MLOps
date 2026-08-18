FROM ccr.ccs.tencentyun.com/cube-studio/ubuntu-gpu:cuda11.8.0-cudnn8-python3.9

WORKDIR /whisper

RUN set -eux; \
    for attempt in 1 2 3; do \
        apt-get clean; \
        rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*; \
        apt-get \
            -o Acquire::Retries=5 \
            -o Acquire::http::No-Cache=true \
            -o Acquire::https::No-Cache=true \
            -o Acquire::http::Pipeline-Depth=0 \
            update -y; \
        if apt-get \
            -o Acquire::Retries=5 \
            -o Acquire::http::No-Cache=true \
            -o Acquire::https::No-Cache=true \
            -o Acquire::http::Pipeline-Depth=0 \
            install -y --no-install-recommends ffmpeg libsndfile1; then \
            break; \
        fi; \
        if [ "${attempt}" -eq 3 ]; then \
            exit 1; \
        fi; \
    done; \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY requirements.txt /whisper/requirements.txt
RUN pip install --no-cache-dir -r /whisper/requirements.txt \
    --index-url https://mirrors.aliyun.com/pypi/simple
RUN pip install --no-cache-dir modelscope==1.20.1 \
    --index-url https://mirrors.aliyun.com/pypi/simple

COPY . /whisper/

ARG WHISPER_MODEL_SOURCE=openai-mirror/whisper-base
ENV WHISPER_MODEL_SOURCE=${WHISPER_MODEL_SOURCE}
ENV WHISPER_MODEL_TARGET=/whisper/models/whisper-base
ENV MODELSCOPE_CACHE=/tmp/modelscope-cache
RUN set -eux; \
    for attempt in 1 2 3; do \
        if python /whisper/download_model.py; then \
            break; \
        fi; \
        if [ "${attempt}" -eq 3 ]; then \
            exit 1; \
        fi; \
        sleep 5; \
    done; \
    rm -rf /tmp/modelscope-cache

ENV TZ=Asia/Shanghai
ENV MODELPATH=/whisper/models/whisper-base
ENV KUBEFLOW_MODEL_NAME=whisper-base
ENV KUBEFLOW_MODEL_VERSION=v1
ENV MAX_AUDIO_MB=100
ENV MAX_AUDIO_SECONDS=600

EXPOSE 8000
ENTRYPOINT ["python", "server.py"]
