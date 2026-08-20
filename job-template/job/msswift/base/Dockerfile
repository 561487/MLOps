# Keep this image independently rebuildable. Do not inherit from an older
# msswift-base revision, because Harbor only retains the current base tag.
FROM modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.9.0-vllm0.13.0-modelscope1.33.0-swift3.12.5

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive

RUN curl -fsSL https://github.com/stern/stern/releases/download/v1.26.0/stern_1.26.0_linux_amd64.tar.gz \
    | tar xz -C /usr/local/bin stern && \
    chmod +x /usr/local/bin/stern

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple && \
    pip install --no-cache-dir --upgrade \
      "kubernetes==25.3.0" "psutil==7.2.2" "requests==2.34.2" \
      "ms-swift==4.5.0" "transformers==5.12.1" \
      "qwen-vl-utils==0.0.14" "peft==0.19.1" \
      "datasets==4.8.4" "liger-kernel==0.8.1" \
      "deepspeed==0.18.9" && \
    pip install --no-cache-dir "flash-linear-attention==0.5.2" --no-build-isolation && \
    pip install --no-cache-dir \
      "causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d@3a4c88e599cd7dec333cac727bd59f2a41a8aad5" \
      --no-build-isolation && \
    pip install --no-cache-dir "flash-attn==2.8.3" --no-build-isolation && \
    pip uninstall -y vllm lmdeploy evalscope outlines outlines-core && \
    pip install --no-cache-dir "opencv-python-headless==4.11.0.86"

RUN pip install --no-cache-dir \
    "mistral-common==1.11.7" "pysnooper==1.2.3" \
    "swanlab==0.9.4" "psutil==7.2.2" && \
    pip check
