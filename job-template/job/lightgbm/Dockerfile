FROM 10.121.177.20:8082/mlops/nlp-analyze-data:20250601

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive

RUN python3 -m pip config set global.index-url https://mirrors.aliyun.com/pypi/simple || true

# Try install OpenCL runtime; NVIDIA ICD is provided by nvidia-container-toolkit at runtime.
# If apt fails (repo issues), OpenCL won't work but image still builds.
RUN apt-get update -qq 2>/dev/null || true && \
    apt-get install -y -qq --no-install-recommends ocl-icd-libopencl1 clinfo 2>/dev/null || true && \
    rm -rf /var/lib/apt/lists/*

# Create NVIDIA OpenCL ICD so LightGBM can find GPU via OpenCL
RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

RUN python3 -m pip install --no-cache-dir \
    scikit-learn \
    lightgbm==4.6.0 \
    joblib \
    pandas \
    numpy \
    requests \
    "swanlab>=0.8.4"

COPY job/lightgbm/launcher.py /app/launcher.py
COPY common/ /app/common/

WORKDIR /app
ENV PYTHONPATH=/app

RUN python3 -c "import lightgbm as lgb; import swanlab; \
    from common.training_monitor import TrainingMonitor; \
    print('lightgbm:', lgb.__version__); \
    print('swanlab:', swanlab.__version__); \
    print('deps OK')"

ENTRYPOINT ["python3", "launcher.py"]
