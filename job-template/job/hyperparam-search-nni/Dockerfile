FROM python:3.9

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive

RUN python3 -m pip config set global.index-url https://mirrors.aliyun.com/pypi/simple || true

RUN /usr/local/bin/python -m pip install --upgrade pip && \
    pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scipy==1.11.4 \
    scikit-learn==1.3.2 \
    joblib==1.3.2 \
    threadpoolctl==3.5.0 \
    xgboost \
    lightgbm \
    "swanlab>=0.8.4" \
    psutil \
    requests

# Build context: job-template/
COPY job/hyperparam-search-nni/launcher.py /app/launcher.py
COPY common/ /app/common/

WORKDIR /app
ENV PYTHONPATH=/app

RUN python3 -c "import numpy, pandas, scipy, sklearn, joblib, xgboost, lightgbm, swanlab, psutil; from common.training_monitor import TrainingMonitor; print('deps OK')"

CMD ["python3", "launcher.py"]
