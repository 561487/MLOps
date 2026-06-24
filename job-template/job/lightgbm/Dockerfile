FROM 10.121.177.20:8082/mlops/nlp-analyze-data:20250601

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive

RUN python3 -m pip config set global.index-url https://mirrors.aliyun.com/pypi/simple || true

RUN python3 -m pip install --no-cache-dir \
    scikit-learn \
    lightgbm \
    joblib

COPY . /app/

WORKDIR /app
ENV PYTHONPATH=/app:${PYTHONPATH:-}

ENTRYPOINT ["python3", "launcher.py"]