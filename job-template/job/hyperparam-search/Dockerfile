FROM python:3.9

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive

ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

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
    "swanlab[dashboard]" \
    requests

# 复制算子代码和公共模块（构建上下文为 job-template/ 根目录）
COPY job/hyperparam-search/ /app/
COPY common/ /app/common/

WORKDIR /app
ENV PYTHONPATH=/app

RUN ls -lah /app && test -f /app/launcher.py && \
    python3 -c "import numpy, pandas, scipy, sklearn, joblib, xgboost, lightgbm, swanlab, requests; print('deps ok')"

CMD ["python3", "launcher.py"]
