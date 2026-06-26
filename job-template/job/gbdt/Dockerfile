FROM python:3.9

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive

# 设置 pip 全局镜像源
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

RUN /usr/local/bin/python -m pip install --upgrade pip && \
    pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scipy==1.11.4 \
    scikit-learn==1.3.2 \
    joblib==1.3.2 \
    threadpoolctl==3.5.0

COPY . /app/

WORKDIR /app
ENV PYTHONPATH=/app

CMD ["python3", "launcher.py"]