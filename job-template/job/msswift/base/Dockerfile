FROM 10.121.177.20:8082/mlops/msswift-base:4.5.0-py311-cu128-r1

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive

RUN pip install --no-cache-dir \
    "mistral-common==1.11.7" "pysnooper==1.2.3" \
    "swanlab==0.9.4" "psutil==7.2.2" && \
    pip check
