FROM python:3.9

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
RUN pip install requests pysnooper pandas

COPY job-template/job/labelstudio-import/ /app/

WORKDIR /app
ENV PYTHONPATH=/app:${PYTHONPATH:-}

ENTRYPOINT ["python3", "launcher.py"]
