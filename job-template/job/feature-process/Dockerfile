FROM python:3.9

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive


ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/


COPY job/feature-process/requirement.txt /tmp/requirement.txt
RUN pip install -r /tmp/requirement.txt

# 复制代码
COPY job/feature-process/ /app/

WORKDIR /app
ENV PYTHONPATH=/app

#ENTRYPOINT ["python3", "launcher.py"]
