FROM python:3.9

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

COPY job-template/job/audio-process/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir imageio-ffmpeg \
    && python3 - <<'PY'
import os
import imageio_ffmpeg

ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
link_path = "/usr/local/bin/ffmpeg"

if os.path.exists(link_path):
    os.remove(link_path)

os.symlink(ffmpeg_path, link_path)
print("ffmpeg linked to:", ffmpeg_path)
PY

COPY job-template/job/audio-process/ /app/

WORKDIR /app

ENTRYPOINT ["python3", "launcher.py"]