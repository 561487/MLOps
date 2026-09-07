# job-template/job/model-evaluate/Dockerfile
# 业务镜像：基于 model-evaluate:base-py310-cu128-r1，仅复制/安装入口脚本，日常改代码走此 Dockerfile。
#
# 构建（项目根目录）:
#   bash job-template/job/model-evaluate/build.sh
#
# 首次或升级依赖时先构建 base:
#   bash job-template/job/model-evaluate/build-base.sh
ARG BASE_IMAGE=10.121.177.20:8082/mlops/model-evaluate:base-py310-cu128-r1
FROM ${BASE_IMAGE}

LABEL maintainer="MLops"
LABEL description="OpenCompass LLM Evaluation Tool - MLops Job Template (app layer)"

# ── 复制入口脚本 ──
COPY job-template/job/model-evaluate/src/ /app/
RUN chmod +x /app/*.py

# ── 安装 sitecustomize.py 到 site-packages（Python 启动时自动加载） ──
# 为 MsDataset.load 注入 trust_remote_code=True，在 opencompass 子进程中也生效
# 使用 python3 -S + sysconfig，避免 import site 时加载 /app/sitecustomize 污染 stdout
RUN python3 -S -c "\
import sysconfig, shutil; \
dst = sysconfig.get_paths()['purelib'] + '/sitecustomize.py'; \
shutil.copy('/app/sitecustomize.py', dst); \
print('sitecustomize.py installed to:', dst)"

ENTRYPOINT ["python3", "/app/run_evaluation.py"]
