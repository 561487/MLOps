# 基础镜像由 build_base.sh 构建，包含 PyTorch、DeepSpeed、ms-swift、Megatron 与运行依赖
FROM 10.121.177.20:8082/mlops/msswift-base:20260710-v1
RUN pip install --no-cache-dir pysnooper
COPY job-template/job/msswift/launcher.py /app/
COPY job-template/job/pkgs /app/job/pkgs
WORKDIR /app
ENV PYTHONPATH=/app
ENTRYPOINT ["python3", "launcher.py"]

