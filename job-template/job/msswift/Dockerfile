# 基础镜像由 build_base.sh 构建，包含 PyTorch、DeepSpeed、ms-swift、Megatron 与运行依赖
FROM 10.121.177.20:8082/mlops/msswift-base:4.5.0-py311-cu128-r1
RUN pip install --no-cache-dir pysnooper "swanlab>=0.8.4" psutil
COPY job-template/job/msswift/launcher.py /mnt/
COPY job-template/job/msswift/swift_training_wrapper.py /mnt/
COPY job-template/job/pkgs /mnt/job/pkgs
COPY job-template/common/ /mnt/common/
WORKDIR /mnt
ENV PYTHONPATH=/mnt
ENTRYPOINT ["python3", "launcher.py"]

