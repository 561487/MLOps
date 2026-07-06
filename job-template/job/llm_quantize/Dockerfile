FROM 10.121.177.20:8082/mlops/gptqmodel:3.0.0

# 安装 torchvision（gptqmodel 7.x 导入需要）
RUN pip install --no-cache-dir torchvision

# 清理旧 CUDA 12 库，避免 bitsandbytes 冲突（当前 torch 2.12.1 已用 cu130）
RUN pip uninstall -y nvidia-nvjitlink-cu12 2>/dev/null; exit 0

COPY launcher.py /app/launcher.py
WORKDIR /app

ENTRYPOINT ["python3", "/app/launcher.py"]
