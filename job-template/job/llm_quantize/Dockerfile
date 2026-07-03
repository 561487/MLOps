FROM 10.121.177.20:8082/mlops/gptqmodel:3.0.0

# 安装 gptqmodel 后 torch 可能被升级到 cu130，重新降级到 cu128（匹配驱动 CUDA 12.8）
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 清理旧 CUDA 12/13 库冲突
RUN pip uninstall -y nvidia-nvjitlink-cu12 nvidia-nvjitlink-cu13 2>/dev/null; exit 0

COPY launcher.py /app/launcher.py
WORKDIR /app

ENTRYPOINT ["python3", "/app/launcher.py"]
