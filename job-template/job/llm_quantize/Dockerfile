FROM python:3.11-slim

# 安装 torch 2.6.0（与 GPTQModel 2.2.0 匹配，cu128 兼容 CUDA 13.2）
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu128

# 安装 gptqmodel 2.2.0 + 依赖（Qwen 官方基准测试验证的版本组合）
RUN pip install --no-cache-dir gptqmodel==2.2.0 autoawq bitsandbytes datasets accelerate transformers==4.51.3

COPY launcher.py /app/launcher.py
WORKDIR /app

ENTRYPOINT ["python3", "/app/launcher.py"]
