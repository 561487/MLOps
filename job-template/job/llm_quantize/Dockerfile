FROM python:3.11-slim

# 安装 PyTorch（LLM Compressor 需要 torch，Linux x86_64 默认带 CUDA）
RUN pip install --no-cache-dir torch

# LLM Compressor + 依赖
RUN pip install --no-cache-dir llmcompressor && \
    pip install --no-cache-dir datasets accelerate transformers

# 预下载校准数据集（避免 Pod 运行时联网）
RUN python3 -c "\
from datasets import load_dataset; \
ds = load_dataset('wikitext2', 'wikitext-2-raw-v1', split='train', cache_dir='/app/datasets'); \
ds.save_to_disk('/app/datasets/wikitext2'); \
print('Dataset cached to /app/datasets/wikitext2')"

COPY launcher.py /app/launcher.py
WORKDIR /app

ENTRYPOINT ["python3", "/app/launcher.py"]
