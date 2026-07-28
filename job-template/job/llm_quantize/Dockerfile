FROM 10.121.177.20:8082/mlops/gptqmodel:3.1.1

# gptqmodel 7.x imports torchvision. Keep dependency resolution in the base
# image and fail the build early if any runtime engine is missing.
RUN pip install --no-cache-dir modelscope torchvision
RUN pip install --no-cache-dir "swanlab>=0.8.4" psutil
RUN pip uninstall -y nvidia-nvjitlink-cu12 2>/dev/null || true
RUN python3 -c "import awq, bitsandbytes, datasets, gptqmodel, modelscope, psutil, swanlab, torch, transformers"

COPY job-template/job/llm_quantize/launcher.py /app/launcher.py
COPY job-template/common/ /app/common/
WORKDIR /app
ENV PYTHONPATH=/app

ENTRYPOINT ["python3", "/app/launcher.py"]
