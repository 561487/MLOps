FROM 10.121.177.20:8082/mlops/gptqmodel:3.0.0

# gptqmodel 7.x imports torchvision. Keep dependency resolution in the base
# image and fail the build early if any runtime engine is missing.
RUN pip install --no-cache-dir torchvision
RUN pip uninstall -y nvidia-nvjitlink-cu12 2>/dev/null || true
RUN python3 -c "import awq, bitsandbytes, datasets, gptqmodel, torch, transformers"

COPY launcher.py /app/launcher.py
COPY tests /app/tests
WORKDIR /app
RUN python3 -m unittest discover -s /app/tests -v

ENTRYPOINT ["python3", "/app/launcher.py"]
