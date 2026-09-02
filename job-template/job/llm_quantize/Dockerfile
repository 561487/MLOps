# 模型量化 Runtime（gptqmodel）—— 依赖显式锁定版（r2）
#
# 历史问题（2026-08-21 审计确认）：tag 3.3.0 / 3.1.1 为手工代号，与内部依赖版本无关。
#   gptqmodel:3.3.0 内部实际环境（docker run 实测）：
#     Python 3.11.15 / CUDA 13.0 / PyTorch 2.12.1+cu130 / Triton 3.7.1
#     GPTQModel 7.1.0 / Transformers 5.12.1 / Accelerate 1.14.0 / Datasets 5.0.0
#     AutoAWQ 0.2.9 / BitsAndBytes 0.49.2 / Modelscope 1.39.1 / SwanLab 0.9.3
#   该组合已实测加载 Qwen3.5-2B 成功（真实量化 Pod 日志）。
#
# r2 = r1 + GPTQ 校准 batch_size=1：
#   Qwen3.5-2B GPTQ 4bit 量化在 model.quantize(texts, batch_size=2) 时报
#   "The expanded size of the tensor (341) must match the existing size (2)"，
#   定位为 batch=2 校准数据按批 pad 成 [B, S] 2D attention_mask，在 Qwen3.5
#   sdpa 路径上 2D mask 广播失败；改为 batch_size=1 逐样本校准（见 launcher.py）。
#   依赖组合与 r1 完全一致，仅 launcher.py 一处修改。
#
# 构建目标：
#   1. 依赖显式锁定（requirements.txt 全部 == 固化，可复现）
#   2. 命名纠正：tag = <GPTQModel>-py<python>-cu<cuda>-r<revision>
#      （7.1.0-py311-cu130-r2，如实表达实际环境）
#   3. 基于 3.3.0 构建（= 历史 3.1.1 + 平台附加依赖 modelscope/torchvision/swanlab/psutil），
#      层已在本机缓存，pip 锁定版本均已满足 → 构建增量最小；旧镜像 3.3.0/3.1.1/r1 保留。
FROM 10.121.177.20:8082/mlops/gptqmodel:3.3.0

# 依赖显式锁定：顶层包由 requirements 锁定，pip 检测已满足即跳过，不重复下载。
# --no-deps：依赖树由基础镜像固化（例如 protobuf 6.33.6 同时被 gptqmodel 7.1.0 与
# swanlab 0.9.3 实际使用，二者元数据版本区间不相交，重新解析会失败；
# 该组合已实测运行正常，故不做二次依赖解析，仅锁定顶层版本）。
# GPU 计算栈（torch 2.12.1+cu130 / torchvision 0.27.1 / triton 3.7.1）由基础镜像提供。
COPY job-template/job/llm_quantize/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --no-deps -r /app/requirements.txt

# 构建期版本断言校验：全部关键依赖可导入，且实际版本与锁定一致
# （基础镜像 tag 被覆盖导致版本漂移时，构建直接失败，保证可复现）
RUN python3 -c "\
import importlib.metadata as md; \
import torch, torchvision, triton; \
expect = {'gptqmodel': '7.1.0', 'transformers': '5.12.1', 'accelerate': '1.14.0', \
          'datasets': '5.0.0', 'autoawq': '0.2.9', 'bitsandbytes': '0.49.2', \
          'modelscope': '1.39.1', 'swanlab': '0.9.3'}; \
bad = [f'{pkg}=={md.version(pkg)}(expect {v})' for pkg, v in expect.items() if md.version(pkg) != v]; \
assert not bad, 'version mismatch: ' + '; '.join(bad); \
assert torch.__version__ == '2.12.1+cu130', f'torch {torch.__version__} != 2.12.1+cu130'; \
assert torch.version.cuda == '13.0', f'cuda {torch.version.cuda} != 13.0'; \
assert torchvision.__version__ == '0.27.1+cu130', f'torchvision {torchvision.__version__} != 0.27.1+cu130'; \
assert triton.__version__ == '3.7.1', f'triton {triton.__version__} != 3.7.1'; \
print('VERSION_LOCK_OK')"

COPY job-template/job/llm_quantize/launcher.py /app/launcher.py
COPY job-template/common/ /app/common/
WORKDIR /app
ENV PYTHONPATH=/app

ENTRYPOINT ["python3", "/app/launcher.py"]
