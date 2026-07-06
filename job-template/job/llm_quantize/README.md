# LLM 量化模板 (llm-quantize)

基于 LLM Compressor 的 LLM 量化任务模板，支持三种量化方法：

| 方法 | 引擎 | 特点 |
|------|------|------|
| **GPTQ** | LLM Compressor | Hessian 矩阵逐层量化，需校准数据，精度最高 |
| **AWQ** | LLM Compressor | 激活感知量化，需校准数据，速度快精度高 |
| **RTN** | LLM Compressor | Round-to-Nearest，无需校准数据，适合快速原型 |

## 参数说明

| 环境变量 | 命令行参数 | 默认值 | 说明 |
|---------|-----------|--------|------|
| `MODEL_PATH` | `--model` | (必填) | 待量化模型路径或 HuggingFace ID |
| `QUANT_METHOD` | `--method` | `gptq` | 量化方法: gptq / awq / rtn |
| `QUANT_BITS` | `--bits` | `4` | 量化位数: 4, 8 |
| `QUANT_GROUP_SIZE` | `--group_size` | `128` | GPTQ group size |
| `QUANT_DATASET` | `--dataset` | `wikitext2` | GPTQ 校准数据集 |
| `QUANT_NSAMPLES` | `--nsamples` | `128` | GPTQ 校准样本数 |
| `OUTPUT_PATH` | `--output` | `/mnt/admin/models/quant` | 输出路径 |

## 使用示例

```bash
# GPTQ 4bit 量化
python3 launcher.py \
    --model /mnt/models/qwen-7b \
    --method gptq \
    --bits 4 \
    --output /mnt/models/qwen-7b-gptq

# AWQ 量化
python3 launcher.py \
    --model /mnt/models/qwen-7b \
    --method awq \
    --bits 4

# RTN 快速量化（无需校准数据）
python3 launcher.py \
    --model /mnt/models/qwen-7b \
    --method rtn \
    --bits 4
```

## 输出

量化完成后，输出目录包含：

- 量化后的模型文件（可直接加载到 vLLM 推理）
- `quant_manifest.json` — 量化报告
