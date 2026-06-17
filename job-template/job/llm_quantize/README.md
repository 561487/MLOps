# LLM 量化模板 (llm-quantize)

基于 GPTQModel 的 LLM 量化任务模板，支持三种量化方法：

| 方法 | 引擎 | 特点 |
|------|------|------|
| **GPTQ** | GPTQModel | 后训练量化，需校准数据，精度损失小 |
| **AWQ** | AutoAWQ | 激活感知量化，精度最高，无需校准数据 |
| **bitsandbytes** | BNB | 运行时量化，无需预量化，适合快速原型 |

## 参数说明

| 环境变量 | 命令行参数 | 默认值 | 说明 |
|---------|-----------|--------|------|
| `MODEL_PATH` | `--model` | (必填) | 待量化模型路径或 HuggingFace ID |
| `QUANT_METHOD` | `--method` | `gptq` | 量化方法: gptq / awq / bnb |
| `QUANT_BITS` | `--bits` | `4` | 量化位数: 2, 3, 4, 8 |
| `QUANT_GROUP_SIZE` | `--group_size` | `128` | GPTQ group size |
| `QUANT_DATASET` | `--dataset` | `c4` | GPTQ 校准数据集 |
| `QUANT_NSAMPLES` | `--nsamples` | `128` | GPTQ 校准样本数 |
| `OUTPUT_PATH` | `--output` | `/mnt/admin/models/quant` | 输出路径 |

## 使用示例

```bash
# GPTQ 4bit 量化
python3 launcher.py \
    --model /mnt/models/llama-7b \
    --method gptq \
    --bits 4 \
    --output /mnt/models/llama-7b-gptq

# AWQ 量化
python3 launcher.py \
    --model /mnt/models/llama-7b \
    --method awq \
    --bits 4

# bitsandbytes 快速量化
python3 launcher.py \
    --model /mnt/models/llama-7b \
    --method bnb \
    --bits 4
```

## 输出

量化完成后，输出目录包含：

- 量化后的模型文件（格式取决于量化方法）
- `quant_manifest.json` — 量化报告（含引擎、参数、状态等信息）
