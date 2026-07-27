# LLM 模型量化节点（model-quantization）

该节点用于对 HuggingFace 格式的大语言模型执行离线量化。任务运行在 GPU 节点，输出模型和 `quant_manifest.json` 结果清单。

## 支持的方法

| 方法 | 引擎 | 位宽 | 校准数据 | 适用场景 |
|---|---|---:|---|---|
| GPTQ | GPTQModel | 2/3/4/8 bit | 必需 | 精度优先、可接受较长量化时间 |
| AWQ | AutoAWQ | 4 bit | 必需 | 推理性能与精度平衡 |
| BnB | bitsandbytes | 4/8 bit | 不需要 | 快速试用或 Transformers 推理 |

> AWQ 当前只支持 4 bit。平台表单提供 4/8 bit，选择 AWQ + 8 bit 时任务会在加载模型前给出明确错误。

## 参数

| 环境变量 | 参数 | 默认值 | 说明 |
|---|---|---|---|
| `MODEL_PATH` | `--model` | 必填 | 本地/PVC 模型目录或 HuggingFace Model ID |
| `OUTPUT_PATH` | `--output` | `/mnt/admin/models/quant` | 输出目录，不能与输入目录相同 |
| `QUANT_METHOD` | `--method` | `gptq` | `gptq`、`awq` 或 `bnb` |
| `QUANT_BITS` | `--bits` | `4` | 位宽，受方法约束 |
| `QUANT_GROUP_SIZE` | `--group_size` | `128` | GPTQ 分组大小：32/64/128/256 |
| `QUANT_DATASET` | `--dataset` | `wikitext2` | 校准集目录、CSV/JSON/JSONL/TXT 文件或 HF 数据集名 |
| `QUANT_NSAMPLES` | `--nsamples` | `128` | 校准样本数，1–10000 |
| `QUANT_FORCE` | `--force` | `false` | 是否允许复用非空输出目录 |

本地数据集会依次从 `models-storage` 和 `models-share-volume` 查找。支持的文本列名包括 `text`、`content`、`prompt`、`instruction`、`question` 和 `input`。找不到 `wikitext2` 时会使用内嵌离线样本，避免任务因外网不可用而直接失败。

## 示例

```bash
python3 /app/launcher.py \
  --model /mnt/storage/models-share-volume/models/Qwen2.5-0.5B-Instruct \
  --output /mnt/storage/models-share-volume/output/qwen-gptq \
  --method gptq --bits 4 --group_size 128 \
  --dataset wikitext2 --nsamples 128

python3 /app/launcher.py \
  --model /mnt/storage/models-share-volume/models/Qwen2.5-0.5B-Instruct \
  --output /mnt/storage/models-share-volume/output/qwen-awq \
  --method awq --bits 4 --dataset /mnt/storage/models-storage/datasets/calibration

python3 /app/launcher.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output /mnt/storage/models-share-volume/output/qwen-bnb \
  --method bnb --bits 4
```

## 运行状态与结果

日志会输出 `validate`、`calibration`、`load_model`、`quantize`、`complete`/`failed` 等 JSON 事件，便于平台实时展示阶段状态。

`quant_manifest.json` 包含方法、参数、依赖版本、开始/结束时间、耗时、产物文件数和总字节数；失败时记录异常类型和堆栈。清单采用临时文件原子替换，避免任务中断留下半份 JSON。

若输出目录已有成功清单且未设置 `--force true`，节点会幂等跳过。若目录非空但没有成功清单，节点会拒绝覆盖并提示人工确认。
