# 模型剪枝模板 (model-prune)

基于 Torch-Pruning (DepGraph) 的模型结构化剪枝任务模板。

## 剪枝方法

| 方法 | 参数值 | 说明 |
|------|--------|------|
| **结构化剪枝** | `structural` | 使用 DepGraph 自动管理层间依赖，适合 CV 模型 |
| **LLM 剪枝** | `llm` | 针对大语言模型优化，支持注意力头剪枝和层级移除 |
| **层级移除** | `layer` | 按名称精确移除指定网络层 |

## 参数说明

| 环境变量 | 命令行参数 | 默认值 | 说明 |
|---------|-----------|--------|------|
| `MODEL_PATH` | `--model` | (必填) | 模型路径或 HuggingFace ID |
| `PRUNE_METHOD` | `--method` | `structural` | 剪枝方法 |
| `PRUNE_RATIO` | `--ratio` | `0.3` | 剪枝比例 (0.0~1.0) |
| `EXAMPLE_SHAPE` | `--example_shape` | `1,3,224,224` | 示例输入形状 (仅 structural) |
| `PRUNE_HEADS` | `--prune_heads` | `true` | 是否剪枝注意力头 (仅 llm) |
| `PRUNE_LAYERS` | `--prune_layers` | `false` | 是否剪枝 Transformer 层 (仅 llm) |
| `N_LAYERS_REMOVE` | `--n_layers_remove` | `0` | 移除的 Transformer 层数 (仅 llm) |
| `LAYER_NAMES` | `--layer_names` | `` | 按名称移除的层名,逗号分隔 (仅 layer) |
| `OUTPUT_PATH` | `--output` | `/mnt/admin/models/pruned` | 输出路径 |

## 使用示例

```bash
# 结构化剪枝 (CV 模型)
python3 launcher.py \
    --model /mnt/models/resnet50 \
    --method structural \
    --ratio 0.3

# LLM 结构化剪枝
python3 launcher.py \
    --model /mnt/models/llama-7b \
    --method llm \
    --ratio 0.2 \
    --prune_heads true

# LLM 层级移除
python3 launcher.py \
    --model /mnt/models/llama-7b \
    --method llm \
    --prune_layers true \
    --n_layers_remove 2

# 按名称移除指定层
python3 launcher.py \
    --model /mnt/models/llama-7b \
    --method layer \
    --layer_names "model.layers.0,model.layers.1"
```

## 输出

剪枝完成后，输出目录包含：

- 剪枝后的模型文件
- `prune_manifest.json` — 剪枝报告
