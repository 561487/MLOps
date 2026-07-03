# 模型蒸馏节点 (Model Distillation)

基于 EasyDistill 的大语言模型知识蒸馏节点，支持白盒蒸馏（KL散度匹配）和黑盒蒸馏（SFT蒸馏）。

## 使用方式

在 Pipeline 编辑器中拖入"大模型蒸馏"节点，配置参数后运行。

## 参数说明

### 模型配置

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:--:|--------|------|
| `--teacher_model` | str | 是 | — | Teacher 模型，ModelScope ID 或本地路径 |
| `--student_model` | str | 是 | — | Student 模型，ModelScope ID 或本地路径 |
| `--output_path` | str | 是 | — | 蒸馏后模型保存路径 |
| `--distill_type` | 下拉 | 是 | whitebox | `whitebox`（白盒）/ `blackbox`（黑盒） |

### 数据配置

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:--:|--------|------|
| `--data_path` | str | 是 | — | JSONL 训练数据路径 |
| `--max_samples` | str | 否 | 0 | 最大样本数，0=全部 |
| `--data_synthesis` | 下拉 | 否 | false | 是否启用 EasyDistill 内置数据合成 |

### 蒸馏参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:--:|--------|------|
| `--temperature` | str | 否 | 4.0 | softmax 温度，越高标签越平滑 |
| `--alpha` | str | 否 | 0.5 | 硬标签权重，0~1，仅白盒模式 |
| `--epochs` | str | 否 | 3 | 训练轮数 |
| `--batch_size` | str | 否 | 4 | 每批样本数（LLM 显存消耗大） |
| `--learning_rate` | str | 否 | 2e-5 | 学习率 |

## 蒸馏原理

### 白盒蒸馏 (whitebox)

**Loss = α × CE(student_logits, labels) + (1-α) × T² × KL(teacher_softmax, student_softmax)**

- **T (Temperature)**: 越高，Teacher 输出的概率分布越平滑，非正确类别的信息更丰富
- **α (Alpha)**: 平衡硬标签（正确答案）和软标签（Teacher 知识）的权重

### 黑盒蒸馏 (blackbox)

Teacher 生成高质量回答 → 结合原始数据 SFT 训练 Student。适用于 Teacher 权重不可获取的场景（如 GPT-4、闭源商业模型）。

## 依赖

- EasyDistill: https://github.com/modelscope/easydistill
- PyTorch >= 2.0, CUDA >= 11.8

## 资源建议

- LLM 蒸馏即使 batch_size=4 也需要较大显存（7B 模型约需 40GB+），建议使用 A100/V100
- 建议将 `MODELSCOPE_CACHE` 指向挂载卷，避免重复下载模型
