# 大模型评测（model-evaluate）使用文档

## 一、组件说明

`model-evaluate` 是 Cube Studio 的 LLM 评测任务模板，基于 [OpenCompass](https://github.com/open-compass/opencompass) 框架，对 HuggingFace 格式的大语言模型进行自动化评测，输出标准化指标。

### 运行环境

| 项目 | 值 |
|---|---|
| 业务镜像 | `10.121.177.20:8082/mlops/model-evaluate:main-py310-cu128-r12` |
| BASE 镜像 | `10.121.177.20:8082/mlops/model-evaluate:base-py310-cu128-r1` |
| 底层 CUDA | nvidia/cuda:12.8.0-runtime-ubuntu22.04 |
| Python | 3.10 |
| PyTorch | 2.7.0+cu128（支持 RTX 5090 / Blackwell sm_120） |
| Transformers | ≥5.2（支持 Qwen3.5 等新架构） |
| OpenCompass | main 分支（含 Qwen3.5 兼容补丁） |
| 入口命令 | `python3 /app/run_evaluation.py` |
| 挂载卷 | `models-storage(storage):/mnt/storage/models-storage` |

### 核心能力

- 内置 14 个有 OSS 直链的标准评测数据集（C-Eval/MMLU/CMMLU/GSM8K/HumanEval 等，均可自动下载）
- 支持自定义数据集（jsonl/json/csv，自动推断列名/题型/指标）
- 生成式（`_gen`）与困惑度（`_ppl`）两种评测模式
- 自动 GPU 检测、显存诊断、OpenCompass CLI 健康检查
- 评测结果输出 `metric.json` / `eval_summary.json` / `eval_report.csv`

### 输出文件

评测完成后，结果写入 `--output_path` 指定目录：

| 文件 | 内容 |
|---|---|
| `metric.json` | 按 benchmark 分层的评测指标：`eval_results.<benchmark>.score` 为 benchmark 总分，多 subset benchmark（如 ceval/bbh）另有 `details` 子任务明细；`overall_score` = 成功 benchmark 总分等权平均 |
| `eval_summary.json` | 汇总指标（与 metric.json 的 `eval_results` 同源） |
| `eval_report.csv` | benchmark + subset 两层表格化结果 |
| `dataset_status.json` | 各数据集运行状态 + `runs` 映射（dataset -> run_dir -> summary_file） |
| `opencompass_results/<dataset>/<时间戳>/` | OpenCompass 原始输出，每个数据集一次独立运行，目录与数据集一一对应 |

多数据集评测说明：平台对每个选中的内置数据集**单独启动一次 OpenCompass 运行**
（独立 `--work-dir`），单个数据集失败自动跳过、不影响其他数据集；结果解析按
`dataset_status.json` 的 runs 映射逐数据集聚合，不使用"最新 summary"猜测整个任务的结果。

---

## 二、参数设置

参数分为 4 组：**模型配置**、**评测配置**、**自定义数据集**、**高级配置**。

### 1. 模型配置

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|:---:|---|---|
| `--model_path` | str | 是 | `/mnt/storage/models-storage/models/` | HuggingFace 模型路径或本地模型目录。如 `/mnt/storage/models-storage/models/qwen-7b` 或 `Qwen/Qwen-7B-Chat` |
| `--model_name` | str | 是 | — | 模型名称，用于结果标识。如 `qwen-7b-chat` |
| `--model_type` | str | 是 | `hf_chat` | 模型类型：`hf_chat`=对话模型，`hf_base`=基座模型 |
| `--model_version` | str | 否 | `v1` | 模型版本号，用于区分同一模型不同版本的评测结果 |
| `--output_path` | str | 是 | `/mnt/storage/models-storage/output/opencompass` | 评测结果输出目录 |

### 2. 评测配置

| 参数 | 类型 | 必填 | 默认值 | 范围 | 说明 |
|---|---|:---:|---|---|---|
| `--datasets` | list | 是 | `ceval_gen` | 见下表 | 评测数据集，可多选 |
| `--num_gpus` | int | 是 | `1` | 1-8 | 评测使用的 GPU 数量 |
| `--batch_size` | int | 否 | `64` | 1-512 | 推理批次大小，显存不足时减小 |
| `--max_seq_len` | int | 否 | `2048` | 128-8192 | 模型最大输入 token 数 |
| `--max_out_len` | int | 否 | `512` | 64-4096 | 模型最大输出 token 数 |
| `--few_shot` | int | 否 | `0` | 0-10 | 少样本数量，0=零样本（Zero-Shot） |
| `--max_samples` | int | 否 | `0` | 0-999999 | 每数据集样本上限。0=全量；设 N 则每数据集取前 N 条 |
| `--datasets_cache_dir` | str | 否 | — | — | 数据集缓存目录（挂载卷路径），留空则用容器内置缓存 |

#### 内置数据集

后缀说明：`_gen`=生成式评测（需生成文本，支持所有题型），`_ppl`=困惑度评测（仅选择题，更快省显存）。

| 数据集 | 类型 | 说明 |
|---|---|---|
| `ceval_gen` / `ceval_ppl` | 中文综合 | C-Eval 中文综合考试 |
| `cmmlu_gen` / `cmmlu_ppl` | 中文多任务 | CMMLU |
| `mmlu_gen` / `mmlu_ppl` | 英文综合 | MMLU |
| `gsm8k_gen` | 数学 | GSM8K 小学数学 |
| `math_gen` | 数学 | 竞赛数学 |
| `bbh_gen` | 推理 | Big-Bench-Hard |
| `hellaswag_gen` / `hellaswag_ppl` | 常识 | HellaSwag |
| `piqa_gen` / `piqa_ppl` | 物理 | PIQA |
| `winogrande_gen` | 代词 | WinoGrande |
| `lambada_gen` | 语言 | LAMBADA |
| `triviaqa_gen` | 问答 | TriviaQA |
| `agieval_gen` | 考试 | AGIEval |
| `humaneval_gen` | 代码 | HumanEval |
| `mbpp_gen` | 代码 | MBPP |
| `race_gen` / `race_ppl` | 英文阅读 | RACE |
| `commonsenseqa_gen` / `commonsenseqa_ppl` | 常识 | CommonsenseQA |

> 以上 14 个数据集均有 OSS 直链，可直接 `--datasets` 自动下载（首次下载缓存到 `--datasets_cache_dir` 或容器默认缓存）。OBQA/SIQA/StrategyQA/XSum/NQ 等因无 OSS 直链且 ModelScope 加载有兼容性问题，暂未列入。

### 3. 自定义数据集

> **互斥规则**：`--custom_dataset_path` 与 `--datasets` 不可同时填写，否则报错。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|:---:|---|---|
| `--custom_dataset_path` | str | 否 | — | 自定义数据集路径（容器内绝对路径，支持文件或目录）。提供时走自定义评测路径 |
| `--custom_columns` | str(JSON) | 否 | — | 列名覆盖。如 `{"input":"q","target":"ans","choices":"opt1,opt2,opt3,opt4"}` |
| `--custom_metric` | str | 否 | 自动 | 评测指标：`accuracy`/`exact_match`/`bleu`/`rouge`，留空自动推断 |
| `--custom_prompt_template` | str | 否 | 自动 | 自定义 prompt 模板，含 `{input}`/`{A}` 等占位符 |

#### 自定义数据集规则

- **文件格式**：`.json` / `.jsonl` / `.csv`
- **目录场景**：每个文件作为独立数据集，目录内格式必须统一（不可混用 jsonl 和 csv）
- **自动推断**：列名、题型（选择题/生成式）、指标，无需手动配置
- **列名覆盖**：自动推断不准时用 `--custom_columns` 指定
- **指标推断**：选择题→`accuracy`，生成式→`exact_match`

#### `--custom_columns` 字段说明

| 字段 | 含义 | 示例 |
|---|---|---|
| `input` | 问题列名 | `"q"` |
| `target` | 答案列名 | `"ans"` |
| `choices` | 选项列名（逗号分隔，非空=选择题） | `"opt1,opt2,opt3,opt4"` 或 `["opt1","opt2"]` |

### 4. 高级配置

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|:---:|---|---|
| `--model_kwargs` | json | 否 | `{"device_map":"auto","torch_dtype":"bfloat16"}` | 额外模型加载参数；留空用默认值，填写后与默认合并（同名字段用户优先） |

---

## 三、使用示例

### 示例 1：C-Eval 中文综合评测（零样本）

最基础用法，评测模型在 C-Eval 上的表现。

```
模型配置:
  模型路径:   /mnt/storage/models-storage/models/qwen-7b
  模型名称:   qwen-7b-chat
  模型类型:   hf_chat
  模型版本:   v1
  输出路径:   /mnt/storage/models-storage/output/opencompass

评测配置:
  评测数据集:  ceval_gen
  GPU 数量:   1
  Batch Size: 64
  Few-Shot:   0
  每数据集样本数上限: 0   (全量)
```

### 示例 2：多数据集组合评测

同时评测中文+英文+数学能力。

```
模型配置:
  模型路径:   Qwen/Qwen-7B-Chat
  模型名称:   qwen-7b-chat
  模型类型:   hf_chat
  模型版本:   v2

评测配置:
  评测数据集:  ceval_gen, mmlu_gen, gsm8k_gen, cmmlu_gen
  GPU 数量:   1
  Batch Size: 32          (多数据集显存占用大，适当减小)
  最大输入长度: 2048
  最大输出长度: 512
  Few-Shot:   0
```

### 示例 3：自定义数据集 - 选择题（自动推断）

使用本地 jsonl 选择题数据集，列名规范时自动推断。

**数据文件** `/mnt/storage/models-storage/datasets/my_mcq.jsonl`：
```json
{"question": "中国首都是？", "answer": "B", "A": "上海", "B": "北京", "C": "广州", "D": "成都"}
{"question": "1+1=?", "answer": "A", "A": "2", "B": "3", "C": "4", "D": "5"}
```

**参数配置**：
```
自定义数据集:
  自定义数据集路径: /mnt/storage/models-storage/datasets/my_mcq.jsonl
  列名覆盖(JSON): (留空，自动推断)
  评测指标:       (留空，自动推断为 accuracy)
  自定义 Prompt:  (留空，自动用 ANSWER 模板)
```

自动推断结果：问题列=`question`，答案列=`answer`，选项列=`A,B,C,D`，题型=选择题，指标=`accuracy`。

### 示例 4：自定义数据集 - 生成式（列名覆盖）

列名不规范时用 `--custom_columns` 指定。

**数据文件** `/mnt/storage/models-storage/datasets/my_qa.jsonl`：
```json
{"q": "解释什么是机器学习", "ans": "机器学习是让计算机从数据中学习规律的方法"}
{"q": "Python 是什么", "ans": "Python 是一种高级编程语言"}
```

**参数配置**：
```
自定义数据集:
  自定义数据集路径: /mnt/storage/models-storage/datasets/my_qa.jsonl
  列名覆盖(JSON): {"input":"q","target":"ans"}
  评测指标:       exact_match
  自定义 Prompt:  请回答以下问题:\n{input}
```

### 示例 5：目录场景 - 多文件评测

目录内多个 jsonl 文件，每个作为独立数据集。

**目录结构**：
```
/mnt/storage/models-storage/datasets/eval_set/
├── knowledge.jsonl      (知识点评测)
├── reasoning.jsonl       (推理评测)
└── safety.jsonl          (安全评测)
```

**参数配置**：
```
自定义数据集:
  自定义数据集路径: /mnt/storage/models-storage/datasets/eval_set
  列名覆盖(JSON): (留空)
  评测指标:       (留空)
  自定义 Prompt:  (留空)
```

3 个文件分别作为独立数据集评测，结果汇总到 `eval_summary.json`。

### 示例 6：快速验证（限制样本数）

开发调试时限制每个数据集只取少量样本，快速验证流程。

```
模型配置:
  模型路径:   /mnt/storage/models-storage/models/qwen-7b
  模型名称:   qwen-7b-chat
  模型类型:   hf_chat

评测配置:
  评测数据集:  ceval_gen
  每数据集样本数上限: 100   (每数据集只取前100条，快速验证)
  Batch Size: 8
  GPU 数量:   1
```

### 示例 7：数学推理（GSM8K + Few-Shot）

数学评测常用 few-shot 提示提升效果。

```
模型配置:
  模型路径:   /mnt/storage/models-storage/models/qwen-7b
  模型名称:   qwen-7b-chat
  模型类型:   hf_chat

评测配置:
  评测数据集:  gsm8k_gen
  Few-Shot:   4            (4-shot 提示)
  Batch Size: 16
  最大输入长度: 2048       (few-shot 会增加输入长度)
  最大输出长度: 256
```

### 示例 8：高级配置 - 多卡加载大模型

大模型用多卡加载，显存不足时降低精度。

```
模型配置:
  模型路径:   /mnt/storage/models-storage/models/qwen-72b
  模型名称:   qwen-72b
  模型类型:   hf_chat

评测配置:
  评测数据集:  ceval_gen, mmlu_gen
  GPU 数量:   4            (4卡并行)
  Batch Size: 8
  最大输入长度: 4096

高级配置:
  模型加载参数: {"device_map": "auto", "torch_dtype": "float16"}
```

---

## 附：注意事项

1. **互斥校验**：`--custom_dataset_path` 与 `--datasets` 必须二选一，同时填写会报错
2. **目录格式统一**：自定义数据集目录内所有文件必须同格式（不可混用 jsonl 和 csv）
3. **镜像版本**：业务镜像 `main-py310-cu128-r12`；重依赖在 `model-evaluate:base-py310-cu128-r1`，日常改代码只重建业务层
4. **构建方式**：升级依赖跑 `build-base.sh`，改 `src/` 跑 `build.sh`
5. **首次运行**：OpenCompass CLI 冷启动较慢（import torch/transformers），属正常现象
6. **结果对比**：`eval_report.csv` 累积多次评测结果，便于横向对比不同模型/版本
7. **自定义数据集缩写**：单文件用文件名（去扩展名），目录场景用文件名作为数据集标识
