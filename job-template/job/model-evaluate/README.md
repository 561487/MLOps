# Model Evaluate - OpenCompass

基于 [OpenCompass](https://github.com/open-compass/opencompass) 的 LLM 模型评测任务模板，用于 Cube Studio Pipeline。

## 用法

在 Cube Studio 平台创建 Pipeline 节点时，选择此模板并配置以下参数：

### 模型参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--model_path` | string | 是 | - | HuggingFace 模型本地路径或目录 |
| `--model_name` | string | 否 | `model` | 模型名称 |
| `--model_version` | string | 否 | `v1` | 模型版本号 |
| `--model_type` | string | 否 | `hf_chat` | 模型类型: `hf_chat`(对话模型) / `hf_base`(基座模型) |
| `--model_kwargs` | string | 否 | `""` | 模型加载额外参数 (JSON), 如 `{"device_map":"auto"}` |

### 评测参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--datasets` | string | 是 | - | 评测数据集, 逗号分隔, 如 `ceval_gen,gsm8k_gen` |
| `--output_path` | string | 是 | - | 评测结果输出根目录 |

### 推理参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--num_gpus` | int | 否 | `1` | 使用的 GPU 数量 |
| `--batch_size` | int | 否 | `64` | 推理 batch size |
| `--max_seq_len` | int | 否 | `2048` | 模型最大输入长度 |
| `--max_out_len` | int | 否 | `512` | 模型最大输出长度 |
| `--few_shot` | int | 否 | `0` | Few-shot 样本数 |

## 输出文件

| 文件 | 说明 |
|------|------|
| `metric.json` | 平台自动采集的指标文件 |
| `eval_summary.json` | 结构化评测摘要 |
| `eval_report.csv` | 评测结果表格 |
| `eval_run.log` | OpenCompass 运行日志 |
| `opencompass_results/` | OpenCompass 原始输出目录 |

## 使用自定义数据集

除内置数据集(`--datasets`)外，可使用本地 jsonl/csv/json 文件评测自己的数据集。
**两种模式互斥**：填了"自定义数据集路径"就忽略"评测数据集"，二者都填会报错退出。

### 路径

- 路径是**容器内绝对路径**，复用节点既有挂载卷 `models-storage`，路径形如
  `/mnt/storage/models-storage/datasets/my_eval.jsonl`
- 支持**单文件**或**单目录**：
  - 单文件：一个数据集，abbr 用文件名(去扩展名)
  - 目录：目录下**每个文件作为独立数据集**，abbr 用各自文件名；目录内格式必须统一(.json/.jsonl 或 .csv)，混放会报错
- 要评多个自定义数据集 → 放同一目录即可，一次跑完

### 自动推断(核心特性)

只需填 `--custom_dataset_path`，模板会**读取数据首行自动推断**列名/题型/指标，无需手动配置：

| 推断项 | 规则 |
|--------|------|
| 格式 | 按扩展名：`.json`/`.jsonl` → json loader；`.csv` → csv loader |
| 答案列 | 优先 `target` > `answer` > `label` > `gold` > `gt`，否则取最后一列 |
| 选项列(选择题) | 字段含 `A,B,C,D`(≥2个大写单字母) 或 `option_*`/`option1..n`(≥2个) → 选择题；否则生成式 |
| 问题列 | 优先 `input` > `question` > `query` > `prompt`，否则取剩余第一个字段 |
| 指标 | 选择题 → `accuracy`；生成式 → `exact_match` |

推断结果会打印到日志(`[INFO] 自动推断: ...`)，可直观确认是否正确。

### 自定义数据集参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--custom_dataset_path` | string | 是* | - | 数据文件/目录路径。*与 `--datasets` 二选一 |
| `--custom_columns` | string | 否 | `""` | 列名覆盖(JSON)，推断不准时救场 |
| `--custom_metric` | string | 否 | `""` | 指标覆盖；空=自动(选择题→accuracy / 生成式→exact_match) |
| `--custom_prompt_template` | string | 否 | `""` | 自定义 prompt；空=自动模板 |

### 数据格式样例

#### 选择题 — 自动识别(有 A/B/C/D 列)

`mcq.jsonl` (每行一个 JSON)：
```json
{"question":"1+1=?","A":"1","B":"2","C":"3","D":"4","answer":"B"}
{"question":"中国的首都是？","A":"上海","B":"北京","C":"广州","D":"深圳","answer":"B"}
```
→ 自动推断：问题列=`question`，选项=`A,B,C,D`，答案列=`answer`，指标=`accuracy`

默认选择题 prompt(自动按列名生成)：
```
Answer the following multiple choice question. The last line of your response
should be of the following format: 'ANSWER: $LETTER' (without quotes) where
LETTER is one of A-D. Think step by step before answering.

{question}

A) {A}
B) {B}
C) {C}
D) {D}
```

#### 生成式问答 — 自动识别(无选项列)

`gen.jsonl`：
```json
{"input":"请把这句话翻译成英文：今天天气很好","target":"The weather is nice today."}
```
→ 自动推断：问题列=`input`，答案列=`target`，无选项，指标=`exact_match`

默认 prompt：`{input}`(直接把问题喂给模型)。翻译/摘要类任务建议 `--custom_metric=bleu` 或 `rouge`。

### 列名不规范时：用 `--custom_columns` 救场

若数据字段名不在上述约定里(如 `q`/`ans`/`opt1`)，自动推断会不准，用 JSON 一次性覆盖：

```
--custom_columns='{"input":"q","target":"ans","choices":"opt1,opt2,opt3,opt4"}'
```

| 字段 | 含义 |
|------|------|
| `input` | 问题列名 |
| `target` | 答案列名 |
| `choices` | 选项列名(逗号分隔，非空=选择题) |

### CSV 注意事项

CSV 第一行必须是 header，列名按上述约定命名(如 `question,A,B,C,D,answer`)：

```csv
question,A,B,C,D,answer
"1+1=?","1","2","3","4","B"
"中国首都?","上海","北京","广州","深圳","B"
```

### Cube Studio 节点示例

最简用法(只填路径，其余自动)：

| 参数 | 值 |
|------|----|
| 模型路径 | `/mnt/storage/models-storage/models/Qwen2.5-0.5B-Instruct` |
| 输出路径 | `/mnt/storage/models-storage/output/opencompass` |
| 自定义数据集路径 | `/mnt/storage/models-storage/datasets/mcq.jsonl` |
| 每数据集样本数上限 | `10` (冒烟测试用) |

评测完成后 `metric.json` 里会出现以文件名命名的数据集(如 `mcq`)及其指标。目录场景则每个文件各一个数据集。

## 构建镜像

镜像分为两层，日常改业务代码只需重建业务层（快）：

```bash
cd /path/to/project/root

# 首次 / 升级 PyTorch、OpenCompass、transformers 等重依赖时（慢，偶尔执行）
bash job-template/job/model-evaluate/build-base.sh

# 日常改 src/ 业务逻辑后（快，几十秒）
bash job-template/job/model-evaluate/build.sh
```

| 镜像 | Tag 示例 | 内容 |
|------|----------|------|
| `model-evaluate` | `base-py310-cu128` | CUDA、PyTorch、OpenCompass、transformers 补丁 |
| `model-evaluate` | `main-py310-cu128-r10` | base + `/app` 入口脚本 |

## 镜像内容

- BASE: `nvidia/cuda:12.8.0-runtime-ubuntu22.04` + PyTorch 2.7.0+cu128
- OpenCompass main + transformers ≥5.2 + encode_plus 兼容补丁
- 国内镜像源加速 (清华 apt/pypi + hf-mirror + ModelScope)
