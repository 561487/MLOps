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

## 构建镜像

```bash
cd /path/to/project/root
bash job-template/job/model-evaluate-opencompass/build.sh
```

或手动构建：

```bash
docker build -t 10.121.177.20:8082/model-evaluate-opencompass:latest \
    -f job-template/job/model-evaluate-opencompass/Dockerfile .
```

## 镜像内容

- 基础镜像: `nvidia/cuda:12.1.0-runtime-ubuntu22.04`
- PyTorch 2.3.0 (CUDA 12.1)
- OpenCompass (通过 pip 安装)
- 国内镜像源加速 (清华 apt/pypi + hf-mirror + ModelScope)
