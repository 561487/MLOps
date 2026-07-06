# LLaMA 离线推理（LLaMA Offline Predict）

## 一、功能概述

对微调产出的模型进行批量离线推理。支持两种推理后端：

| 后端 | 参数值 | 适用场景 |
|------|--------|---------|
| HuggingFace Transformers | `transformers` | 小模型（<7B），默认 |
| vLLM | `vllm` | 大模型（>=7B），省显存、吞吐量高 |

## 二、整体架构

```
┌───────────────────────────────────────────────────────────────┐
│  model-offline-predict 任务模板                                 │
│  launcher-rabbitmq.py（编排器 Pod）                             │
│  ├─ 创建 RabbitMQ Pod + Service（消息队列）                     │
│  ├─ 创建 VolcanoJob（N 个 worker Pod）                         │
│  │  ├─ Worker 0（生产者）：读取输入文件 → 逐条发到 RabbitMQ     │
│  │  └─ Worker 1~N-1（消费者）：取消息 → 推理 → 写结果           │
│  └─ 消费完毕 → 清理 RabbitMQ                                    │
└───────────────────────────────────────────────────────────────┘
```

## 三、参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | `/mnt/storage/models-share-volume/output/lora/qwen3-0.6b-lora` | 模型路径 |
| `--input_file` | `/mnt/storage/models-share-volume/data/input.txt` | 输入文件，每行一条待推理文本 |
| `--output_file` | `/mnt/storage/models-share-volume/output/inference/result.jsonl` | 输出文件（JSONL 格式） |
| `--max_new_tokens` | `512` | 最大生成 token 数 |
| `--backend` | `transformers` | 推理后端：`transformers` 或 `vllm` |

## 四、输入输出格式

### 输入文件（input.txt）

```
中国的首都是哪里？
1+1等于几？
什么是机器学习？
```

### 输出文件（result.jsonl）

```jsonl
{"input": "中国的首都是哪里？", "output": "中国的首都是北京。"}
{"input": "1+1等于几？", "output": "1+1等于2。"}
{"input": "什么是机器学习？", "output": "机器学习是人工智能的一个分支..."}
```

## 五、使用流程

### 步骤 1：准备输入数据

在共享存储中创建输入文件，每行一条待推理文本。

### 步骤 2：构建并推送镜像

```bash
cd /home/stu05/mlops/job-template/job/llm_offline_predict
bash build.sh
```

### 步骤 3：导入流水线模板

```bash
docker exec ops_zjm_dev-myapp-1 bash -c "cd /home/myapp && FLASK_APP=myapp flask init"
```

### 步骤 4：创建并运行流水线

从模板创建流水线 → 调整参数 → 运行。

## 六、运行流程详解

```
1. 用户点击"运行"
2. model-offline-predict 节点被调度
3. launcher-rabbitmq.py 启动：
   ├─ 创建 RabbitMQ Pod + Service
   ├─ 创建 VolcanoJob（--num_worker 个 Pod）
   │   ├─ Worker 0: VC_TASK_INDEX=0
   │   │     → datasource() 读取 input.txt
   │   │     → 逐条发到 RabbitMQ
   │   │     → 等待消费完毕
   │   ├─ Worker 1: VC_TASK_INDEX=1
   │   │     → 从 RabbitMQ 取消息
   │   │     → predict(text) 推理
   │   │     → 写 result.jsonl
   │   └─ Worker 2: ...（同上，并行处理）
   ├─ 消费完毕 → 清理 RabbitMQ Pod
   └─ VolcanoJob 标记 Completed
4. 结果文件在 output_file 路径
```

## 七、切换推理后端

### 使用 transformers（默认）

```
--backend transformers
```

### 使用 vLLM

```
--backend vllm
```

vLLM 优势：
- PagedAttention 机制，显存利用率高
- 连续批处理，吞吐量高
- 适合大模型（7B、14B 等）

## 八、注意事项

1. 输入文件必须在共享存储中已存在
2. 模型路径指向微调产出的合并模型目录
3. `--num_worker` 控制并行度，通常设为 3（1 个生产者 + 2 个消费者）
4. 使用 vLLM 后端需确保镜像中已安装 `vllm` 包
