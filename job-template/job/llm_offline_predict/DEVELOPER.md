# 开发文档

## 一、文件结构

```
job-template/job/llm_offline_predict/
├── predict.py           # 推理逻辑（核心代码）
├── predict_model.py     # 离线推理框架（从 model_offline_predict 复制）
├── py_rabbit.py         # RabbitMQ 客户端（从 model_offline_predict 复制）
├── Dockerfile           # 镜像构建
├── build.sh             # 构建推送脚本
├── README.md            # 技术文档
└── DEVELOPER.md         # 开发文档（本文件）
```

## 二、依赖关系

```
predict.py
  ├── Offline_Predict（predict_model.py 中的基类）
  │     ├── datasource()  ← 开发者实现：定义数据来源
  │     ├── predict()     ← 开发者实现：定义推理逻辑
  │     ├── producter()   ← 框架实现：发送数据到 RabbitMQ
  │     ├── consumer()    ← 框架实现：从 RabbitMQ 消费数据
  │     └── run()         ← 框架实现：自动判断角色（生产者/消费者）
  ├── transformers / vLLM  ← 推理后端
  └── argparse            ← 参数解析
```

## 三、关键代码说明

### 3.1 推理逻辑（predict.py）

**Python 文件：** `/home/stu05/mlops/job-template/job/llm_offline_predict/predict.py`

**类继承结构：**

```python
class LLaMA_Offline_Predict(Offline_Predict):  # 继承框架基类
```

**初始化流程：**

```
__init__()
  ├─ 解析命令行参数（model_path, input_file, output_file, max_new_tokens, backend）
  ├─ 根据 backend 选择初始化方式
  │   ├─ "transformers" → _init_transformers()
  │   │     ├─ AutoTokenizer.from_pretrained()
  │   │     └─ AutoModelForCausalLM.from_pretrained()
  │   └─ "vllm" → _init_vllm()
  │         ├─ LLM(model_path)
  │         └─ SamplingParams()
  └─ 准备输出目录
```

**必须实现的两个方法：**

```python
def datasource(self):
    """生产者调用：读取输入文件，返回 List[str]"""
    return open(input_file).readlines()

def predict(self, text):
    """消费者调用：对单条文本推理，返回 str"""
    # transformers 方式
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=N)
    result = tokenizer.decode(outputs[0])
    
    # 写结果到文件
    f.write(json.dumps({"input": text, "output": result}))
    return result
```

**run() 方法的调度逻辑（框架实现，predict.py 无需修改）：**

```python
def start_predict(self, local_rank):
    VC_TASK_INDEX = os.environ.get('VC_TASK_INDEX', None)
    if not VC_TASK_INDEX:
        # 单节点模式：直接顺序处理
        for item in self.datasource():
            self.predict(item)
    else:
        # 分布式模式：
        #   VC_TASK_INDEX=0 + local_rank=0 → 生产者（producter）
        #   其他 → 消费者（consumer）
        self.create_queue()
        VC_TASK_INDEX = int(VC_TASK_INDEX)
        if VC_TASK_INDEX == 0 and local_rank == 0:
            self.producter()     # 调 datasource() 发消息
            self.wait_finish()   # 等消费完毕
        else:
            self.consumer()      # 调 predict() 处理消息
```

### 3.2 任务模板配置（init-pipeline.json）

**文件位置：** `/home/stu05/mlops/myapp/init/init-pipeline.json`

**模板名称：** `llm-finetune-eval-full`

**新增的离线推理节点配置：**

```json
// dag_json 中的节点依赖
"llm-offline-predict": {
    "upstream": ["deploy-service"]
}

// expand 中的节点位置和连线
{
    "id": "llm-offline-predict",
    "type": "dataSet",
    "position": { "x": 410, "y": 700 },
    "data": {
        "info": { "describe": "离线推理" },
        "name": "llm-offline-predict",
        "label": "离线推理"
    }
}

// tasks 中的任务定义
{
    "job_templete": "model-offline-predict",   // 复用现有的分布式离线推理模板
    "name": "llm-offline-predict",
    "label": "离线推理",
    "resource_memory": "10G",
    "resource_cpu": "4",
    "resource_gpu": "1",
    "node_selector": "gpu=true,train=true;mps=false",
    "args": {
        "--image": "10.121.177.20:8082/mlops/llm-offline-predict:20260630",
        "--command": "python3 /app/predict.py --model_path ... --input_file ... --backend transformers",
        "--num_worker": "3"
    }
}
```

## 四、关键修改记录

### 4.1 新增文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `job-template/job/llm_offline_predict/predict.py` | LLaMA 离线推理核心代码 |
| 新建 | `job-template/job/llm_offline_predict/Dockerfile` | 基于 `llama_factory:20250712` |
| 新建 | `job-template/job/llm_offline_predict/build.sh` | 构建并推送镜像到仓库 |
| 复制 | `job-template/job/llm_offline_predict/predict_model.py` | 从 `model_offline_predict/` 复制 |
| 复制 | `job-template/job/llm_offline_predict/py_rabbit.py` | 从 `model_offline_predict/` 复制 |

### 4.2 修改文件

| 修改 | 文件 | 说明 |
|------|------|------|
| 新增任务节点 | `myapp/init/init-pipeline.json` | 在 `llm-finetune-eval-full` 模板中新增 `llm-offline-predict` 节点 |

### 4.3 init-pipeline.json 具体修改

在 `llm-finetune-eval-full` 模板中：

1. **dag_json** — 新增 `llm-offline-predict` 节点，上游依赖 `deploy-service`
2. **expand** — 新增节点布局和连线，位置在 `deploy-service` 下方（y=700），增加 `deploy-service → llm-offline-predict` 连线
3. **tasks** — 新增任务，job_template 使用已存在的 `model-offline-predict`（id=66）

## 五、实现思路（本项目的实际工作）

### 5.1 做了什么

本项目的工作是：**在已有 `model-offline-predict` 分布式框架的基础上，实现 LLaMA 模型的离线推理逻辑**。不改动框架本身。

核心产出只有一个文件：`predict.py`。

### 5.2 predict.py 的实现思路

```
用户传参 → 加载模型 → 读取数据 → 逐条推理 → 写结果
```

具体来说：

```python
# ── 第一步：参数化 ──
# 用 argparse 定义 5 个参数，让用户在流水线表单里就能配置
# 模型路径、数据路径、输出路径、最大生成长度、推理后端
parser.add_argument('--model_path')       # 模型在哪？
parser.add_argument('--input_file')       # 数据在哪？
parser.add_argument('--output_file')      # 结果写哪？
parser.add_argument('--max_new_tokens')   # 生成多长？
parser.add_argument('--backend')          # 用什么推理引擎？

# ── 第二步：适配框架接口 ──
# Offline_Predict 要求子类实现两个方法：
#   datasource() → 返回 List[str]，每条一条数据
#   predict(str) → 对单条数据推理，返回结果字符串

def datasource(self):
    # 实现：打开输入文件，按行读取，去掉空行
    return [line.strip() for line in open(input_file) if line.strip()]

def predict(self, text):
    # 实现：tokenizer编码 → model.generate → tokenizer解码 → 写文件
    inputs = self.tokenizer(text, return_tensors="pt")
    outputs = self.model.generate(**inputs, max_new_tokens=N)
    result = self.tokenizer.decode(outputs[0])
    save_to_file(result)    # 追加写 JSONL
    return result

# ── 第三步：两种推理后端可选 ──
# transformers：直接 from_pretrained，零额外依赖
# vLLM：LLM(model_path) 初始化，适合大模型
# 通过 --backend 参数切换，设计上可扩展更多后端
```

### 5.3 参数传递的"曲线救国"思路

`model-offline-predict` 模板的编排器（`launcher-rabbitmq.py`）只接受 4 个固定参数：

```
--image       worker 用的镜像
--command     worker 启动命令
--num_worker  并发数
--working_dir 工作目录
```

但我们的推理需要传入更多参数（model_path、input_file 等）。

**解决办法：** 把推理参数拼到 `--command` 字符串里，"骗过"编排器，让它原样传给 worker。

```
编排器收到的 --command:
  "python3 /app/predict.py --model_path /mnt/... --input_file /mnt/... --backend transformers"

编排器把这个命令原样传给每个 worker Pod。
worker Pod 启动后，predict.py 自己用 argparse 解析这些参数。
```

### 5.4 Dockerfile 的思路

基于已有的 `llama_factory:20250712` 镜像（已有 torch、transformers），只需额外安装 3 个包（pika、psutil、pysnooper），再把 3 个 Python 文件复制进去。

不重新安装 torch/transformers，构建速度快。


## 六、扩展指南

### 添加新的推理后端

在 `predict.py` 的 `LLaMA_Offline_Predict` 类中：

1. 在 `__init__` 的 `--backend` 参数中新增 `choices`
2. 添加 `_init_xxx()` 方法初始化模型
3. 添加 `_predict_xxx()` 方法实现推理
4. 在 `predict()` 方法中添加分支调用

示例：

```python
# 新增 backend
parser.add_argument('--backend', default='transformers',
                    choices=['transformers', 'vllm', 'tgi'])

# 初始化
def _init_tgi(self):
    ...

# 推理
def _predict_tgi(self, text):
    ...

# 分发
def predict(self, text):
    if self.args.backend == 'vllm':
        result = self._predict_vllm(text)
    elif self.args.backend == 'tgi':
        result = self._predict_tgi(text)
    else:
        result = self._predict_transformers(text)
```
