paddle 在 k8s 之上的分布式训练

使用该模板能够帮助你在 k8s 自动创建一个 PaddleJob 分布式训练集群。前提是训练代码已经按 PaddlePaddle 官方分布式方案编写，例如使用 `paddle.distributed` / Fleet，并能从 PaddleJob 注入的分布式环境变量中完成初始化。

# 分布式说明

该模板最终创建的 CRD 为：

```yaml
apiVersion: kubeflow.org/v1
kind: PaddleJob
spec:
  paddleReplicaSpecs:
    Master:
      replicas: 1
    Worker:
      replicas: num_worker - 1
```

平台参数 `--num_worker` 表示总节点数，包含 1 个 Master。比如 `--num_worker=3` 时，会创建 1 个 Master 和 2 个 Worker。

# 参数

```bash
{
    "分布式训练": {
        "--image": {
            "type": "str",
            "label": "镜像",
            "default": "paddlepaddle/paddle:latest-gpu-cuda11.8-cudnn8.6-trt8.5",
            "describe": "worker镜像，直接运行你代码的环境镜像"
        },
        "--working_dir": {
            "type": "str",
            "label": "启动目录",
            "default": "/mnt/{{creator}}/pipeline/example/paddle",
            "describe": "命令的启动目录"
        },
        "--command": {
            "type": "str",
            "label": "启动命令",
            "default": "bash start.sh",
            "describe": "启动命令，例如 python3 train.py"
        },
        "--num_worker": {
            "type": "str",
            "label": "worker数量",
            "default": "3",
            "describe": "分布式训练总节点数，包含1个master"
        }
    }
}
```

# 环境变量

平台会为每个 Pod 注入资源和任务环境变量；PaddleJob Operator 会为 Paddle 分布式训练注入框架所需的角色、rank、endpoint 等信息。训练脚本中通常通过 PaddlePaddle 分布式接口读取这些信息完成初始化。

常见使用方式：

```python
import paddle
import paddle.distributed as dist

if dist.get_world_size() > 1:
    dist.init_parallel_env()

model = ...
if dist.get_world_size() > 1:
    model = paddle.DataParallel(model)
```

# 启动方式

### 直接执行训练脚本

```bash
python3 train.py
```

### 通过启动脚本封装

推荐将环境准备、依赖安装、数据检查和训练命令写入 `start.sh`：

```bash
#!/bin/bash
set -ex

python3 train.py --config configs/train.yaml
```

# 注意事项

- 用户镜像需要提前安装好 PaddlePaddle、CUDA/CUDNN 等训练依赖。
- 训练代码必须按 Paddle 分布式方式编写，模板只负责创建 PaddleJob，不会自动把单机代码改造成分布式代码。
- GPU 资源由平台任务资源配置控制，模板会将 `KFJ_TASK_RESOURCE_GPU` 转换为容器资源请求。
- 日志由 launcher 使用 `stern` 跟踪，PaddleJob 结束后根据状态返回成功或失败。

