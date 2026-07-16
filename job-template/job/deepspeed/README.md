# DeepSpeed 分布式训练模板

该模板提供封装式 DeepSpeed 分布式训练节点。平台 launcher 创建 PyTorchJob，并在 Master/Worker Pod template 中设置 `schedulerName=volcano`；训练 Pod 内执行 `/app/deepspeed_entrypoint.sh`，自动生成 `/tmp/ds_config.json` 后使用 `torchrun` 启动多机多卡训练，Hugging Face Trainer 根据该配置启用 DeepSpeed。

## 训练模式

- `finetune`：使用 HuggingFace causal language model 做继续训练或指令微调。
- `pretrain`：使用 HuggingFace causal language modeling 做基础继续预训练。
- `custom`：执行用户自定义命令，平台仍负责 PyTorchJob、Volcano、DeepSpeed 配置和多机启动。

## 常用参数

```bash
python3 launcher.py \
  --train_mode finetune \
  --image 10.121.177.20:8082/mlops/deepspeed:20260716-cuda121-ds0144-hf4442-r3 \
  --num_worker 2 \
  --num_gpus 1 \
  --model_name_or_path /mnt/admin/models/qwen \
  --dataset_path /mnt/admin/data/train.jsonl \
  --output_dir /mnt/admin/output/qwen \
  --zero_stage 2 \
  --precision fp16
```

自定义模式示例：

```bash
python3 launcher.py \
  --train_mode custom \
  --working_dir /workspace \
  --command "python train.py --epochs 1" \
  --append_config_to_command true \
  --config_arg_name deepspeed
```

## 镜像要求

默认构建出的 `10.121.177.20:8082/mlops/deepspeed:20260716-cuda121-ds0144-hf4442-r3` 镜像同时可作为 launcher 镜像和 worker 镜像。worker 镜像至少需要包含：

- Python、PyTorch、DeepSpeed
- Transformers、Datasets、Accelerate、PEFT
- `/app/deepspeed_entrypoint.sh`
- `/app/train_runner.py`
- `/app/ds_config.py`

如果替换为自定义 worker 镜像，需要确保这些入口文件存在，或通过 `--entrypoint` 指定镜像内等价脚本。

`--master_port` 会注册为 PyTorchJob 的 `pytorchjob-port`，应设置为 1 到 65535 之间的可用端口。`--num_worker` 和 `--num_gpus` 必须大于 0。

使用 NVMe offload 时，需要通过任务挂载配置将节点本地 NVMe 目录挂载到所有训练 Pod，并用 `--nvme_path` 指向该目录；目录不存在或不可写时任务会在训练启动前直接报错。参数 offload 仅支持 ZeRO-3，优化器 offload 需要启用 ZeRO stage 1、2 或 3。

## 本地验证

```bash
python3 -m unittest job-template/job/deepspeed/tests/test_ds_config.py -v
python3 -m unittest job-template/job/deepspeed/tests/test_train_runner.py -v
bash -n job-template/job/deepspeed/deepspeed_entrypoint.sh
python3 job-template/job/deepspeed/launcher.py --help
python3 -m json.tool myapp/init/init-job-template.json >/tmp/init-job-template.json
```
