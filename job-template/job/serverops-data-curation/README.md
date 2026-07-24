# ServerOps 数据集制备节点

该节点从 ModelScope 下载公开网络安全问答数据（已存在时跳过），在挂载
`models-storage` PVC 的任务 Pod 内流式筛选服务器运维数据，并输出
MS-SWIFT 可直接用于 Qwen3 SFT 的 JSONL。

## 输入与输出

默认原始目录：

```text
/mnt/storage/models-storage/dataset/cybersecurity-raw
```

默认输出目录：

```text
/mnt/storage/models-storage/dataset/cybersecurity-serverops-v1
```

原始目录只读使用，不会删除或覆盖。输出目录存在 `_SUCCESS` 时默认跳过，
防止误覆盖已经完成的数据版本。

## 输出结构

```text
cybersecurity-serverops-v1/
├── sft/
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
├── metadata/
│   └── records.jsonl
├── reports/
│   ├── statistics.json
│   └── rejected_samples.jsonl
├── README.md
└── _SUCCESS
```

SFT 文件采用 MS-SWIFT 标准格式：

```json
{
  "id": "sha256...",
  "category": "kubernetes",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "<think>\n\n</think>\n\n..."}
  ]
}
```

Qwen3 训练时应同时配置：

```text
--loss_scale ignore_empty_think
```

## 推荐资源

```text
CPU: 8-16
Memory: 32-64Gi
GPU: 0
Volume: models-storage(storage):/mnt/storage/models-storage
```

## 冒烟测试

```bash
python3 launcher.py \
  --download_if_missing false \
  --raw_dir /mnt/storage/models-storage/dataset/cybersecurity-raw \
  --output_dir /mnt/storage/models-storage/dataset/cybersecurity-serverops-smoke \
  --target_size 1000 \
  --max_source_records 5000
```

## 正式运行

```bash
python3 launcher.py \
  --dataset_id hcnote/Cybersecurity-High-Quality-Dataset \
  --raw_dir /mnt/storage/models-storage/dataset/cybersecurity-raw \
  --output_dir /mnt/storage/models-storage/dataset/cybersecurity-serverops-v1 \
  --download_if_missing true \
  --target_size 60000 \
  --min_domain_score 3 \
  --min_quality_score 5 \
  --seed 42
```

若 ModelScope 下载需要认证，应通过任务环境变量/Secret 注入令牌，禁止将令牌
写入节点参数、镜像或日志。
