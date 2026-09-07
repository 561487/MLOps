# DatasetMergeSplit

DatasetMergeSplit 从 PVC 读取一个或多个 DatasetConvert 输出目录，按照 DatasetConvert 的 `dataset_manifest.json` 定位输出文件、识别目标格式和版本，合并相同格式的 SFT JSONL，并按比例输出训练集、验证集和测试集。

## 支持格式

- `messages`
- `qa`

同一次任务的所有输入必须具有相同的 `format_type` 和 `schema_version`。组件不转换格式；测试集是按比例保留的离线评测数据，不参与训练。

## 输入目录

```text
converted-dataset/
├── converted.jsonl
└── dataset_manifest.json
```

组件优先读取 DatasetConvert Manifest 字段：

- `output_file`
- `target_schema`
- `version`
- `valid_samples`

并兼容旧字段 `data_file`、`format_type`、`schema_version`、`sample_count`。

## 输出

```text
output_dir/
├── train.jsonl
├── validation.jsonl
├── test.jsonl
├── rejected.jsonl
├── conflicts.jsonl
├── merge_split_report.json
└── dataset_manifest.json
```

当 `validation_ratio=0` 或 `test_ratio=0` 时，不生成对应的 `validation.jsonl` 或 `test.jsonl`。

## 示例

```bash
python3 launcher.py \
  --dataset_paths "/mnt/converted/warning;/mnt/converted/diagnosis;/mnt/converted/maintenance" \
  --format_type auto \
  --train_ratio 0.9 \
  --validation_ratio 0.1 \
  --test_ratio 0 \
  --shuffle_before_split true \
  --seed 42 \
  --output_dir /mnt/merged/industrial-sft-v1
```

`dataset_paths` 使用英文分号 `;` 分隔路径，至少填写一个目录，数量不设上限。只配置一个输入目录时，组件直接对该数据集执行校验、去重和训练/验证比例分割。

运行期间会实时输出结构化进度日志，包括输入识别、数据集开始、每处理 1000 条的进度、索引完成、比例分割、输出写入和最终完成状态。

三路分割时可配置：

```text
train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1
```
