# DatasetProcess

DatasetProcess 从 PVC 读取 DatasetDown 下载的原始数据集，完成字段选择、条件过滤、抽样、基础清洗和 SFT 格式转换，输出可直接交给 ms-swift 的 messages JSONL。

## 支持格式

- JSON、JSONL
- CSV、TSV
- Parquet
- TXT（每个非空行作为一条记录）

## 转换模式

- `qa`：自动识别 question/query/prompt 与 answer/response/output。
- `instruction`：组合 instruction 与 input，使用 output 作为回答。
- `conversation`：规范化 messages/conversations 中的 role/content。
- `custom`：通过字段列表或 `{{field}}` 模板构造 user 和 assistant。

## 输出

```text
output_dir/
├── train.jsonl
├── val.jsonl
├── rejected.jsonl
└── processing_report.json
```

ms-swift 对接：

```text
--dataset=/mnt/.../dataset-process/train.jsonl
--val_dataset=/mnt/.../dataset-process/val.jsonl
```

## 示例

```bash
python3 launcher.py \
  --input_path /mnt/admin/raw/fault.csv \
  --input_format csv \
  --required_fields alarm_text,solution \
  --filter_expression "fault_type is not null" \
  --template_type custom \
  --system_prompt "你是工业设备故障预警助手。" \
  --user_template "设备：{{device_name}}\n告警：{{alarm_text}}" \
  --assistant_template "故障：{{fault_type}}\n建议：{{solution}}" \
  --train_ratio 0.9 \
  --val_ratio 0.1 \
  --output_dir /mnt/admin/dataset-process/fault-v1
```

节点只执行安全的字段条件比较，不使用 `eval` 执行过滤表达式。正式结果先写入临时目录，成功后再替换目标目录，避免产生半成品。
