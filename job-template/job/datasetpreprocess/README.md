# DatasetPreprocess

DatasetPreprocess 从 PVC 读取 DatasetDown 下载的原始数据集，完成通用清洗与样本选择，输出一个完整、未切分且不绑定训练框架的 JSONL 数据集。

## 组件边界

- 负责：文件发现、格式读取、字段选择/删除/重命名、条件过滤、必填校验、文本清洗、长度过滤、去重、抽样和异常记录。
- 不负责：QA/Instruction/Messages 格式转换、system prompt 构造、训练集/验证集/评测集切分。

## 支持的输入

- JSON、JSONL
- CSV、TSV
- Parquet
- TXT（每个非空行作为一条记录）

## 输出

```text
output_dir/
├── dataset.jsonl
├── rejected.jsonl
└── preprocessing_report.json
```

`dataset.jsonl` 保留清洗后的原始业务字段，后续由 DatasetConvert 组件按 SFT 或评测用途进行格式转换和切分。

## 示例

```bash
python3 launcher.py \
  --input_path /mnt/admin/raw/fault.csv \
  --input_format csv \
  --required_fields alarm_text,solution \
  --text_fields alarm_text,solution \
  --filter_expression "fault_type is not null" \
  --rename_fields "device_name:device,alarm_text:alarm" \
  --dedup_fields device,alarm \
  --output_dir /mnt/admin/dataset-preprocess/fault-v1
```

节点不使用 `eval` 执行过滤表达式。结果先写入同级临时目录，全部成功后再原子替换目标目录，避免留下半成品。
