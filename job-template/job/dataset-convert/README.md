# Dataset Convert

Dataset Convert 是"模型数据格式适配组件"，将用户已有数据集读取、识别并转换为平台后续训练 / 评测可直接使用的标准 JSONL 格式。

```text
各种外部格式
    ↓
text / alpaca / messages / sharegpt / qa / prompt_response / custom
    ↓
Dataset Convert
    ↓
messages / text / eval_qa
    ↓
训练 / 评测
```

本组件只负责：识别文件格式 → 识别数据 Schema → Schema 转换 → 格式合法性校验 → 输出标准 JSONL。

本组件**不负责**（由未来 Dataset Split 等承担）：train/val/test 划分、shuffle、sampling、deduplicate、长度过滤、条件过滤、数据清洗、质量评分。

## 用法

```bash
python3 launcher.py \
  --input_path /mnt/admin/raw/fault.jsonl \
  --output_dir /mnt/admin/dataset-convert/fault-v1 \
  --input_format auto \
  --source_schema auto \
  --target_schema messages
```

## 参数

### 基础参数

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `--input_path` | 必填 | 输入数据集文件或目录（支持递归） |
| `--output_dir` | 必填 | 输出目录 |
| `--input_format` | `auto` | `auto`/`json`/`jsonl`/`csv`/`tsv`/`parquet`/`txt` |
| `--source_schema` | `auto` | `auto`/`text`/`alpaca`/`messages`/`sharegpt`/`qa`/`prompt_response`/`custom` |
| `--target_schema` | `messages` | `messages`/`text`/`eval_qa`（用户显式选择，不自动猜测） |

### 高级参数

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `--field_mapping` | `""` | `source_schema=custom` 必填。JSON 对象，如 `{"query_text":"user","reply_text":"assistant"}` |
| `--system_prompt` | `""` | 仅 `target_schema=messages` 生效。原 messages 已存在 system 轮次时不重复注入 |
| `--keep_metadata` | `false` | true 时多余字段统一收进 `metadata` 对象（不平铺） |
| `--invalid_policy` | `reject` | `reject` 写 rejected.jsonl；`skip` 仅计数；`fail` 遇首条无效记录即失败 |
| `--recursive` | `true` | 目录输入时递归扫描 |
| `--file_pattern` | `*` | 目录输入时文件名匹配，如 `*.csv` |
| `--encoding` | `utf-8` | 文本类文件编码 |
| `--overwrite` | `false` | 允许覆盖已有输出目录 |

## 支持的转换矩阵（V1）

| Source \ Target | messages | text | eval_qa |
|-----------------|----------|------|---------|
| messages        | ✅ 规范化 | ❌ | ❌ 暂不支持 |
| sharegpt        | ✅ | ❌ | ❌ 暂不支持 |
| alpaca          | ✅ | ❌ | ✅ |
| qa              | ✅ | ❌ | ✅ |
| prompt_response | ✅ | ❌ | ✅ |
| text            | ❌（明确报错） | ✅ 透传 | ❌ |
| custom          | ✅ | ✅ | ✅ |

`text → messages` 默认禁止，报错提示：

> 纯文本数据无法自动确定 user / assistant 角色，请使用 target_schema=text，或通过 source_schema=custom + field_mapping 明确指定字段语义。

## 规则要点

- **alpaca**：`input` 非空时 `user = instruction + "\n" + input`；`output` 缺失或为空按 invalid_policy 处理（reason=missing_output）。
- **sharegpt**：`human→user`、`gpt→assistant`、`user→user`、`assistant→assistant`、`system→system`；未知 role（function/tool/developer 等）**拒绝**（reason=invalid_role），不允许静默丢弃。同时兼容 `{"role":"user","content":...}` 写法。
- **messages 校验**：必须是 list；每项 dict；必须存在 role/content；role 在 system/user/assistant 集合；content 非空；至少一条 user 和一条 assistant 轮次。
- **qa**：优先 `question/answer`，其次 `query/response`（严格配对，不混搭）。
- **custom**：mapping 目标字段必须符合当前 target_schema（messages→user/assistant；text→text；eval_qa→input/target）；不允许未知目标字段；JSON 解析失败任务直接失败。
- **system_prompt**：已有 system → 保留原 system 不额外插入；没有 system → 在最前面插入。

## 输出契约

```text
output_dir/
├── converted.jsonl
├── dataset_manifest.json
└── rejected.jsonl   ← 仅当存在被拒绝记录时创建；无拒绝记录时不创建
```

`dataset_manifest.json` 中的 `input_format` / `source_schema` 写入**实际识别结果**（auto 时写探测出的真实值）。

## 多模态（V1 边界）

> 当前 V1 仅支持文本数据转换。多模态将在确认当前平台 LLaMA-Factory 与 ms-swift 实际数据契约后扩展。

- 输入 messages 中 `content` 为 list/dict（多模态内容数组），或记录顶层出现 `image/images/image_path/video/videos/video_path/audio/audios/audio_path` 字段，一律 **reject**，reason=**multimodal_schema_not_supported_v1**，不允许悄悄丢弃。
- 代码中已预留扩展位置（`schema.py::check_multimodal_record` / `converter.py` 分派处），未来 V1.1 在此接入媒体校验与路径规范化。

## 异常处理

- `reject`（默认）：拒绝记录写入 rejected.jsonl，格式：`{"source_file": ..., "line": ..., "reason": ..., "record": {...}}`
- `skip`：不写 rejected，仅计入统计。
- `fail`：遇到第一条无效记录即失败退出。
- **有效样本数为 0 时，无论 invalid_policy 是什么，任务一律非 0 退出**（不允许"任务成功但输出 0 条有效数据"）。

## 错误码

| 码 | 含义 |
|----|------|
| 0 | 成功 |
| 1 | 通用错误 |
| 2 | 输入路径不存在 / 无支持的文件 |
| 3 | 源结构自动识别失败 |
| 4 | 不支持的转换组合 |
| 5 | field_mapping 非法 |
| 6 | 输出目录冲突（已存在且未开 overwrite） |
| 7 | 有效样本数为 0 |
| 8 | invalid_policy=fail 触发 |

## 已知限制（V1）

- JSON 输入使用 `json.load` 整体载入内存，超大 JSON 有内存风险（建议转 JSONL）。
- 不支持远程 URL、HuggingFace `save_to_disk`、ModelScope/HF Dataset ID。
- 输出固定 JSONL，无 JSON/CSV/Parquet 输出选项。
- `keep_metadata=true` 时输出的 `metadata` 列是否被 LLaMA-Factory / ms-swift 兼容需下游最小加载测试确认；若发现兼容问题，V1 建议固定 `keep_metadata=false`。

## 测试

```bash
python3 tests/run_tests.py
```

覆盖 T1-T19（含多文件目录、Parquet、多模态拒绝、UTF-8 中文、Job FAIL 等）。
