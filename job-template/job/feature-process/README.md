# 特征处理模板

镜像：`10.121.177.20:8082/mlops/feature-process:latest`

统一入口，通过 `--process_type` 参数分发到不同的特征处理逻辑。

## 公共参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `--process_type` | select | ✅ | 处理类型：`calculate_correlation` \| `sample` \| `union_join` \| `drop_duplicates` |

---

## 一、数据合并（union_join）

支持 **Union（行级拼接）** 与 **Join（列级关联）** 两种合并方式，分别对应 SQL 中的 UNION ALL 和 JOIN 操作。

### 参数说明

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:---:|--------|------|
| `--merge_type` | select | ✅ | `union` | 合并类型：`union`=行级拼接，`join`=列级关联 |
| `--input_file_a` | str | ✅ | — | 左表/上表 CSV 文件路径 |
| `--input_file_b` | str | ✅ | — | 右表/下表 CSV 文件路径 |
| `--output_file` | str | ✅ | — | 合并结果输出 CSV 文件路径 |
| `--join_on` | str | 仅join | — | 关联键列名，多列用逗号分隔，如 `id,name` |
| `--join_how` | select | 仅join | `inner` | 关联方式：`inner` / `left` / `right` / `outer` |

### 使用示例

#### Union 行级拼接

将两个结构相同的数据集纵向追加合并：

```bash
python3 launcher.py \
  --process_type union_join \
  --merge_type union \
  --input_file_a /mnt/admin/data/train_set.csv \
  --input_file_b /mnt/admin/data/val_set.csv \
  --output_file /mnt/admin/pipeline/all_data.csv
```

**行为说明**：取两个文件**共有的列**进行拼接，共有的列 dtype 不一致时自动统一。拼接后行数 = A行数 + B行数。

#### Join 列级关联（内连接）

按指定列将两个表的行关联起来，只保留两表都匹配的行：

```bash
python3 launcher.py \
  --process_type union_join \
  --merge_type join \
  --join_on id \
  --join_how inner \
  --input_file_a /mnt/admin/data/user_info.csv \
  --input_file_b /mnt/admin/data/user_label.csv \
  --output_file /mnt/admin/pipeline/user_merged.csv
```

#### Join 左连接

保留左表所有行，右表匹配不上的补空：

```bash
python3 launcher.py \
  --process_type union_join \
  --merge_type join \
  --join_on id \
  --join_how left \
  --input_file_a /mnt/admin/data/user_info.csv \
  --input_file_b /mnt/admin/data/user_label.csv \
  --output_file /mnt/admin/pipeline/user_left_join.csv
```

### Join 方式速查

| 方式 | 说明 |
|------|------|
| `inner` | 取交集，只保留两表都匹配的行 |
| `left` | 保留左表全部行，右表无匹配则填空值 |
| `right` | 保留右表全部行，左表无匹配则填空值 |
| `outer` | 保留两表全部行，无匹配侧填空值 |

---

## 二、特征向量相关性计算（calculate_correlation）

计算两个特征向量之间的余弦相似度。

### 参数说明

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:---:|--------|------|
| `--input_file_a` | str | ✅ | — | 特征向量A文件路径（JSON 数组格式，如 `[0.12, 0.85, ...]`） |
| `--input_file_b` | str | ✅ | — | 特征向量B文件路径（需与A等长） |
| `--output_file` | str | ✅ | — | 相似度结果输出路径（JSON 格式） |

### 使用示例

```bash
python3 launcher.py \
  --process_type calculate_correlation \
  --input_file_a /mnt/admin/data/vector_a.json \
  --input_file_b /mnt/admin/data/vector_b.json \
  --output_file /mnt/admin/pipeline/similarity_result.json
```

输出格式：

```json
{
  "similarity_score": 0.952341,
  "vector_a_length": 128,
  "vector_b_length": 128
}
```

---

## 三、采样（sample）

支持随机采样、分层采样、SMOTE 过采样、欠采样四种方式。

### 参数说明

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:---:|--------|------|
| `--input_file` | str | ✅ | — | 输入 CSV 文件路径 |
| `--output_file` | str | ✅ | — | 采样结果输出 CSV 文件路径 |
| `--sample_type` | select | ✅ | — | `random` / `stratified` / `oversampling` / `undersampling` |
| `--sample_rate` | float | — | `0.5` | 采样比例（0~1）或目标行数（>1） |
| `--stratify_column` | str | — | — | 分层/标签列名 |
| `--random_seed` | int | — | `42` | 随机种子 |
| `--replace` | bool | — | `false` | 是否放回采样（仅 random 有效） |

### 使用示例

```bash
# 随机采样 30%
python3 launcher.py \
  --process_type sample \
  --sample_type random \
  --sample_rate 0.3 \
  --input_file /mnt/admin/data/dataset.csv \
  --output_file /mnt/admin/pipeline/sampled.csv

# 按 label 列分层采样 50%
python3 launcher.py \
  --process_type sample \
  --sample_type stratified \
  --sample_rate 0.5 \
  --stratify_column label \
  --input_file /mnt/admin/data/dataset.csv \
  --output_file /mnt/admin/pipeline/stratified.csv
```

---

## 四、去除重复样本（drop_duplicates）

支持 **精确去重（exact）** 与 **模糊去重（fuzzy）** 两种方式。

### 参数说明

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:---:|--------|------|
| `--input_file` | str | ✅ | — | 输入 CSV 文件路径 |
| `--output_file` | str | ✅ | — | 去重结果输出 CSV 文件路径 |
| `--dedup_method` | select | ✅ | `exact` | 去重方法：`exact`=精确匹配 / `fuzzy`=模糊去重 |
| `--dedup_subset` | str | — | — | 去重依据列名，多列逗号分隔；留空=所有列 |
| `--dedup_keep` | select | — | `first` | 保留策略：`first` / `last` / `false`(全删) |
| `--dedup_threshold` | float | — | `0.95` | 模糊去重相似度阈值(0~1)，仅 fuzzy 时生效 |

### 使用示例

#### 精确去重

删除所有列值完全相同的重复行，保留第一条：

```bash
python3 launcher.py \
  --process_type drop_duplicates \
  --dedup_method exact \
  --dedup_keep first \
  --input_file /mnt/admin/data/dataset.csv \
  --output_file /mnt/admin/pipeline/dedup.csv
```

#### 按指定列去重

只根据 `id` 和 `name` 两列判断重复：

```bash
python3 launcher.py \
  --process_type drop_duplicates \
  --dedup_method exact \
  --dedup_subset id,name \
  --dedup_keep last \
  --input_file /mnt/admin/data/dataset.csv \
  --output_file /mnt/admin/pipeline/dedup_by_id_name.csv
```

#### 模糊去重

数值列特征相似度 >= 0.95 视为重复，阈值越小去重越激进：

```bash
python3 launcher.py \
  --process_type drop_duplicates \
  --dedup_method fuzzy \
  --dedup_threshold 0.95 \
  --dedup_keep first \
  --input_file /mnt/admin/data/dataset.csv \
  --output_file /mnt/admin/pipeline/fuzzy_dedup.csv
```

#### 删除所有重复行（一条不留）

```bash
python3 launcher.py \
  --process_type drop_duplicates \
  --dedup_method exact \
  --dedup_keep false \
  --input_file /mnt/admin/data/dataset.csv \
  --output_file /mnt/admin/pipeline/no_dup.csv
```

### 保留策略说明

| 策略 | 说明 |
|------|------|
| `first` | 保留每组重复行中的第一条 |
| `last` | 保留每组重复行中的最后一条 |
| `false` | 删除所有重复行（只要重复就全部丢弃） |