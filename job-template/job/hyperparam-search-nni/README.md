# hyperparam-search-nni 算子

NNI 兼容格式的超参数搜索算子，用于 GBDT 模型的超参数优化。

**当前版本**：Pipeline 友好的单进程实现。支持 NNI search_space JSON 格式，内置 Random / GridSearch 调参器（TPE / Anneal 降级为 Random）。不需要外部 NNI Manager 服务。

## 镜像

```text
10.121.177.20:8082/mlops/hyperparam-search-nni:20260626-v1
```

## 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input_csv` | (必填) | 训练数据 CSV |
| `--label_col` | `label` | 标签列 |
| `--task_type` | `classification` | `classification` / `regression` |
| `--search_space` | (空) | NNI 格式 JSON，空则用默认空间 |
| `--tuner_name` | `TPE` | `TPE` / `Random` / `Anneal` / `GridSearch` |
| `--max_trial_number` | `10` | 最大 trial 数 |
| `--trial_concurrency` | `1` | 并发数 |
| `--cv` | `3` | 交叉验证折数 |
| `--scoring` | `auto` | 评分函数 |
| `--test_size` | `0.2` | 测试集比例 |
| `--random_state` | `42` | 随机种子 |
| `--output_model_path` | (必填) | 模型输出路径 |
| `--output_metrics_path` | (必填) | 指标输出路径 |
| `--output_nni_result_path` | (可选) | NNI 实验结果路径 |

## 使用示例

```bash
python3 launcher.py \
  --input_csv /mnt/storage/zzzdn/gbdt/train.csv \
  --label_col label \
  --task_type classification \
  --tuner_name Random \
  --max_trial_number 3 \
  --cv 2 \
  --output_model_path /mnt/storage/zzzdn/hyperparam-search-nni/output/model.pkl \
  --output_metrics_path /mnt/storage/zzzdn/hyperparam-search-nni/output/metrics.json \
  --output_nni_result_path /mnt/storage/zzzdn/hyperparam-search-nni/output/nni_result.json
```

## 默认搜索空间

```json
{
  "n_estimators": {"_type": "choice", "_value": [50, 100, 200]},
  "learning_rate": {"_type": "choice", "_value": [0.03, 0.05, 0.1, 0.2]},
  "max_depth": {"_type": "choice", "_value": [2, 3, 5]},
  "subsample": {"_type": "choice", "_value": [0.8, 1.0]}
}
```

## 构建与推送

```bash
cd job-template/job/hyperparam-search-nni
bash build.sh
```

## Pipeline 路径

必须填写容器内真实挂载路径，例如：

```text
input_csv=/mnt/storage/zzzdn/gbdt/train.csv
output_model_path=/mnt/storage/zzzdn/hyperparam-search-nni/output/model.pkl
output_metrics_path=/mnt/storage/zzzdn/hyperparam-search-nni/output/metrics.json
output_nni_result_path=/mnt/storage/zzzdn/hyperparam-search-nni/output/nni_result.json
```
