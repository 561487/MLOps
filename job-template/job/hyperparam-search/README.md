# hyperparam-search 算子

基于 sklearn `GridSearchCV` / `RandomizedSearchCV` 对 `GradientBoostingClassifier` / `GradientBoostingRegressor` 进行超参数搜索，输出最优模型与评估指标。

## 镜像

```text
10.121.177.20:8082/mlops/hyperparam-search:20260626
```

## 参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--input_csv` | str | (空, 必填) | 训练数据 CSV 路径 |
| `--label_col` | str | `label` | 标签列名 |
| `--task_type` | str | `classification` | `classification` 或 `regression` |
| `--search_method` | str | `random` | `grid` 或 `random` |
| `--param_grid` | str | (空) | 自定义搜索空间 JSON 字符串 |
| `--cv` | int | `3` | 交叉验证折数 |
| `--n_iter` | int | `10` | 随机搜索抽样次数 |
| `--scoring` | str | `auto` | 评分函数, 分类默认 accuracy, 回归默认 r2 |
| `--test_size` | float | `0.2` | 测试集比例 |
| `--random_state` | int | `42` | 随机种子 |
| `--output_model_path` | str | (空, 必填) | 模型输出路径 |
| `--output_metrics_path` | str | (空, 必填) | 指标输出路径 |

## 示例

### 分类

```bash
python3 launcher.py \
  --input_csv /mnt/storage/models-storage/output/datasets/hyperparam-search/train.csv \
  --label_col label \
  --task_type classification \
  --search_method random \
  --cv 3 \
  --n_iter 10 \
  --output_model_path /mnt/storage/models-storage/models/hyperparam-search/best_model.pkl \
  --output_metrics_path /mnt/storage/models-storage/output/hyperparam-search/metrics.json
```

### 回归

```bash
python3 launcher.py \
  --input_csv /mnt/storage/models-storage/output/datasets/hyperparam-search/train.csv \
  --label_col price \
  --task_type regression \
  --search_method grid \
  --cv 5 \
  --output_model_path /mnt/storage/models-storage/models/hyperparam-search/best_model.pkl \
  --output_metrics_path /mnt/storage/models-storage/output/hyperparam-search/metrics.json
```

### 自定义搜索空间

```bash
--param_grid '{"n_estimators":[100,200],"learning_rate":[0.05,0.1],"max_depth":[3,5,7]}'
```

## metrics.json

分类示例:

```json
{
  "task_type": "classification",
  "search_method": "random",
  "best_params": {"model__n_estimators": 200, "model__learning_rate": 0.1, ...},
  "best_cv_score": 0.95,
  "test_metrics": {
    "accuracy": 0.93,
    "f1_macro": 0.91,
    "f1_weighted": 0.93
  }
}
```

回归示例:

```json
{
  "task_type": "regression",
  "search_method": "random",
  "best_params": {...},
  "best_cv_score": -0.23,
  "test_metrics": {
    "rmse": 0.31,
    "mae": 0.21,
    "r2": 0.87
  }
}
```

## 构建与推送

```bash
cd job-template/job/hyperparam-search
bash build.sh
docker push 10.121.177.20:8082/mlops/hyperparam-search:20260626
```

## 本地测试

```bash
cat > /tmp/hyperparam_train.csv <<'EOF'
f1,f2,category,label
1.0,2.0,A,0
1.2,1.9,A,0
2.0,1.0,B,1
2.1,1.1,B,1
0.9,2.2,A,0
2.2,0.8,B,1
1.1,2.1,A,0
2.3,0.7,B,1
EOF

python3 launcher.py \
  --input_csv /tmp/hyperparam_train.csv \
  --label_col label \
  --task_type classification \
  --search_method random \
  --cv 2 --n_iter 2 \
  --output_model_path /tmp/best_model.pkl \
  --output_metrics_path /tmp/metrics.json
```

## Pipeline 路径注意事项

路径必须使用容器内真实路径，例如挂载了 `models-storage` 卷，则填写:

```text
input_csv=/mnt/storage/models-storage/output/datasets/hyperparam-search/train.csv
output_model_path=/mnt/storage/models-storage/models/hyperparam-search/best_model.pkl
output_metrics_path=/mnt/storage/models-storage/output/hyperparam-search/metrics.json
```
