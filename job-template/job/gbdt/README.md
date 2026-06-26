GBDT 算子模板
该模板用于在 MLOps 平台中运行基于 scikit-learn 的 GBDT 训练任务，支持分类和回归。
主要流程
读取容器内 CSV 文件。
校验标签列和特征列。
自动识别数值特征与类别特征。
数值特征使用 median 填充，类别特征使用 most_frequent 填充并进行 One-Hot 编码。
训练 GradientBoostingClassifier 或 GradientBoostingRegressor。
保存 `model.pkl` 和 `metrics.json`。
示例命令
```bash
python3 launcher.py \
  --input_csv /mnt/storage/models-storage/output/datasets/gbdt/train.csv \
  --label_col label \
  --task_type classification \
  --test_size 0.2 \
  --random_state 42 \
  --n_estimators 100 \
  --learning_rate 0.1 \
  --max_depth 3 \
  --subsample 1.0 \
  --min_samples_split 2 \
  --min_samples_leaf 1 \
  --output_model_path /mnt/storage/models-storage/models/gbdt/model.pkl \
  --output_metrics_path /mnt/storage/models-storage/output/gbdt/metrics.json
```
镜像构建
在 `job-template/job/gbdt` 目录下执行：
```bash
bash build.sh 10.121.177.20:8082/mlops/gbdt:20260625
docker push 10.121.177.20:8082/mlops/gbdt:20260625
```
输出文件
`model.pkl`：包含 preprocessing pipeline、GBDT 模型、特征列、标签列等信息。
`metrics.json`：分类任务输出 accuracy、f1、precision、recall；回归任务输出 rmse、mae、r2。