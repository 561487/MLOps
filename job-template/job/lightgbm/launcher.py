import argparse
import json
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error
from lightgbm import LGBMClassifier, LGBMRegressor


def parse_args():
    parser = argparse.ArgumentParser(description="LightGBM training operator")

    parser.add_argument("--input_csv", default="", help="训练数据 CSV 路径")
    parser.add_argument("--label_col", default="label", help="标签列名")
    parser.add_argument(
        "--task_type",
        default="classification",
        choices=["classification", "regression"],
        help="任务类型：classification 或 regression"
    )

    parser.add_argument("--test_size", type=float, default=0.2, help="测试集比例")
    parser.add_argument("--random_state", type=int, default=42, help="随机种子")

    parser.add_argument("--n_estimators", type=int, default=100, help="树数量")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="学习率")
    parser.add_argument("--num_leaves", type=int, default=31, help="叶子节点数")
    parser.add_argument("--max_depth", type=int, default=-1, help="最大深度")

    parser.add_argument("--output_model_path", default="", help="模型输出路径")
    parser.add_argument("--output_metrics_path", default="", help="指标输出路径")

    return parser.parse_args()


def main():
    args = parse_args()

    print("========== LightGBM Operator Start ==========")
    print(f"input_csv: {args.input_csv}")
    print(f"label_col: {args.label_col}")
    print(f"task_type: {args.task_type}")
    print(f"output_model_path: {args.output_model_path}")
    print(f"output_metrics_path: {args.output_metrics_path}")

    if not args.input_csv:
        raise ValueError(
            "input_csv 不能为空，请填写训练数据 CSV 在挂载卷中的实际路径，"
            "例如 /mnt/storage/models-storage/output/datasets/lightgbm/train.csv"
        )
    if not args.output_model_path:
        raise ValueError(
            "output_model_path 不能为空，请填写模型输出路径，"
            "例如 /mnt/storage/models-storage/models/lightgbm/model.pkl"
        )
    if not args.output_metrics_path:
        raise ValueError(
            "output_metrics_path 不能为空，请填写指标输出路径，"
            "例如 /mnt/storage/models-storage/output/lightgbm/metrics.json"
        )

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"训练数据不存在: {args.input_csv}")

    df = pd.read_csv(args.input_csv)

    if df.empty:
        raise ValueError(f"训练数据为空: {args.input_csv}")

    if args.label_col not in df.columns:
        raise ValueError(f"标签列 {args.label_col} 不存在，当前字段为: {list(df.columns)}")

    y = df[args.label_col]
    X = df.drop(columns=[args.label_col])

    if X.shape[1] == 0:
        raise ValueError(f"除标签列 {args.label_col} 外没有特征列，当前字段为: {list(df.columns)}")

    X = pd.get_dummies(X, dummy_na=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state
    )

    if args.task_type == "classification":
        model = LGBMClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            max_depth=args.max_depth,
            random_state=args.random_state
        )
    else:
        model = LGBMRegressor(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            max_depth=args.max_depth,
            random_state=args.random_state
        )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    if args.task_type == "classification":
        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1_macro": float(f1_score(y_test, pred, average="macro"))
        }
    else:
        metrics = {
            "mse": float(mean_squared_error(y_test, pred)),
            "mae": float(mean_absolute_error(y_test, pred))
        }

    os.makedirs(os.path.dirname(args.output_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_metrics_path), exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "feature_columns": list(X.columns),
            "label_col": args.label_col,
            "task_type": args.task_type
        },
        args.output_model_path
    )

    with open(args.output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("========== LightGBM Operator Finished ==========")
    print(f"model saved to: {args.output_model_path}")
    print(f"metrics saved to: {args.output_metrics_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()