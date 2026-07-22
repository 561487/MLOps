#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""XGBoost training launcher for the MLOps job-template platform.

The launcher trains a classification or regression model from a CSV file,
evaluates it on a hold-out test set, and writes a reusable joblib artifact plus
a JSON metrics report. Preprocessing is stored in the same sklearn Pipeline as
the estimator so downstream prediction can consume the original input columns.
"""

import argparse
import json
import logging
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor


LOGGER = logging.getLogger("xgb-launcher")
SUPPORTED_TASK_TYPES = {"classification", "regression"}
ARTIFACT_VERSION = "1.0"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 XGBoost 训练表格数据分类或回归模型"
    )

    data_group = parser.add_argument_group("数据配置")
    data_group.add_argument("--input_csv", default="", help="训练数据 CSV 路径")
    data_group.add_argument("--label_col", default="label", help="标签列名称")
    data_group.add_argument(
        "--drop_cols",
        default="",
        help="不参与训练的列，多个列名使用英文逗号分隔",
    )
    data_group.add_argument(
        "--task_type",
        default="classification",
        choices=sorted(SUPPORTED_TASK_TYPES),
        help="任务类型：classification 或 regression",
    )

    split_group = parser.add_argument_group("训练参数")
    split_group.add_argument("--test_size", type=float, default=0.2, help="测试集比例")
    split_group.add_argument(
        "--random_state", type=int, default=42, help="数据切分和模型训练随机种子"
    )

    model_group = parser.add_argument_group("模型参数")
    model_group.add_argument(
        "--n_estimators", type=int, default=100, help="Boosting 树数量"
    )
    model_group.add_argument(
        "--learning_rate", type=float, default=0.1, help="学习率"
    )
    model_group.add_argument("--max_depth", type=int, default=6, help="单棵树最大深度")
    model_group.add_argument(
        "--min_child_weight", type=float, default=1.0, help="子节点最小权重和"
    )
    model_group.add_argument(
        "--subsample", type=float, default=1.0, help="每棵树使用的样本比例"
    )
    model_group.add_argument(
        "--colsample_bytree", type=float, default=1.0, help="每棵树使用的特征比例"
    )
    model_group.add_argument(
        "--gamma", type=float, default=0.0, help="节点继续分裂所需最小损失下降"
    )
    model_group.add_argument(
        "--reg_alpha", type=float, default=0.0, help="L1 正则化系数"
    )
    model_group.add_argument(
        "--reg_lambda", type=float, default=1.0, help="L2 正则化系数"
    )
    model_group.add_argument(
        "--n_jobs", type=int, default=-1, help="并行线程数，-1 表示使用全部可用核心"
    )

    output_group = parser.add_argument_group("输出配置")
    output_group.add_argument(
        "--output_model_path", default="", help="模型 joblib/pkl 输出路径"
    )
    output_group.add_argument(
        "--output_metrics_path", default="", help="指标 JSON 输出路径"
    )

    return parser.parse_args(argv)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_comma_separated(value: str) -> List[str]:
    """Parse comma-separated columns, preserving order and removing duplicates."""
    result: List[str] = []
    seen = set()
    for item in (value or "").split(","):
        normalized = item.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_csv:
        raise ValueError("input_csv 不能为空，请填写容器内可访问的训练 CSV 路径")
    if not args.label_col or not args.label_col.strip():
        raise ValueError("label_col 不能为空")
    if not args.output_model_path:
        raise ValueError("output_model_path 不能为空，请填写模型输出路径")
    if not args.output_metrics_path:
        raise ValueError("output_metrics_path 不能为空，请填写指标输出路径")

    if os.path.abspath(args.output_model_path) == os.path.abspath(
        args.output_metrics_path
    ):
        raise ValueError("output_model_path 和 output_metrics_path 不能是同一路径")
    if args.task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(f"task_type 仅支持: {sorted(SUPPORTED_TASK_TYPES)}")
    if not 0 < args.test_size < 1:
        raise ValueError("test_size 必须在 (0, 1) 范围内")
    if args.random_state < 0:
        raise ValueError("random_state 必须为非负整数")
    if args.n_estimators < 1:
        raise ValueError("n_estimators 必须 >= 1")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate 必须 > 0")
    if args.max_depth < 1:
        raise ValueError("max_depth 必须 >= 1")
    if args.min_child_weight < 0:
        raise ValueError("min_child_weight 必须 >= 0")
    if not 0 < args.subsample <= 1:
        raise ValueError("subsample 必须在 (0, 1] 范围内")
    if not 0 < args.colsample_bytree <= 1:
        raise ValueError("colsample_bytree 必须在 (0, 1] 范围内")
    if args.gamma < 0:
        raise ValueError("gamma 必须 >= 0")
    if args.reg_alpha < 0:
        raise ValueError("reg_alpha 必须 >= 0")
    if args.reg_lambda < 0:
        raise ValueError("reg_lambda 必须 >= 0")
    if args.n_jobs != -1 and args.n_jobs < 1:
        raise ValueError("n_jobs 仅支持 -1 或正整数")


def ensure_input_file(path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"训练数据文件不存在: {path}")


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def json_safe(value: Any) -> Any:
    """Convert numpy/pandas values into strict JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def make_one_hot_encoder() -> OneHotEncoder:
    """Create an encoder compatible with sklearn before and after 1.2."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def make_simple_imputer(strategy: str) -> SimpleImputer:
    """Keep all-empty columns when supported by the installed sklearn."""
    try:
        return SimpleImputer(strategy=strategy, keep_empty_features=True)
    except TypeError:
        return SimpleImputer(strategy=strategy)


def build_preprocessor(
    numeric_features: List[str], categorical_features: List[str]
) -> ColumnTransformer:
    transformers = []
    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[("imputer", make_simple_imputer("median"))]
        )
        transformers.append(("num", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", make_simple_imputer("most_frequent")),
                ("onehot", make_one_hot_encoder()),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError("没有可用特征列，请检查 label_col 和 drop_cols")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
    )


def load_dataset(
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    ensure_input_file(args.input_csv)
    LOGGER.info("读取训练数据: %s", args.input_csv)
    dataframe = pd.read_csv(args.input_csv)
    if dataframe.empty:
        raise ValueError("输入 CSV 为空")
    if args.label_col not in dataframe.columns:
        raise ValueError(
            f"标签列不存在: {args.label_col}; 当前列: {list(dataframe.columns)}"
        )

    rows_before_cleaning = len(dataframe)
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
    dataframe = dataframe.dropna(subset=[args.label_col]).copy()
    dropped_label_rows = rows_before_cleaning - len(dataframe)
    if dataframe.empty:
        raise ValueError("去除空标签后数据为空")

    requested_drop_cols = parse_comma_separated(args.drop_cols)
    effective_drop_cols = [
        column
        for column in requested_drop_cols
        if column in dataframe.columns and column != args.label_col
    ]
    ignored_drop_cols = [
        column
        for column in requested_drop_cols
        if column not in dataframe.columns or column == args.label_col
    ]

    feature_columns = [
        column
        for column in dataframe.columns
        if column != args.label_col and column not in effective_drop_cols
    ]
    if not feature_columns:
        raise ValueError("没有可用特征列，请检查 label_col 和 drop_cols")

    features = dataframe.loc[:, feature_columns].copy()
    target = dataframe.loc[:, args.label_col].copy()

    invalid_regression_rows = 0
    if args.task_type == "regression":
        numeric_target = pd.to_numeric(target, errors="coerce")
        finite_mask = numeric_target.notna() & np.isfinite(numeric_target)
        invalid_regression_rows = int((~finite_mask).sum())
        features = features.loc[finite_mask].copy()
        target = numeric_target.loc[finite_mask].astype(float)
        if target.empty:
            raise ValueError("回归标签全部无法转换为有限数值")

    if len(features) < 2:
        raise ValueError("有效样本数不足，至少需要 2 行数据")

    numeric_features = list(features.select_dtypes(include=[np.number]).columns)
    categorical_features = [
        column for column in feature_columns if column not in numeric_features
    ]

    metadata = {
        "rows_before_cleaning": rows_before_cleaning,
        "rows_after_cleaning": len(features),
        "dropped_label_rows": dropped_label_rows,
        "invalid_regression_label_rows": invalid_regression_rows,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "effective_drop_cols": effective_drop_cols,
        "ignored_drop_cols": ignored_drop_cols,
    }
    return features, target, metadata


def encode_target(
    target: pd.Series, task_type: str
) -> Tuple[pd.Series, Optional[LabelEncoder], Optional[List[Any]]]:
    if task_type == "regression":
        return target.astype(float), None, None

    encoder = LabelEncoder()
    try:
        encoded = encoder.fit_transform(target.to_numpy())
    except TypeError as exc:
        raise ValueError(
            "分类标签包含无法统一排序的混合类型，请将标签列统一为字符串或数字"
        ) from exc

    if len(encoder.classes_) < 2:
        raise ValueError("分类任务至少需要 2 个类别")

    encoded_target = pd.Series(encoded, index=target.index, name=target.name)
    return encoded_target, encoder, list(encoder.classes_)


def choose_stratify_y(
    target: pd.Series, task_type: str, test_size: float
) -> Tuple[Optional[pd.Series], List[str]]:
    warnings: List[str] = []
    if task_type != "classification":
        return None, warnings

    value_counts = target.value_counts(dropna=False)
    class_count = len(value_counts)
    if class_count < 2:
        raise ValueError("分类任务至少需要 2 个类别")
    if value_counts.min() < 2:
        warnings.append("部分类别样本少于 2 条，无法分层抽样，已使用普通随机切分")
        return None, warnings

    test_count = int(math.ceil(len(target) * test_size))
    train_count = len(target) - test_count
    if test_count < class_count or train_count < class_count:
        warnings.append("训练集或测试集容量不足以覆盖全部类别，已使用普通随机切分")
        return None, warnings
    return target, warnings


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    stratify_target, warnings = choose_stratify_y(
        target, args.task_type, args.test_size
    )
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=stratify_target,
        )
    except ValueError as exc:
        raise ValueError(f"训练集/测试集切分失败: {exc}") from exc

    if len(X_train) < 1 or len(X_test) < 1:
        raise ValueError("训练集或测试集为空，请调整数据量或 test_size")

    if args.task_type == "classification":
        all_classes = set(target.unique())
        missing_train_classes = all_classes - set(y_train.unique())
        if missing_train_classes:
            moved_indices: List[Any] = []
            for class_value in sorted(missing_train_classes):
                candidates = y_test[y_test == class_value]
                if candidates.empty:
                    continue
                moved_indices.append(candidates.index[0])

            if len(moved_indices) >= len(X_test):
                raise ValueError(
                    "分类数据过少，无法保证训练集包含全部类别且测试集非空"
                )

            X_train = pd.concat([X_train, X_test.loc[moved_indices]])
            y_train = pd.concat([y_train, y_test.loc[moved_indices]])
            X_test = X_test.drop(index=moved_indices)
            y_test = y_test.drop(index=moved_indices)
            warnings.append("普通随机切分导致训练集缺少类别，已将对应样本移入训练集")

    return X_train, X_test, y_train, y_test, warnings


def resolve_xgb_params(
    args: argparse.Namespace, class_count: Optional[int] = None
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "gamma": args.gamma,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "n_jobs": args.n_jobs,
        "random_state": args.random_state,
        "tree_method": "hist",
        "verbosity": 1,
    }
    if args.task_type == "classification":
        if class_count is None or class_count < 2:
            raise ValueError("分类任务无法确定有效类别数")
        if class_count == 2:
            params.update({"objective": "binary:logistic", "eval_metric": "logloss"})
        else:
            params.update(
                {
                    "objective": "multi:softprob",
                    "eval_metric": "mlogloss",
                    "num_class": class_count,
                }
            )
    else:
        params.update({"objective": "reg:squarederror", "eval_metric": "rmse"})
    return params


def build_estimator(task_type: str, model_params: Dict[str, Any]) -> Any:
    if task_type == "classification":
        return XGBClassifier(**model_params)
    return XGBRegressor(**model_params)


def build_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    estimator: Any,
) -> Pipeline:
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def classification_metrics(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    class_count: int,
) -> Tuple[Dict[str, Any], List[str]]:
    predictions = pipeline.predict(features)
    metrics: Dict[str, Any] = {
        "accuracy": accuracy_score(target, predictions),
        "precision_macro": precision_score(
            target, predictions, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            target, predictions, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(target, predictions, average="macro", zero_division=0),
        "f1_weighted": f1_score(
            target, predictions, average="weighted", zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            target, predictions, labels=np.arange(class_count)
        ),
    }
    warnings: List[str] = []

    try:
        probabilities = pipeline.predict_proba(features)
        if class_count == 2:
            if len(np.unique(target)) == 2:
                metrics["roc_auc"] = roc_auc_score(target, probabilities[:, 1])
            else:
                warnings.append("当前数据子集只包含一个类别，已跳过二分类 ROC-AUC")
        elif len(np.unique(target)) == class_count:
            metrics["roc_auc_ovr_macro"] = roc_auc_score(
                target,
                probabilities,
                labels=np.arange(class_count),
                multi_class="ovr",
                average="macro",
            )
        else:
            warnings.append("当前数据子集未覆盖全部类别，已跳过多分类 ROC-AUC")
    except (AttributeError, IndexError, ValueError) as exc:
        warnings.append(f"ROC-AUC 计算失败，已跳过: {exc}")

    return metrics, warnings


def regression_metrics(
    pipeline: Pipeline, features: pd.DataFrame, target: pd.Series
) -> Tuple[Dict[str, Any], List[str]]:
    predictions = pipeline.predict(features)
    mse = mean_squared_error(target, predictions)
    metrics: Dict[str, Any] = {
        "rmse": math.sqrt(float(mse)),
        "mae": mean_absolute_error(target, predictions),
    }
    warnings: List[str] = []
    if len(target) >= 2:
        metrics["r2"] = r2_score(target, predictions)
    else:
        metrics["r2"] = None
        warnings.append("当前数据子集少于 2 条样本，R² 无法计算")
    return metrics, warnings


def extract_feature_importances(
    pipeline: Pipeline, limit: int = 100
) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    estimator = pipeline.named_steps["model"]
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return [], ["当前模型未提供 feature_importances_，已跳过特征重要性"]

    try:
        names = pipeline.named_steps["preprocess"].get_feature_names_out()
    except (AttributeError, ValueError) as exc:
        warnings.append(f"无法获取编码后特征名，已使用索引名称: {exc}")
        names = np.asarray([f"feature_{index}" for index in range(len(importances))])

    if len(names) != len(importances):
        warnings.append("特征名数量与重要性数量不一致，已使用索引名称")
        names = np.asarray([f"feature_{index}" for index in range(len(importances))])

    ranked = sorted(
        (
            {"feature": str(name), "importance": float(importance)}
            for name, importance in zip(names, importances)
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )
    return ranked[:limit], warnings


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_artifact(
    pipeline: Pipeline,
    label_encoder: Optional[LabelEncoder],
    classes: Optional[List[Any]],
    args: argparse.Namespace,
    dataset_metadata: Dict[str, Any],
    model_params: Dict[str, Any],
    created_at: str,
) -> Dict[str, Any]:
    return {
        "artifact_version": ARTIFACT_VERSION,
        "algorithm": "xgboost",
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "task_type": args.task_type,
        "label_col": args.label_col,
        "feature_columns": dataset_metadata["feature_columns"],
        "numeric_features": dataset_metadata["numeric_features"],
        "categorical_features": dataset_metadata["categorical_features"],
        "classes": classes,
        "model_params": model_params,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
        "created_at": created_at,
    }


def train(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    started_at = time.monotonic()
    created_at = utc_now_iso()
    features, raw_target, dataset_metadata = load_dataset(args)
    target, label_encoder, classes = encode_target(raw_target, args.task_type)
    class_count = len(classes) if classes is not None else None

    X_train, X_test, y_train, y_test, split_warnings = split_dataset(
        features, target, args
    )
    dataset_metadata.update(
        {
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "feature_count": len(dataset_metadata["feature_columns"]),
        }
    )

    model_params = resolve_xgb_params(args, class_count)
    estimator = build_estimator(args.task_type, model_params)
    pipeline = build_pipeline(
        dataset_metadata["numeric_features"],
        dataset_metadata["categorical_features"],
        estimator,
    )

    LOGGER.info(
        "开始训练: task=%s, rows=%d, train=%d, test=%d, features=%d",
        args.task_type,
        len(features),
        len(X_train),
        len(X_test),
        len(dataset_metadata["feature_columns"]),
    )
    pipeline.fit(X_train, y_train)

    metric_warnings: List[str] = []
    if args.task_type == "classification":
        assert class_count is not None
        train_metrics, train_warnings = classification_metrics(
            pipeline, X_train, y_train, class_count
        )
        test_metrics, test_warnings = classification_metrics(
            pipeline, X_test, y_test, class_count
        )
    else:
        train_metrics, train_warnings = regression_metrics(
            pipeline, X_train, y_train
        )
        test_metrics, test_warnings = regression_metrics(
            pipeline, X_test, y_test
        )
    metric_warnings.extend([f"训练集: {item}" for item in train_warnings])
    metric_warnings.extend([f"测试集: {item}" for item in test_warnings])

    feature_importances, importance_warnings = extract_feature_importances(pipeline)
    warnings = split_warnings + metric_warnings + importance_warnings
    artifact = build_artifact(
        pipeline,
        label_encoder,
        classes,
        args,
        dataset_metadata,
        model_params,
        created_at,
    )
    report = {
        "status": "success",
        "algorithm": "xgboost",
        "artifact_version": ARTIFACT_VERSION,
        "task_type": args.task_type,
        "dataset": {
            "input_csv": os.path.abspath(args.input_csv),
            **dataset_metadata,
        },
        "classes": classes,
        "model_params": model_params,
        "metrics": {"train": train_metrics, "test": test_metrics},
        "feature_importances": feature_importances,
        "warnings": warnings,
        "duration_seconds": time.monotonic() - started_at,
        "created_at": created_at,
    }
    return artifact, report


def save_outputs(
    artifact: Dict[str, Any], report: Dict[str, Any], args: argparse.Namespace
) -> None:
    ensure_parent_dir(args.output_model_path)
    ensure_parent_dir(args.output_metrics_path)

    joblib.dump(artifact, args.output_model_path)
    with open(args.output_metrics_path, "w", encoding="utf-8") as file_handle:
        json.dump(
            json_safe(report),
            file_handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    LOGGER.info("模型已保存: %s", args.output_model_path)
    LOGGER.info("指标已保存: %s", args.output_metrics_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    artifact, report = train(args)
    save_outputs(artifact, report, args)
    LOGGER.info("XGBoost 训练完成")
    return 0


if __name__ == "__main__":
    configure_logging()
    try:
        sys.exit(main())
    except Exception as error:  # noqa: BLE001 - CLI must expose a clear failure status.
        LOGGER.error("XGBoost 任务失败: %s", error)
        raise
