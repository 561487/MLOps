#!/usr/bin/env python3
"""hyperparam-search-nni — NNI-compatible multi-model hyperparameter search."""

import argparse, json, os, sys, itertools, random as _random, time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib, numpy as np, pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, KFold, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC, SVR

try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

# ---------------------------------------------------------------------------
SUPPORTED_TASK_TYPES = {"classification", "regression"}
SUPPORTED_TUNERS = {"TPE", "Random", "Anneal", "GridSearch"}
FALLBACK_TUNERS = {"TPE", "Anneal"}
CLF_ONLY = {"logistic_regression"}
REG_ONLY = {"ridge"}

MODEL_TYPES = [
    "gbdt", "random_forest", "extra_trees", "ada_boost",
    "svm", "logistic_regression", "ridge", "xgboost", "lightgbm",
]


def build_estimator(model_type: str, task_type: str, random_state: int):
    mt = model_type.lower()
    if task_type == "classification":
        if mt in REG_ONLY: raise ValueError(f"model_type={model_type} 不支持 classification")
        if mt == "gbdt": return GradientBoostingClassifier(random_state=random_state)
        if mt == "random_forest": return RandomForestClassifier(random_state=random_state, n_jobs=-1)
        if mt == "extra_trees": return ExtraTreesClassifier(random_state=random_state, n_jobs=-1)
        if mt == "ada_boost": return AdaBoostClassifier(random_state=random_state)
        if mt == "svm": return SVC(probability=True, random_state=random_state)
        if mt == "logistic_regression": return LogisticRegression(max_iter=1000, n_jobs=-1, random_state=random_state)
        if mt == "xgboost":
            if not _HAS_XGB: raise ImportError("xgboost 未安装")
            return XGBClassifier(random_state=random_state, eval_metric="logloss", n_jobs=-1)
        if mt == "lightgbm":
            if not _HAS_LGB: raise ImportError("lightgbm 未安装")
            return LGBMClassifier(random_state=random_state, n_jobs=-1, verbose=-1)
    else:
        if mt in CLF_ONLY: raise ValueError(f"model_type={model_type} 不支持 regression")
        if mt == "gbdt": return GradientBoostingRegressor(random_state=random_state)
        if mt == "random_forest": return RandomForestRegressor(random_state=random_state, n_jobs=-1)
        if mt == "extra_trees": return ExtraTreesRegressor(random_state=random_state, n_jobs=-1)
        if mt == "ada_boost": return AdaBoostRegressor(random_state=random_state)
        if mt == "svm": return SVR()
        if mt == "ridge": return Ridge(random_state=random_state)
        if mt == "xgboost":
            if not _HAS_XGB: raise ImportError("xgboost 未安装")
            return XGBRegressor(random_state=random_state, n_jobs=-1)
        if mt == "lightgbm":
            if not _HAS_LGB: raise ImportError("lightgbm 未安装")
            return LGBMRegressor(random_state=random_state, n_jobs=-1, verbose=-1)
    raise ValueError(f"未知 model_type: {model_type}")


DEFAULT_NNI_SEARCH_SPACES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "gbdt": {
        "n_estimators": {"_type": "choice", "_value": [50, 100, 200]},
        "learning_rate": {"_type": "choice", "_value": [0.03, 0.05, 0.1, 0.2]},
        "max_depth": {"_type": "choice", "_value": [2, 3, 5]},
        "subsample": {"_type": "choice", "_value": [0.8, 1.0]},
    },
    "random_forest": {
        "n_estimators": {"_type": "choice", "_value": [100, 200]},
        "max_depth": {"_type": "choice", "_value": [None, 5, 10]},
        "min_samples_split": {"_type": "choice", "_value": [2, 5]},
        "min_samples_leaf": {"_type": "choice", "_value": [1, 2]},
    },
    "extra_trees": {
        "n_estimators": {"_type": "choice", "_value": [100, 200]},
        "max_depth": {"_type": "choice", "_value": [None, 5, 10]},
        "min_samples_split": {"_type": "choice", "_value": [2, 5]},
        "min_samples_leaf": {"_type": "choice", "_value": [1, 2]},
    },
    "ada_boost": {
        "n_estimators": {"_type": "choice", "_value": [50, 100, 200]},
        "learning_rate": {"_type": "choice", "_value": [0.03, 0.05, 0.1, 0.2]},
    },
    "svm": {
        "C": {"_type": "choice", "_value": [0.1, 1.0, 10.0]},
        "kernel": {"_type": "choice", "_value": ["linear", "rbf"]},
        "gamma": {"_type": "choice", "_value": ["scale"]},
    },
    "logistic_regression": {
        "C": {"_type": "choice", "_value": [0.1, 1.0, 10.0]},
        "solver": {"_type": "choice", "_value": ["lbfgs", "liblinear"]},
    },
    "ridge": {
        "alpha": {"_type": "choice", "_value": [0.1, 1.0, 10.0, 100.0]},
    },
    "xgboost": {
        "n_estimators": {"_type": "choice", "_value": [50, 100, 200]},
        "learning_rate": {"_type": "choice", "_value": [0.03, 0.05, 0.1]},
        "max_depth": {"_type": "choice", "_value": [3, 5, 7]},
        "subsample": {"_type": "choice", "_value": [0.8, 1.0]},
    },
    "lightgbm": {
        "n_estimators": {"_type": "choice", "_value": [50, 100, 200]},
        "learning_rate": {"_type": "choice", "_value": [0.03, 0.05, 0.1]},
        "num_leaves": {"_type": "choice", "_value": [15, 31, 63]},
        "max_depth": {"_type": "choice", "_value": [-1, 5, 10]},
    },
}

# ---------------------------------------------------------------------------
def log(msg): print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def json_safe(obj):
    if isinstance(obj, dict): return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)): return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if obj is None: return None
    return obj

def make_one_hot_encoder():
    try: return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError: return OneHotEncoder(handle_unknown="ignore", sparse=False)

def parse_search_space(raw: str) -> Dict[str, Dict[str, Any]]:
    if not raw or not raw.strip(): return {}
    try: space = json.loads(raw)
    except json.JSONDecodeError as e: raise ValueError(f"search_space JSON 解析失败: {e}")
    if not isinstance(space, dict): raise ValueError("search_space 必须是 JSON 对象")
    for k, s in space.items():
        if not isinstance(s, dict) or "_type" not in s:
            raise ValueError(f"search_space 中 {k} 格式错误，需要 _type 字段")
    return space

def expand_param_choices(space: Dict) -> Dict[str, List[Any]]:
    choices = {}
    for key, spec in space.items():
        t = spec.get("_type", "choice")
        if t == "choice":
            choices[key] = list(spec.get("_value", []))
        elif t in ("uniform", "quniform", "loguniform", "qloguniform"):
            low, high = spec.get("_value", [0, 1])
            if t == "loguniform": low, high = np.log(max(low, 1e-10)), np.log(max(high, 1e-10))
            samples = int(spec.get("_sample_cnt", 10))
            vals = np.linspace(low, high, samples).tolist()
            if t == "loguniform": vals = np.exp(vals).tolist()
            choices[key] = vals
        elif t == "randint":
            low, high = spec.get("_value", [0, 10])
            choices[key] = list(range(int(low), int(high) + 1))
        else:
            choices[key] = list(spec.get("_value", []))
    return choices

def sample_random(params): return {k: _random.choice(v) for k, v in params.items()}
def generate_grid(params):
    keys = list(params.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*[params[k] for k in keys])]


def parse_args():
    p = argparse.ArgumentParser(description="NNI-compatible multi-model hyperparameter search")
    p.add_argument("--input_csv", default="")
    p.add_argument("--label_col", default="label")
    p.add_argument("--task_type", default="classification", choices=sorted(SUPPORTED_TASK_TYPES))
    p.add_argument("--model_type", default="gbdt", choices=MODEL_TYPES, help="模型类型")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--search_space", default="")
    p.add_argument("--tuner_name", default="TPE", choices=sorted(SUPPORTED_TUNERS))
    p.add_argument("--max_trial_number", type=int, default=10)
    p.add_argument("--trial_concurrency", type=int, default=1)
    p.add_argument("--cv", type=int, default=3)
    p.add_argument("--scoring", default="auto")
    p.add_argument("--max_train_samples", type=int, default=10000, help="单 trial 最大训练样本数(防 SVM 大数据卡住)")
    p.add_argument("--trial_timeout_seconds", type=int, default=600, help="单 trial 超时秒数")
    p.add_argument("--output_model_path", default="")
    p.add_argument("--output_metrics_path", default="")
    p.add_argument("--output_nni_result_path", default="")
    return p.parse_args()


def main():
    args = parse_args()
    log("========== NNI Hyperparameter Search Start ==========")
    log(f"model_type: {args.model_type}  task_type: {args.task_type}  tuner: {args.tuner_name}")

    if not args.input_csv: log("ERROR: input_csv 不能为空"); return 1
    if not args.output_model_path: log("ERROR: output_model_path 不能为空"); return 1
    if not args.output_metrics_path: log("ERROR: output_metrics_path 不能为空"); return 1
    if not os.path.isfile(args.input_csv): log(f"ERROR: 文件不存在: {args.input_csv}"); return 1

    df = pd.read_csv(args.input_csv)
    if df.empty: log("ERROR: CSV 为空"); return 1
    if args.label_col not in df.columns:
        log(f"ERROR: label_col '{args.label_col}' 不存在; 列: {list(df.columns)}"); return 1

    df = df.dropna(subset=[args.label_col])
    if df.empty: log("ERROR: 删除空标签后数据为空"); return 1
    log(f"数据行数: {len(df)}")

    y_raw = df[args.label_col]; X = df.drop(columns=[args.label_col])
    if args.task_type == "classification":
        y = LabelEncoder().fit_transform(y_raw.astype(str))
    else:
        y = pd.to_numeric(y_raw, errors="coerce")
        valid = y.notna(); X = X.loc[valid]; y = y.loc[valid]
    if len(X) < 2: log("ERROR: 样本不足"); return 1

    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    cat_cols = [c for c in X.columns if c not in num_cols]
    log(f"特征: numeric={len(num_cols)}, categorical={len(cat_cols)}")

    strat = y if args.task_type == "classification" and len(set(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=strat)

    n_train_orig = len(X_train)
    downsampled = False
    if args.max_train_samples > 0 and len(X_train) > args.max_train_samples:
        if args.model_type == "svm":
            log(f"[WARN] model_type=svm, train samples {len(X_train)} > max_train_samples={args.max_train_samples}, downsampling")
        rng = np.random.RandomState(args.random_state)
        idx = rng.choice(len(X_train), size=args.max_train_samples, replace=False)
        X_train = X_train.iloc[idx]
        y_train = y_train[idx] if isinstance(y_train, np.ndarray) else np.array(y_train)[idx]
        downsampled = True
        log(f"Downsampled: {n_train_orig} -> {len(X_train)}")

    log(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    transformers = []
    if num_cols: transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols))
    if cat_cols: transformers.append(("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")), ("ohe", make_one_hot_encoder())]), cat_cols))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    scoring = args.scoring if args.scoring != "auto" else ("accuracy" if args.task_type == "classification" else "r2")

    space = parse_search_space(args.search_space)
    if not space: space = DEFAULT_NNI_SEARCH_SPACES.get(args.model_type, DEFAULT_NNI_SEARCH_SPACES["gbdt"])
    log(f"使用搜索空间 ({args.model_type}): {json.dumps({k: list(v.get('_value',[])) for k,v in space.items()}, ensure_ascii=False, default=str)}")
    param_choices = expand_param_choices(space)

    fallback = args.tuner_name in FALLBACK_TUNERS
    effective = "Random" if fallback else args.tuner_name
    if fallback: log(f"tuner={args.tuner_name} 降级为 Random")

    if effective == "GridSearch":
        configs = generate_grid(param_choices)
        if args.max_trial_number > 0 and len(configs) > args.max_trial_number:
            log(f"GridSearch {len(configs)} 组, 截断至 {args.max_trial_number}")
            configs = configs[:args.max_trial_number]
    else:
        configs = [sample_random(param_choices) for _ in range(args.max_trial_number)]

    log(f"共 {len(configs)} 个 trial")
    t0_total = time.time()
    best_params, best_score, effective_cv = None, float("-inf"), args.cv
    all_results = []

    for i, params in enumerate(configs):
        t0 = time.time()
        mp = {k: int(v) if isinstance(v, float) and v == int(v) else v for k, v in params.items()}
        log(f"[TRIAL {i+1}/{len(configs)}] start params={mp}")
        try:
            est = build_estimator(args.model_type, args.task_type, args.random_state)
            est.set_params(**{k: v for k, v in mp.items() if k in est.get_params()})
            pipe = Pipeline([("pre", preprocessor), ("model", est)])
            n, n_classes = len(X_train), len(set(y_train)) if args.task_type == "classification" else 0
            cv = min(args.cv, n - 1)
            if args.task_type == "classification" and n_classes > 0:
                min_cls = min(pd.Series(y_train).value_counts())
                cv = min(cv, min_cls)
            if cv < 2:
                log(f"  SKIP: cv={cv} < 2 (n={n})")
                continue
            effective_cv = cv
            sp = StratifiedKFold(n_splits=cv, shuffle=True, random_state=args.random_state) if args.task_type == "classification" else KFold(n_splits=cv, shuffle=True, random_state=args.random_state)
            scores = cross_val_score(pipe, X_train, y_train, cv=sp, scoring=scoring, n_jobs=-1)
            score = float(scores.mean())
            elapsed = time.time() - t0
            log(f"[TRIAL {i+1}/{len(configs)}] score={score:.6f} elapsed={elapsed:.1f}s cv={cv}")
            all_results.append({"trial_id": i+1, "params": json_safe(params), "score": score,
                                "status": "success", "elapsed_seconds": round(elapsed, 2)})
            if score > best_score: best_score = score; best_params = dict(params)
        except Exception as exc:
            elapsed = time.time() - t0
            log(f"[TRIAL {i+1}/{len(configs)}] FAILED elapsed={elapsed:.1f}s: {exc}")
            all_results.append({"trial_id": i+1, "params": json_safe(params), "status": "failed",
                                "elapsed_seconds": round(elapsed, 2), "error": str(exc)})

    if best_params is None: log("ERROR: 所有 trial 失败"); return 1
    log(f"最佳参数: {best_params}, score={best_score:.6f}")

    bmp = {k: int(v) if isinstance(v, float) and v == int(v) else v for k, v in best_params.items()}
    best_est = build_estimator(args.model_type, args.task_type, args.random_state)
    best_est.set_params(**{k: v for k, v in bmp.items() if k in best_est.get_params()})
    final_pipe = Pipeline([("pre", preprocessor), ("model", best_est)])
    final_pipe.fit(X_train, y_train)
    y_pred = final_pipe.predict(X_test)

    if args.task_type == "classification":
        tm = {"accuracy": float(accuracy_score(y_test, y_pred)),
              "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
              "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0))}
    else:
        mse = mean_squared_error(y_test, y_pred)
        tm = {"rmse": float(np.sqrt(mse)), "mae": float(mean_absolute_error(y_test, y_pred)),
              "r2": float(r2_score(y_test, y_pred))}

    for p in [args.output_model_path, args.output_metrics_path, args.output_nni_result_path]:
        if p:
            d = os.path.dirname(os.path.abspath(p))
            if d: os.makedirs(d, exist_ok=True)

    artifact = {"best_model": best_est, "best_params": best_params, "best_cv_score": best_score,
                "task_type": args.task_type, "model_type": args.model_type, "label_col": args.label_col,
                "feature_columns": list(X.columns), "tuner_name": args.tuner_name}
    joblib.dump(artifact, args.output_model_path)
    log(f"模型已保存: {args.output_model_path}")

    total_elapsed = time.time() - t0_total
    report = {"task_type": args.task_type, "model_type": args.model_type, "search_engine": "nni-compat",
              "tuner_name": args.tuner_name, "best_params": json_safe(best_params), "best_score": best_score,
              "scoring": scoring, "cv": args.cv, "effective_cv": effective_cv,
              "max_trial_number": args.max_trial_number,
              "trial_concurrency": args.trial_concurrency, "test_metrics": tm,
              "label_col": args.label_col, "input_csv": args.input_csv,
              "output_model_path": args.output_model_path, "n_samples": len(X), "n_features": len(X.columns),
              "n_trials_completed": len([r for r in all_results if r.get("status") == "success"]),
              "fallback_from_tuner": fallback,
              "runtime_info": {
                  "max_train_samples": args.max_train_samples,
                  "downsampled": downsampled,
                  "original_train_samples": n_train_orig,
                  "used_train_samples": len(X_train),
                  "trial_timeout_seconds": args.trial_timeout_seconds,
                  "elapsed_seconds": round(total_elapsed, 2),
              }}
    with open(args.output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(report), f, ensure_ascii=False, indent=2)
    log(f"指标已保存: {args.output_metrics_path}")

    if args.output_nni_result_path:
        nr = {"search_space": space, "tuner_name": args.tuner_name, "model_type": args.model_type,
              "task_type": args.task_type,
              "best_trial": {"params": json_safe(best_params), "score": best_score},
              "trials_completed": len(all_results), "all_results": all_results}
        with open(args.output_nni_result_path, "w", encoding="utf-8") as f:
            json.dump(json_safe(nr), f, ensure_ascii=False, indent=2)
        log(f"NNI 结果已保存: {args.output_nni_result_path}")

    log("========== NNI Hyperparameter Search Finished ==========")
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2), flush=True)
    return 0

if __name__ == "__main__":
    try: sys.exit(main())
    except Exception as exc: log(f"任务执行失败: {type(exc).__name__}: {exc}"); raise
