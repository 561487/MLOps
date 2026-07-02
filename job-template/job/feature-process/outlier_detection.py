"""
异常值检测核心逻辑
支持 Z-Score、IQR、Isolation Forest、Percentile 四种检测方法
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def zscore_detect(df, columns, threshold=3.0):
    """
    Z-Score 检测：将 |z-score| > threshold 的值标记为异常

    Args:
        df: pandas DataFrame
        columns: 要检测的列名列表
        threshold: Z-Score 阈值，默认 3.0

    Returns:
        DataFrame，新增 {col}_outlier 标记列（1=异常，0=正常）
    """
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            logger.warning(f"列 '{col}' 不存在，跳过")
            continue
        col_data = result[col].dropna()
        if len(col_data) == 0:
            logger.warning(f"列 '{col}' 无有效数据，跳过")
            continue
        mean = col_data.mean()
        std = col_data.std()
        if std < 1e-10:
            logger.warning(f"列 '{col}' 标准差接近0，跳过")
            continue
        z_scores = np.abs((result[col] - mean) / std)
        outlier_col = f"{col}_outlier"
        result[outlier_col] = (z_scores > threshold).astype(int)
        n_outliers = result[outlier_col].sum()
        logger.info(f"Z-Score检测 列'{col}': {n_outliers} 个异常值 "
                    f"({n_outliers/len(result):.2%}, threshold={threshold})")
    return result


def iqr_detect(df, columns, multiplier=1.5):
    """
    IQR 检测：将超出 [Q1 - k*IQR, Q3 + k*IQR] 的值标记为异常

    Args:
        df: pandas DataFrame
        columns: 要检测的列名列表
        multiplier: IQR 倍数，默认 1.5

    Returns:
        DataFrame，新增 {col}_outlier 标记列
    """
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            logger.warning(f"列 '{col}' 不存在，跳过")
            continue
        col_data = result[col].dropna()
        if len(col_data) == 0:
            logger.warning(f"列 '{col}' 无有效数据，跳过")
            continue
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        if iqr < 1e-10:
            logger.warning(f"列 '{col}' IQR 接近0，跳过")
            continue
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        outlier_col = f"{col}_outlier"
        result[outlier_col] = ((result[col] < lower) | (result[col] > upper)).astype(int)
        n_outliers = result[outlier_col].sum()
        logger.info(f"IQR检测 列'{col}': {n_outliers} 个异常值 "
                    f"({n_outliers/len(result):.2%}, "
                    f"范围=[{lower:.4f}, {upper:.4f}])")
    return result


def isolation_forest_detect(df, columns, contamination=0.1, random_seed=42):
    """
    Isolation Forest 检测：使用孤立森林算法检测异常值

    Args:
        df: pandas DataFrame
        columns: 要检测的列名列表
        contamination: 预期异常比例，默认 0.1
        random_seed: 随机种子

    Returns:
        DataFrame，新增 _iforest_outlier 列（检测列联合判定）
    """
    from sklearn.ensemble import IsolationForest

    result = df.copy()
    valid_cols = [c for c in columns if c in result.columns]
    if not valid_cols:
        logger.warning("没有有效的列可用于 Isolation Forest")
        return result

    X = result[valid_cols].copy()
    # 用中位数填充缺失值
    for c in valid_cols:
        med = X[c].median()
        X[c] = X[c].fillna(med)

    model = IsolationForest(
        contamination=contamination,
        random_state=random_seed,
        n_estimators=100
    )
    preds = model.fit_predict(X)
    # IsolationForest: 1=正常, -1=异常 -> 转为 0=正常, 1=异常
    result['_iforest_outlier'] = (preds == -1).astype(int)
    n_outliers = result['_iforest_outlier'].sum()
    logger.info(f"IsolationForest检测: {n_outliers} 个异常值 "
                f"({n_outliers/len(result):.2%}, contamination={contamination})")
    return result


def percentile_detect(df, columns, lower_pct=1.0, upper_pct=99.0):
    """
    Percentile 检测：将超出指定百分位范围的值标记为异常

    Args:
        df: pandas DataFrame
        columns: 要检测的列名列表
        lower_pct: 下界百分位，默认 1.0
        upper_pct: 上界百分位，默认 99.0

    Returns:
        DataFrame，新增 {col}_outlier 标记列
    """
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            logger.warning(f"列 '{col}' 不存在，跳过")
            continue
        col_data = result[col].dropna()
        if len(col_data) == 0:
            logger.warning(f"列 '{col}' 无有效数据，跳过")
            continue
        lower = np.percentile(col_data, lower_pct)
        upper = np.percentile(col_data, upper_pct)
        outlier_col = f"{col}_outlier"
        result[outlier_col] = ((result[col] < lower) | (result[col] > upper)).astype(int)
        n_outliers = result[outlier_col].sum()
        logger.info(f"Percentile检测 列'{col}': {n_outliers} 个异常值 "
                    f"({n_outliers/len(result):.2%}, "
                    f"范围=[{lower:.4f}, {upper:.4f}])")
    return result


DETECT_FUNCTIONS = {
    'zscore': zscore_detect,
    'iqr': iqr_detect,
    'isolation_forest': isolation_forest_detect,
    'percentile': percentile_detect,
}
