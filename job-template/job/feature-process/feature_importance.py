"""
特征重要性计算核心逻辑
支持随机森林、方差、互信息、相关性、IV值
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def random_forest_importance(df, feature_columns, target_column, random_seed=42):
    """
    随机森林特征重要性

    Args:
        df: pandas DataFrame
        feature_columns: 特征列名列表
        target_column: 目标变量列名
        random_seed: 随机种子

    Returns:
        DataFrame，列: feature, importance
    """
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

    valid_features = [c for c in feature_columns if c in df.columns]
    if target_column not in df.columns:
        logger.error(f"目标列 '{target_column}' 不存在")
        return pd.DataFrame()
    if not valid_features:
        logger.error("没有有效的特征列")
        return pd.DataFrame()

    X = df[valid_features].fillna(df[valid_features].median())
    y = df[target_column].dropna()
    X = X.loc[y.index]

    # 判断分类还是回归
    if y.nunique() <= 10 or y.dtype.kind in ('i', 'O', 'b'):
        model = RandomForestClassifier(
            n_estimators=100, random_state=random_seed, n_jobs=-1
        )
    else:
        model = RandomForestRegressor(
            n_estimators=100, random_state=random_seed, n_jobs=-1
        )

    model.fit(X, y)
    result = pd.DataFrame({
        'feature': valid_features,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    result['importance_norm'] = result['importance'] / result['importance'].sum()

    logger.info(f"随机森林重要性: {len(valid_features)} 特征, "
                f"top3={result['feature'].head(3).tolist()}")
    return result


def variance_importance(df, feature_columns):
    """
    方差重要性：方差越大的特征被认为越重要

    Args:
        df: pandas DataFrame
        feature_columns: 特征列名列表

    Returns:
        DataFrame，列: feature, variance
    """
    valid_features = [c for c in feature_columns
                      if c in df.columns and np.issubdtype(df[c].dtype, np.number)]
    if not valid_features:
        logger.error("没有有效的数值特征列")
        return pd.DataFrame()

    variances = df[valid_features].var()
    result = pd.DataFrame({
        'feature': valid_features,
        'importance': variances.values,
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    result['importance_norm'] = result['importance'] / result['importance'].sum()

    logger.info(f"方差重要性: {len(valid_features)} 特征")
    return result


def mutual_info_importance(df, feature_columns, target_column, random_seed=42):
    """
    互信息特征重要性

    Args:
        df: pandas DataFrame
        feature_columns: 特征列名列表
        target_column: 目标变量列名
        random_seed: 随机种子

    Returns:
        DataFrame，列: feature, mi_score
    """
    from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

    valid_features = [c for c in feature_columns if c in df.columns]
    if target_column not in df.columns:
        logger.error(f"目标列 '{target_column}' 不存在")
        return pd.DataFrame()
    if not valid_features:
        logger.error("没有有效的特征列")
        return pd.DataFrame()

    X = df[valid_features].fillna(df[valid_features].median())
    y = df[target_column].loc[X.index]

    if y.nunique() <= 10 or y.dtype.kind in ('i', 'O', 'b'):
        mi_scores = mutual_info_classif(X, y, random_state=random_seed)
    else:
        mi_scores = mutual_info_regression(X, y, random_state=random_seed)

    result = pd.DataFrame({
        'feature': valid_features,
        'importance': mi_scores,
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    result['importance_norm'] = result['importance'] / max(result['importance'].sum(), 1e-10)

    logger.info(f"互信息重要性: {len(valid_features)} 特征, "
                f"top3={result['feature'].head(3).tolist()}")
    return result


def correlation_importance(df, feature_columns, target_column):
    """
    相关性重要性：特征与目标的 Pearson 相关系数绝对值

    Args:
        df: pandas DataFrame
        feature_columns: 特征列名列表
        target_column: 目标变量列名

    Returns:
        DataFrame，列: feature, correlation
    """
    valid_features = [c for c in feature_columns
                      if c in df.columns and np.issubdtype(df[c].dtype, np.number)]
    if target_column not in df.columns:
        logger.error(f"目标列 '{target_column}' 不存在")
        return pd.DataFrame()
    if not valid_features:
        logger.error("没有有效的数值特征列")
        return pd.DataFrame()

    correlations = []
    for col in valid_features:
        corr = df[col].corr(df[target_column])
        correlations.append(abs(corr) if not np.isnan(corr) else 0.0)

    result = pd.DataFrame({
        'feature': valid_features,
        'importance': correlations,
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    result['importance_norm'] = result['importance'] / max(result['importance'].sum(), 1e-10)

    logger.info(f"相关性重要性: {len(valid_features)} 特征")
    return result


def iv_importance(df, feature_columns, target_column, n_bins=10):
    """
    IV (Information Value) 值计算，用于二分类问题的特征筛选

    Args:
        df: pandas DataFrame
        feature_columns: 特征列名列表
        target_column: 二分类目标列名（值为 0/1）
        n_bins: 分箱数

    Returns:
        DataFrame，列: feature, iv
    """
    valid_features = [c for c in feature_columns
                      if c in df.columns and np.issubdtype(df[c].dtype, np.number)]
    if target_column not in df.columns:
        logger.error(f"目标列 '{target_column}' 不存在")
        return pd.DataFrame()
    if not valid_features:
        logger.error("没有有效的数值特征列")
        return pd.DataFrame()

    y = df[target_column]
    iv_scores = []

    for col in valid_features:
        x = df[col]
        try:
            bins = pd.qcut(x, q=n_bins, duplicates='drop', labels=False)
        except (ValueError, TypeError):
            bins = pd.cut(x, bins=n_bins, labels=False)

        iv = 0.0
        for bin_val in sorted(set(bins) - {np.nan}):
            mask = bins == bin_val
            good = (y[mask] == 0).sum() + 1e-10
            bad = (y[mask] == 1).sum() + 1e-10
            total_good = (y == 0).sum() + 1e-10
            total_bad = (y == 1).sum() + 1e-10

            # WOE = ln((good/total_good) / (bad/total_bad))
            dist_good = good / total_good
            dist_bad = bad / total_bad
            if dist_good > 0 and dist_bad > 0:
                woe = np.log(dist_good / dist_bad)
                iv += (dist_good - dist_bad) * woe

        iv_scores.append(iv)

    result = pd.DataFrame({
        'feature': valid_features,
        'importance': iv_scores,
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    result['importance_norm'] = result['importance'] / max(result['importance'].sum(), 1e-10)

    logger.info(f"IV值: {len(valid_features)} 特征")
    return result


IMPORTANCE_FUNCTIONS = {
    'random_forest': random_forest_importance,
    'variance': variance_importance,
    'mutual_info': mutual_info_importance,
    'correlation': correlation_importance,
    'iv': iv_importance,
}
