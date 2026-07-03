"""
降维核心逻辑
支持 PCA 降维、卡方特征选择
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def pca_reduce(df, columns=None, n_components=2, keep_original=False):
    """
    PCA 降维：主成分分析

    Args:
        df: pandas DataFrame
        columns: 参与降维的列名列表，None 为所有数值列
        n_components: 保留的主成分数 (int 或 0~1 表示解释方差比例)
        keep_original: 是否保留原始列

    Returns:
        降维后的 DataFrame
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    result = df.copy()
    if columns is None:
        columns = result.select_dtypes(include=[np.number]).columns.tolist()
    else:
        columns = [c for c in columns if c in result.columns]

    if not columns:
        logger.error("没有有效的数值列")
        return result

    if len(columns) < 2:
        logger.warning("至少需要 2 列才能进行 PCA 降维")
        return result

    X = result[columns].fillna(result[columns].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 如果 n_components 是 float 且小于1，视为解释方差比例
    if isinstance(n_components, float) and n_components < 1:
        pca = PCA(n_components=n_components)
    else:
        pca = PCA(n_components=min(int(n_components), len(columns)))

    X_pca = pca.fit_transform(X_scaled)

    pca_cols = [f"pca_{i+1}" for i in range(X_pca.shape[1])]
    pca_df = pd.DataFrame(X_pca, columns=pca_cols)

    evr = pca.explained_variance_ratio_
    logger.info(f"PCA 降维: {len(columns)} 列 -> {len(pca_cols)} 个主成分")
    logger.info(f"解释方差比: {[f'{v:.4f}' for v in evr]}, "
                f"累计: {evr.sum():.4f}")

    if keep_original:
        result = pd.concat([result, pca_df], axis=1)
    else:
        result = result.drop(columns=columns)
        result = pd.concat([result, pca_df], axis=1)

    return result


def chi2_select(df, feature_columns, target_column, k=10):
    """
    卡方特征选择：基于卡方检验选择与目标最相关的 k 个特征

    Args:
        df: pandas DataFrame
        feature_columns: 候选特征列名列表
        target_column: 目标变量列名
        k: 选择的特征数

    Returns:
        (降维后的 DataFrame, 特征得分 DataFrame)
    """
    from sklearn.feature_selection import chi2, SelectKBest

    result = df.copy()
    if target_column not in result.columns:
        logger.error(f"目标列 '{target_column}' 不存在")
        return result, pd.DataFrame()

    feature_columns = [c for c in feature_columns if c in result.columns]
    if not feature_columns:
        logger.error("没有有效的特征列")
        return result, pd.DataFrame()

    X = result[feature_columns].fillna(result[feature_columns].median())
    # 将数值缩放到非负范围（卡方检验要求非负）
    X = X - X.min()
    y = result[target_column]

    # 确保 y 是分类标签
    if y.dtype.kind in 'fc':
        y = pd.qcut(y, q=5, labels=False, duplicates='drop')

    k_actual = min(k, len(feature_columns))
    selector = SelectKBest(chi2, k=k_actual)
    selector.fit(X, y)

    scores = pd.DataFrame({
        'feature': feature_columns,
        'chi2_score': selector.scores_,
    }).sort_values('chi2_score', ascending=False)

    selected = scores.head(k_actual)['feature'].tolist()
    logger.info(f"卡方选择: {len(feature_columns)} 特征 -> {len(selected)} 个, "
                f"top5: {selected[:5]}")

    # 只保留选择的特征 + 目标列 + 其他列
    keep_cols = [c for c in result.columns if c not in feature_columns] + selected
    result = result[keep_cols]

    return result, scores


REDUCTION_FUNCTIONS = {
    'pca': pca_reduce,
    'chi2': chi2_select,
}
