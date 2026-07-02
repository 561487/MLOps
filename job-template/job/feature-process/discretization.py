"""
数据离散化核心逻辑
支持等宽、等频、KMeans聚类离散化
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def equal_width_discretize(df, columns, n_bins=5):
    """
    等宽离散化：将数值范围等分为 n_bins 个区间

    Args:
        df: pandas DataFrame
        columns: 要离散化的列名列表
        n_bins: 分箱数

    Returns:
        离散化后的 DataFrame（原始列被替换为区间标签列）
    """
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            logger.warning(f"列 '{col}' 不存在，跳过")
            continue
        col_data = result[col].dropna()
        if len(col_data) < 2:
            logger.warning(f"列 '{col}' 有效数据不足，跳过")
            continue

        bins = np.linspace(col_data.min(), col_data.max(), n_bins + 1)
        bins[0] = -np.inf  # 包含最小值
        bins[-1] = np.inf  # 包含最大值
        labels = [f"[{bins[i]:.2f}, {bins[i+1]:.2f})" for i in range(n_bins)]

        new_col = f"{col}_bin"
        result[new_col] = pd.cut(result[col], bins=bins, labels=labels, include_lowest=True)
        result.drop(columns=[col], inplace=True)
        logger.info(f"等宽离散化 列'{col}': {n_bins} 箱, 范围=[{col_data.min():.4f}, {col_data.max():.4f}]")
    return result


def equal_freq_discretize(df, columns, n_bins=5):
    """
    等频离散化：每个区间包含大致相同数量的样本

    Args:
        df: pandas DataFrame
        columns: 要离散化的列名列表
        n_bins: 分箱数

    Returns:
        离散化后的 DataFrame
    """
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            logger.warning(f"列 '{col}' 不存在，跳过")
            continue
        col_data = result[col].dropna()
        if len(col_data) < n_bins:
            logger.warning(f"列 '{col}' 有效数据({len(col_data)})少于分箱数({n_bins})，跳过")
            continue

        new_col = f"{col}_bin"
        try:
            result[new_col] = pd.qcut(result[col], q=n_bins, duplicates='drop')
        except ValueError as e:
            logger.warning(f"列 '{col}' 等频分箱失败: {e}，尝试减少分箱数")
            result[new_col] = pd.qcut(result[col], q=min(n_bins, col_data.nunique()),
                                       duplicates='drop')
        result.drop(columns=[col], inplace=True)
        logger.info(f"等频离散化 列'{col}': {n_bins} 箱")
    return result


def kmeans_discretize(df, columns, n_bins=5, random_seed=42):
    """
    KMeans 聚类离散化：用 KMeans 对一维数据聚类，以簇边界作为分箱切点

    Args:
        df: pandas DataFrame
        columns: 要离散化的列名列表
        n_bins: 分箱数
        random_seed: 随机种子

    Returns:
        离散化后的 DataFrame
    """
    from sklearn.cluster import KMeans

    result = df.copy()
    for col in columns:
        if col not in result.columns:
            logger.warning(f"列 '{col}' 不存在，跳过")
            continue
        col_data = result[col].dropna()
        n_actual = min(n_bins, len(col_data.unique()))
        if n_actual < 2:
            logger.warning(f"列 '{col}' 唯一值不足，跳过")
            continue

        X = col_data.values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_actual, random_state=random_seed, n_init=10)
        kmeans.fit(X)
        centroids = sorted(kmeans.cluster_centers_.flatten())

        # 用聚类中心的中点作为切点
        if len(centroids) > 1:
            cut_points = [(centroids[i] + centroids[i+1]) / 2 for i in range(len(centroids) - 1)]
            bins = [-np.inf] + cut_points + [np.inf]
            labels = [f"cluster_{i}" for i in range(n_actual)]
        else:
            bins = [-np.inf, np.inf]
            labels = ["cluster_0"]

        new_col = f"{col}_bin"
        result[new_col] = pd.cut(result[col], bins=bins, labels=labels, include_lowest=True)
        result.drop(columns=[col], inplace=True)
        logger.info(f"KMeans离散化 列'{col}': {n_actual} 簇, "
                    f"centroids={[round(c, 4) for c in centroids]}")
    return result


def custom_discretize(df, columns, cut_points_str):
    """
    自定义切点离散化

    Args:
        df: pandas DataFrame
        columns: 要离散化的列名列表
        cut_points_str: 切点字符串，如 "0,10,50,100"

    Returns:
        离散化后的 DataFrame
    """
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            logger.warning(f"列 '{col}' 不存在，跳过")
            continue

        try:
            cut_points = [float(x.strip()) for x in cut_points_str.split(",")]
        except ValueError:
            logger.error(f"切点格式错误: {cut_points_str}")
            continue

        if len(cut_points) < 1:
            logger.warning(f"切点数不足")
            continue

        bins = [-np.inf] + cut_points + [np.inf]
        n_bins = len(bins) - 1
        labels = [f"level_{i}" for i in range(n_bins)]

        new_col = f"{col}_bin"
        result[new_col] = pd.cut(result[col], bins=bins, labels=labels, include_lowest=True)
        result.drop(columns=[col], inplace=True)
        logger.info(f"自定义离散化 列'{col}': 切点={cut_points}")
    return result


DISCRETIZE_FUNCTIONS = {
    'equal_width': equal_width_discretize,
    'equal_freq': equal_freq_discretize,
    'kmeans': kmeans_discretize,
    'custom': custom_discretize,
}
