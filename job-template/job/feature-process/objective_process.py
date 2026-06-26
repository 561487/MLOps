"""
非数值型变量处理核心逻辑
支持 Hash编码、统计量编码(频率/目标)、One-Hot编码
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def hash_encode(df, columns, n_components=8):
    """
    Hash 编码：将类别值通过哈希映射到固定数量的特征列

    Args:
        df: pandas DataFrame
        columns: 要编码的列名列表
        n_components: 哈希后输出的特征维度

    Returns:
        编码后的 DataFrame（原始列被替换为 {列名}_hash_{0..n_components-1}）
    """
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            logger.warning(f"列 '{col}' 不存在，跳过")
            continue

        # 将每个值转为字符串并哈希，然后取模分散到各分量
        hash_values = result[col].astype(str).apply(
            lambda x: abs(hash(x)) % (n_components * 1000)
        )

        for i in range(n_components):
            hash_col = f"{col}_hash_{i}"
            result[hash_col] = (hash_values % (i + 2) == 0).astype(int)
            hash_values = hash_values // (i + 2)

        result.drop(columns=[col], inplace=True)
        logger.info(f"Hash编码 列'{col}': 生成 {n_components} 个哈希特征")
    return result


def frequency_encode(df, columns):
    """
    频率编码：将类别值替换为该值在列中出现的频率

    Args:
        df: pandas DataFrame
        columns: 要编码的列名列表

    Returns:
        编码后的 DataFrame（原始列被替换为频率值，列名不变）
    """
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            logger.warning(f"列 '{col}' 不存在，跳过")
            continue
        freq_map = result[col].value_counts(normalize=True)
        result[col] = result[col].map(freq_map)
        logger.info(f"频率编码 列'{col}': {len(freq_map)} 个不同值")
    return result


def target_encode(df, columns, target_column, smoothing=1.0):
    """
    目标编码：将类别值替换为该类别对应的目标变量均值（带平滑）

    Args:
        df: pandas DataFrame
        columns: 要编码的列名列表
        target_column: 目标变量列名
        smoothing: 平滑参数，越大则越接近全局均值

    Returns:
        编码后的 DataFrame（原始列被替换为目标均值，列名不变）
    """
    result = df.copy()
    if target_column not in result.columns:
        logger.error(f"目标列 '{target_column}' 不存在")
        return result

    global_mean = result[target_column].mean()

    for col in columns:
        if col not in result.columns:
            logger.warning(f"列 '{col}' 不存在，跳过")
            continue

        category_stats = result.groupby(col)[target_column].agg(['mean', 'count'])
        # 带平滑的目标编码: (count * cat_mean + smoothing * global_mean) / (count + smoothing)
        smoothed = (
            (category_stats['count'] * category_stats['mean'] + smoothing * global_mean)
            / (category_stats['count'] + smoothing)
        )
        result[col] = result[col].map(smoothed).fillna(global_mean)
        logger.info(f"目标编码 列'{col}' (目标: {target_column}): "
                    f"{len(category_stats)} 个不同值, 全局均值={global_mean:.4f}")
    return result


def one_hot_encode(df, columns, drop_first=False, max_categories=50):
    """
    One-Hot 编码：将类别值展开为多个二值列

    Args:
        df: pandas DataFrame
        columns: 要编码的列名列表
        drop_first: 是否丢弃每个特征的第一个类别（避免共线性）
        max_categories: 每列最多保留的类别数，超过的稀有类别归为 'other'

    Returns:
        编码后的 DataFrame（原始列被替换为 One-Hot 列）
    """
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            logger.warning(f"列 '{col}' 不存在，跳过")
            continue

        # 统计类别频率，保留 top-k 类别
        value_counts = result[col].value_counts()
        top_categories = value_counts.head(max_categories).index.tolist()

        # 将稀有类别归为 'other'
        col_data = result[col].apply(
            lambda x: x if x in top_categories else '_other_'
        )

        # 生成 One-Hot 编码
        dummies = pd.get_dummies(col_data, prefix=col, drop_first=drop_first)
        dummies = dummies.astype(int)

        # 在原位置插入新列并删除原始列
        col_idx = result.columns.get_loc(col)
        result = pd.concat([
            result.iloc[:, :col_idx],
            dummies,
            result.iloc[:, col_idx + 1:]
        ], axis=1)

        logger.info(f"One-Hot编码 列'{col}': {dummies.shape[1]} 个二值列 "
                    f"(top {max_categories} + other, drop_first={drop_first})")
    return result


ENCODE_FUNCTIONS = {
    'hash': hash_encode,
    'frequency': frequency_encode,
    'target': target_encode,
    'one_hot': one_hot_encode,
}
