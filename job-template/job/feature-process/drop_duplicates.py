"""
去除重复样本核心逻辑
支持精确去重和模糊去重
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def exact_drop_duplicates(df, subset=None, keep='first'):
    """
    精确去重：基于完全相同的行值进行去重

    Args:
        df: pandas DataFrame
        subset: 去重依据的列名列表，None 表示所有列
        keep: 'first' 保留第一条 | 'last' 保留最后一条 | False 全部丢弃

    Returns:
        去重后的 DataFrame
    """
    n_before = len(df)
    result = df.drop_duplicates(subset=subset, keep=keep)
    n_after = len(result)
    logger.info(f"精确去重: {n_before} -> {n_after} 行 "
                f"(删除 {n_before - n_after} 条, 保留策略: {keep})")
    return result


def fuzzy_drop_duplicates(df, subset=None, threshold=0.95, keep='first'):
    """
    模糊去重：基于数值列的余弦相似度进行近似去重

    计算每两行之间在指定数值列上的相似度，若相似度 >= threshold 则视为重复。
    采用贪心策略：遍历排序后的行，跳过与已保留行相似度 >= threshold 的行。

    Args:
        df: pandas DataFrame
        subset: 用于计算相似度的数值列名列表，None 表示所有数值列
        threshold: 相似度阈值，0~1，越大越严格（默认 0.95）
        keep: 'first' 保留第一条 | 'last' 保留最后一条

    Returns:
        去重后的 DataFrame
    """
    if subset is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [c for c in subset if c in df.columns
                        and np.issubdtype(df[c].dtype, np.number)]
    if not numeric_cols:
        logger.warning("没有可用的数值列进行模糊去重，回退到精确去重")
        return exact_drop_duplicates(df, subset=subset, keep=keep)

    n_before = len(df)
    # 提取数值矩阵，填充缺失值为列均值
    data = df[numeric_cols].copy()
    for col in numeric_cols:
        col_mean = data[col].mean()
        data[col] = data[col].fillna(col_mean if not np.isnan(col_mean) else 0.0)

    matrix = data.values.astype(np.float64)
    # 行归一化，用于快速计算余弦相似度
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1e-10
    normalized = matrix / norms

    keep_mask = np.ones(len(df), dtype=bool)
    # 按原始顺序处理
    order = range(len(df)) if keep == 'first' else range(len(df) - 1, -1, -1)

    kept_indices = []
    for i in order:
        if not keep_mask[i]:
            continue
        kept_indices.append(i)
        # 与当前行相似度 >= threshold 的标记为重复
        sim = np.dot(normalized, normalized[i])
        for j in range(len(df)):
            if j != i and sim[j] >= threshold:
                keep_mask[j] = False

    result = df.iloc[sorted(kept_indices)].reset_index(drop=True)
    n_after = len(result)
    logger.info(f"模糊去重 (threshold={threshold}, cols={numeric_cols}): "
                f"{n_before} -> {n_after} 行 (删除 {n_before - n_after} 条)")
    return result


def drop_duplicates(df, method='exact', subset=None, keep='first', threshold=0.95):
    """
    统一入口：根据 method 调用对应的去重函数

    Args:
        df: pandas DataFrame
        method: 去重方法，'exact' 精确去重 | 'fuzzy' 模糊去重
        subset: 去重依据的列名列表，None 表示所有列
        keep: 保留策略，'first' | 'last' | False
        threshold: 模糊去重的相似度阈值

    Returns:
        去重后的 DataFrame
    """
    if method == 'exact':
        return exact_drop_duplicates(df, subset=subset, keep=keep)
    elif method == 'fuzzy':
        return fuzzy_drop_duplicates(df, subset=subset, threshold=threshold, keep=keep)
    else:
        raise ValueError(f"不支持的去重方法: {method}，可选: exact, fuzzy")
