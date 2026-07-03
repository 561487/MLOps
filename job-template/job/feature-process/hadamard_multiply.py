"""
Hadamard 乘积核心逻辑
计算两个向量/矩阵的逐元素乘积
"""
import pandas as pd
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)


def hadamard_multiply_vectors(vec_a, vec_b):
    """
    两个等长向量的 Hadamard 乘积（逐元素相乘）

    Args:
        vec_a: numpy array
        vec_b: numpy array（需与 vec_a 等长）

    Returns:
        numpy array，逐元素乘积结果
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(f"向量长度不一致: {len(vec_a)} vs {len(vec_b)}")

    result = vec_a * vec_b
    logger.info(f"Hadamard 向量乘积: 长度={len(result)}, "
                f"a范围=[{vec_a.min():.4f}, {vec_a.max():.4f}], "
                f"b范围=[{vec_b.min():.4f}, {vec_b.max():.4f}]")
    return result


def hadamard_multiply_dataframes(df_a, df_b):
    """
    两个 DataFrame 的 Hadamard 乘积（逐元素相乘）

    要求两个 DataFrame 具有相同的 shape，数值列逐元素相乘，
    非数值列保留左表的值

    Args:
        df_a: pandas DataFrame
        df_b: pandas DataFrame

    Returns:
        DataFrame，逐元素乘积结果
    """
    if df_a.shape != df_b.shape:
        raise ValueError(f"DataFrame shape 不一致: {df_a.shape} vs {df_b.shape}")

    numeric_cols_a = df_a.select_dtypes(include=[np.number]).columns
    numeric_cols_b = df_b.select_dtypes(include=[np.number]).columns

    common_numeric = [c for c in numeric_cols_a if c in numeric_cols_b]
    if not common_numeric:
        raise ValueError("没有共同的数值列可用于 Hadamard 乘积")

    result = df_a.copy()
    for col in common_numeric:
        result[col] = df_a[col].values * df_b[col].values

    logger.info(f"Hadamard DataFrame 乘积: shape={result.shape}, "
                f"数值列={common_numeric}")
    return result


def load_vector(file_path):
    """加载向量文件（CSV 单列或 JSON 数组）"""
    if not file_path:
        raise FileNotFoundError(f"文件不存在: {file_path}")

    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON 文件内容不是数组")
        return np.array(data, dtype=np.float64)
    else:
        df = pd.read_csv(file_path)
        if len(df.columns) >= 1:
            col = df.select_dtypes(include=[np.number]).columns
            if len(col) > 0:
                return df[col[0]].dropna().values.astype(np.float64)
            return df.iloc[:, 0].dropna().values.astype(np.float64)
        raise ValueError("CSV 文件中没有可用数据")
