"""
特征组合核心逻辑
支持列间四则运算、多项式组合、自定义表达式组合
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def arithmetic_combine(df, col_a, col_b, op, new_col_name=None):
    """
    两列间四则运算组合

    Args:
        df: pandas DataFrame
        col_a: 左操作数列名
        col_b: 右操作数列名
        op: 运算符 'add'/'sub'/'mul'/'div'
        new_col_name: 新列名，默认 "{col_a}_{op}_{col_b}"

    Returns:
        DataFrame with new combined column
    """
    result = df.copy()
    if col_a not in result.columns:
        logger.error(f"列 '{col_a}' 不存在")
        return result
    if col_b not in result.columns:
        logger.error(f"列 '{col_b}' 不存在")
        return result

    if new_col_name is None:
        new_col_name = f"{col_a}_{op}_{col_b}"

    a = pd.to_numeric(result[col_a], errors='coerce')
    b = pd.to_numeric(result[col_b], errors='coerce')

    ops = {
        'add': lambda x, y: x + y,
        'sub': lambda x, y: x - y,
        'mul': lambda x, y: x * y,
        'div': lambda x, y: np.where(np.abs(y) < 1e-10, np.nan, x / y),
    }

    if op not in ops:
        logger.error(f"不支持的运算符: {op}，支持 add/sub/mul/div")
        return result

    result[new_col_name] = ops[op](a.values, b.values)
    logger.info(f"四则运算组合: '{col_a}' {op} '{col_b}' -> '{new_col_name}'")
    return result


def polynomial_combine(df, columns, degree=2, interaction_only=False):
    """
    多项式组合：生成列间乘积、平方等多项式特征

    Args:
        df: pandas DataFrame
        columns: 参与组合的列名列表
        degree: 多项式阶数
        interaction_only: 是否仅生成交互项（不含平方等高阶自项）

    Returns:
        DataFrame（仅包含生成的新特征列）
    """
    from sklearn.preprocessing import PolynomialFeatures

    valid_cols = [c for c in columns if c in df.columns]
    invalid = [c for c in columns if c not in df.columns]
    if invalid:
        logger.warning(f"列不存在: {invalid}")
    if not valid_cols:
        logger.error("没有有效的列")
        return pd.DataFrame()

    X = df[valid_cols].fillna(0).values
    poly = PolynomialFeatures(
        degree=degree,
        interaction_only=interaction_only,
        include_bias=False
    )
    expanded = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(valid_cols)

    result = pd.DataFrame()
    for i, name in enumerate(feature_names):
        col_name = f"poly_{name.replace(' ', '_').replace('^', 'pow')}"
        result[col_name] = expanded[:, i]

    logger.info(f"多项式组合: {len(valid_cols)} 列, degree={degree}, "
                f"生成 {len(feature_names)} 个特征")
    return result


def custom_combine(df, columns, expr, new_col_name='combined'):
    """
    自定义表达式组合：对指定列应用 Python 表达式生成新特征

    Args:
        df: pandas DataFrame
        columns: 表达式中引用的列名列表
        expr: 表达式，如 "x0 + x1 * 2 - x2 / 3"
              列按 columns 顺序替换为 x0, x1, x2...
        new_col_name: 新列名

    Returns:
        DataFrame with new combined column
    """
    result = df.copy()
    valid_cols = [c for c in columns if c in result.columns]
    invalid = [c for c in columns if c not in result.columns]
    if invalid:
        logger.warning(f"列不存在: {invalid}")
    if not valid_cols:
        logger.error("没有有效的列")
        return result

    # 构建命名空间，x0=第一列, x1=第二列...
    namespace = {}
    for i, col in enumerate(valid_cols):
        namespace[f'x{i}'] = pd.to_numeric(result[col], errors='coerce').values.astype(np.float64)
    namespace['np'] = np

    try:
        result[new_col_name] = eval(expr, {"__builtins__": {}}, namespace)
        logger.info(f"自定义组合: expr='{expr}', columns={valid_cols} -> '{new_col_name}'")
    except Exception as e:
        logger.error(f"表达式执行失败: {e}")

    return result


def add_subtract_all_pairs(df, columns):
    """
    生成所有列对的加减组合（A+B, A-B, B-A）

    Args:
        df: pandas DataFrame
        columns: 参与组合的列名列表

    Returns:
        DataFrame（原始列 + 新生成的组合列）
    """
    result = df.copy()
    valid_cols = [c for c in columns if c in result.columns]
    if len(valid_cols) < 2:
        logger.warning("至少需要 2 列才能生成列对组合")
        return result

    for i in range(len(valid_cols)):
        for j in range(i + 1, len(valid_cols)):
            a, b = valid_cols[i], valid_cols[j]
            a_vals = pd.to_numeric(result[a], errors='coerce').values
            b_vals = pd.to_numeric(result[b], errors='coerce').values
            result[f"{a}_add_{b}"] = a_vals + b_vals
            result[f"{a}_sub_{b}"] = a_vals - b_vals
            result[f"{b}_sub_{a}"] = b_vals - a_vals
            result[f"{a}_mul_{b}"] = a_vals * b_vals

    n_pairs = len(valid_cols) * (len(valid_cols) - 1) // 2
    logger.info(f"列对组合: {len(valid_cols)} 列 -> {n_pairs} 对, "
                f"每对生成 4 个新列 (add/sub/sub/mul)")
    return result


COMBINE_FUNCTIONS = {
    'arithmetic': arithmetic_combine,
    'polynomial': polynomial_combine,
    'custom': custom_combine,
    'all_pairs': add_subtract_all_pairs,
}
