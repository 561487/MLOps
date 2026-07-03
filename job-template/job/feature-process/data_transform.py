"""
数据变换核心逻辑
支持 boxcox转换、二值化、数据类型转换、dct变换、
     根据函数转换、ma移动平均、多项式展开
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def boxcox_transform(df, columns, lmbda=None):
    """
    Box-Cox 变换：使数据更接近正态分布，要求输入值为正数

    Args:
        df: pandas DataFrame
        columns: 要变换的列名列表
        lmbda: 指定 lambda 值，为 None 时自动搜索最优 lambda

    Returns:
        变换后的 DataFrame
    """
    from scipy import stats

    result = df.copy()
    for col in columns:
        if col not in result.columns:
            logger.warning(f"列 '{col}' 不存在，跳过")
            continue
        col_data = result[col].dropna()
        if col_data.min() <= 0:
            logger.warning(f"列 '{col}' 包含非正值，自动加偏移量 {1 - col_data.min()} 使数据均为正")
            shift = 1.0 - col_data.min()
            result[col] = result[col] + shift
            col_data = result[col].dropna()
        transformed, fitted_lambda = stats.boxcox(col_data, lmbda=lmbda)
        result.loc[result[col].notna(), col] = transformed
        logger.info(f"Box-Cox 变换 列'{col}': lambda={fitted_lambda:.4f}")
    return result


def binarize(df, columns, threshold=0.0):
    """
    二值化：根据阈值将数值转为 0/1

    Args:
        df: pandas DataFrame
        columns: 要二值化的列名列表
        threshold: 阈值，大于阈值置1，否则置0

    Returns:
        二值化后的 DataFrame
    """
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            logger.warning(f"列 '{col}' 不存在，跳过")
            continue
        result[col] = (result[col] > threshold).astype(int)
        logger.info(f"二值化 列'{col}': threshold={threshold}, "
                    f"1的比例={result[col].mean():.2%}")
    return result


def type_convert(df, columns, target_type='float64'):
    """
    数据类型转换：将指定列转换为目标数据类型

    Args:
        df: pandas DataFrame
        columns: 要转换的列名列表
        target_type: 目标数据类型 (float64, int64, str, category 等)

    Returns:
        转换后的 DataFrame
    """
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            logger.warning(f"列 '{col}' 不存在，跳过")
            continue
        original_type = result[col].dtype
        try:
            if target_type == 'category':
                result[col] = result[col].astype('category')
            elif target_type == 'str':
                result[col] = result[col].astype(str)
            else:
                result[col] = pd.to_numeric(result[col], errors='coerce').astype(target_type)
            logger.info(f"类型转换 列'{col}': {original_type} -> {result[col].dtype}")
        except Exception as e:
            logger.error(f"列 '{col}' 转换失败: {e}")
    return result


def dct_transform(df, columns, n_components=None):
    """
    DCT 变换 (离散余弦变换)：对指定列进行 DCT 变换

    Args:
        df: pandas DataFrame
        columns: 要进行 DCT 的列名列表
        n_components: 保留的 DCT 分量数，None 则保留全部

    Returns:
        DCT 变换后的 DataFrame（新列为 {原列名}_dct_{i}）
    """
    from scipy.fft import dct

    result = df.copy()
    for col in columns:
        if col not in result.columns:
            logger.warning(f"列 '{col}' 不存在，跳过")
            continue
        col_data = result[col].dropna().values.astype(np.float64)
        if len(col_data) == 0:
            logger.warning(f"列 '{col}' 无有效数据，跳过")
            continue
        transformed = dct(col_data, norm='ortho')
        n = len(transformed) if n_components is None else min(n_components, len(transformed))
        for i in range(n):
            dct_col_name = f"{col}_dct_{i}"
            result[dct_col_name] = np.nan
            result.loc[result[col].notna(), dct_col_name] = transformed[i]
        logger.info(f"DCT 变换 列'{col}': 生成 {n} 个分量")
        if n_components and n_components < len(transformed):
            result.drop(columns=[col], inplace=True)
            logger.info(f"DCT 变换后丢弃原始列 '{col}'")
    return result


def apply_function(df, columns, func_expr):
    """
    根据函数转换：对指定列应用自定义表达式

    Args:
        df: pandas DataFrame
        columns: 要应用函数的列名列表
        func_expr: 函数表达式，如 "x ** 2"、"np.log(x+1)"、"x * 2 + 10"

    Returns:
        转换后的 DataFrame
    """
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            logger.warning(f"列 '{col}' 不存在，跳过")
            continue
        x = result[col].values.astype(np.float64)
        # 在安全的命名空间中执行表达式
        namespace = {'x': x, 'np': np}
        try:
            new_values = eval(func_expr, {"__builtins__": {}}, namespace)
            new_col = f"{col}_transformed"
            result[new_col] = new_values
            logger.info(f"函数转换 列'{col}': expr='{func_expr}' -> 新列 '{new_col}'")
        except Exception as e:
            logger.error(f"列 '{col}' 函数转换失败: {e}")
    return result


def moving_average(df, columns, window=3, center=True):
    """
    MA 移动平均：对指定列计算滑动窗口平均值

    Args:
        df: pandas DataFrame
        columns: 要计算移动平均的列名列表
        window: 滑动窗口大小
        center: 是否居中窗口

    Returns:
        包含移动平均列的 DataFrame（新列为 {原列名}_ma_{window}）
    """
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            logger.warning(f"列 '{col}' 不存在，跳过")
            continue
        ma_col_name = f"{col}_ma_{window}"
        result[ma_col_name] = result[col].rolling(window=window, center=center).mean()
        logger.info(f"移动平均 列'{col}': window={window}, center={center} -> 新列 '{ma_col_name}'")
    return result


def polynomial_expand(df, columns, degree=2, interaction_only=False):
    """
    多项式展开：对指定列生成多项式特征和交互特征

    Args:
        df: pandas DataFrame
        columns: 要展开的列名列表
        degree: 多项式阶数
        interaction_only: 是否仅生成交互项（不含平方等高阶自项）

    Returns:
        展开后的 DataFrame（新列为 {原名}_1 {原名}_2 等）
    """
    from sklearn.preprocessing import PolynomialFeatures

    result = df.copy()
    valid_cols = [c for c in columns if c in result.columns]
    if not valid_cols:
        logger.warning("没有有效的列可用于多项式展开")
        return result

    X = result[valid_cols].fillna(0).values
    poly = PolynomialFeatures(
        degree=degree,
        interaction_only=interaction_only,
        include_bias=False
    )
    expanded = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(valid_cols)

    for i, name in enumerate(feature_names):
        col_name = name.replace(' ', '_').replace('^', 'pow')
        result[col_name] = expanded[:, i]

    logger.info(f"多项式展开: {len(valid_cols)} 列, degree={degree}, "
                f"生成 {len(feature_names)} 个特征")
    return result


TRANSFORM_FUNCTIONS = {
    'boxcox': boxcox_transform,
    'binarize': binarize,
    'type_convert': type_convert,
    'dct': dct_transform,
    'apply_function': apply_function,
    'moving_average': moving_average,
    'polynomial_expand': polynomial_expand,
}
