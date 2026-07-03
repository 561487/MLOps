"""
数据拆分类核心逻辑
支持列内拆分、列间拆分、行间拆分（训练/测试/验证）、SVD分解
"""
import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


def split_column(df, column, delimiter=',', new_columns=None):
    """
    列内拆分：将一列按分隔符拆分为多列

    Args:
        df: pandas DataFrame
        column: 要拆分的列名
        delimiter: 分隔符
        new_columns: 新列名列表，None 则自动命名

    Returns:
        拆分后的 DataFrame
    """
    if column not in df.columns:
        logger.error(f"列 '{column}' 不存在")
        return df

    result = df.copy()
    split_df = result[column].astype(str).str.split(delimiter, expand=True)

    n_cols = split_df.shape[1]
    if new_columns:
        new_cols = [c.strip() for c in new_columns.split(",")]
        new_cols = new_cols[:n_cols]
        if len(new_cols) < n_cols:
            new_cols += [f"{column}_{i}" for i in range(len(new_cols), n_cols)]
    else:
        new_cols = [f"{column}_{i}" for i in range(n_cols)]

    split_df.columns = new_cols
    col_idx = result.columns.get_loc(column)
    result = pd.concat([
        result.iloc[:, :col_idx],
        split_df,
        result.iloc[:, col_idx + 1:]
    ], axis=1)

    logger.info(f"列内拆分: '{column}' -> {n_cols} 列 ({new_cols})")
    return result


def split_columns_by_index(df, indices_str):
    """
    列间拆分：按列索引位置分组拆分（输出多个子集）

    Args:
        df: pandas DataFrame
        indices_str: 分组索引，如 "0,1,2:3,4,5" 表示前3列一组、后3列一组

    Returns:
        (结果合并后的 DataFrame, 也可分别保存)
        这里将所有拆分结果横向拼接，列名加 _group{N} 后缀
    """
    groups = []
    group_names = []
    for item in indices_str.split("|"):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            name, indices = item.split(":", 1)
            group_names.append(name.strip())
            idxs = [int(x.strip()) for x in indices.split(",")]
        else:
            group_names.append(f"group_{len(groups)}")
            idxs = [int(x.strip()) for x in item.split(",")]

        valid_idxs = [i for i in idxs if 0 <= i < len(df.columns)]
        invalid = [i for i in idxs if i not in valid_idxs]
        if invalid:
            logger.warning(f"索引超出范围: {invalid}")
        groups.append(df.iloc[:, valid_idxs])

    if not groups:
        logger.error("没有有效的列分组")
        return df

    result = pd.concat(groups, axis=1)
    logger.info(f"列间拆分: {len(df.columns)} 列 -> {len(groups)} 组, "
                f"结果 {len(result.columns)} 列")
    return result


def split_rows(df, ratios='0.7,0.15,0.15', random_seed=42, stratify_column=None,
               output_files=None):
    """
    行间拆分：将数据按比例拆分为训练/测试/验证集

    Args:
        df: pandas DataFrame
        ratios: 拆分比例，如 "0.7,0.15,0.15"
        ratios的格式会影响输出文件数
        random_seed: 随机种子
        stratify_column: 分层列名
        output_files: 输出文件路径列表

    Returns:
        拆分结果列表 [(name, DataFrame), ...]
    """
    from sklearn.model_selection import train_test_split

    ratio_list = [float(r.strip()) for r in ratios.split(",")]
    total = sum(ratio_list)
    ratio_list = [r / total for r in ratio_list]

    data = df.copy()
    if stratify_column and stratify_column in data.columns:
        stratify = data[stratify_column]
    else:
        stratify = None

    results = []
    remaining = data
    remaining_stratify = stratify

    for i in range(len(ratio_list) - 1):
        test_size = ratio_list[i] / sum(ratio_list[i:])
        train, test = train_test_split(
            remaining,
            test_size=test_size,
            random_state=random_seed,
            stratify=remaining_stratify
        )
        name = ['train', 'valid', 'test'][i] if i < 3 else f"split_{i}"
        results.append((name, train))
        remaining = test
        if stratify is not None:
            remaining_stratify = remaining[stratify_column]

    name = ['train', 'valid', 'test'][len(ratio_list) - 1] if len(ratio_list) - 1 < 3 else f"split_{len(ratio_list)-1}"
    results.append((name, remaining))

    sizes = ', '.join([f"{n}={len(d)}" for n, d in results])
    logger.info(f"行间拆分: {len(df)} 行 -> {sizes}")
    return results


def svd_decompose(df, columns=None, n_components=2):
    """
    SVD 奇异值分解：对数据矩阵进行截断 SVD 分解

    Args:
        df: pandas DataFrame
        columns: 参与分解的列
        n_components: 保留的奇异向量数

    Returns:
        SVD 分解后的 DataFrame（U 矩阵的前 n_components 列）
    """
    from sklearn.decomposition import TruncatedSVD

    result = df.copy()
    if columns is None:
        columns = result.select_dtypes(include=[np.number]).columns.tolist()
    else:
        columns = [c for c in columns if c in result.columns]

    if not columns or len(columns) < 2:
        logger.error("至少需要 2 个数值列进行 SVD 分解")
        return result

    X = result[columns].fillna(result[columns].median())
    n_actual = min(n_components, len(columns))
    svd = TruncatedSVD(n_components=n_actual, random_state=42)
    X_svd = svd.fit_transform(X)

    svd_cols = [f"svd_{i+1}" for i in range(n_actual)]
    svd_df = pd.DataFrame(X_svd, columns=svd_cols)
    result = result.drop(columns=columns)
    result = pd.concat([result, svd_df], axis=1)

    logger.info(f"SVD 分解: {len(columns)} 列 -> {n_actual} 个奇异向量, "
                f"解释方差比: {[f'{v:.4f}' for v in svd.explained_variance_ratio_]}, "
                f"累计: {svd.explained_variance_ratio_.sum():.4f}")
    return result


SPLIT_FUNCTIONS = {
    'column_split': split_column,
    'column_group': split_columns_by_index,
    'row_split': split_rows,
    'svd': svd_decompose,
}
