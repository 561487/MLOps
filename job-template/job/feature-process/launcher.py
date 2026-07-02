"""
feature-process: 特征处理统一入口
通过 --process_type 参数分发到不同的特征处理逻辑。
"""
import argparse
import sys
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def handle_calculate_correlation(args):
    """计算两个特征向量间的余弦相似度"""
    import numpy as np
    from correlation import calculate_vector_similarity

    def load_vector(file_path):
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            sys.exit(1)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.error(f"文件内容不是 JSON 数组: {file_path}")
            sys.exit(1)
        if len(data) == 0:
            logger.error(f"向量为空: {file_path}")
            sys.exit(1)
        vector = np.array(data, dtype=np.float64)
        logger.info(f"从 {file_path} 读取向量，长度: {len(vector)}")
        return vector

    logger.info("开始相似度计算")
    logger.info(f"  向量A文件: {args.input_file_a}")
    logger.info(f"  向量B文件: {args.input_file_b}")

    vec_a = load_vector(args.input_file_a)
    vec_b = load_vector(args.input_file_b)

    if len(vec_a) != len(vec_b):
        logger.error(f"两个向量长度不一致: vec_a={len(vec_a)}, vec_b={len(vec_b)}")
        sys.exit(1)

    similarity = calculate_vector_similarity(vec_a, vec_b)
    logger.info(f"余弦相似度: {similarity:.6f}")

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    result = {
        "similarity_score": round(float(similarity), 6),
        "vector_a_length": len(vec_a),
        "vector_b_length": len(vec_b),
    }
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"结果已保存到: {args.output_file}")
    logger.info("相似度计算完成")


def handle_sample(args):
    """采样：随机采样、分层采样、过采样、欠采样"""
    import pandas as pd
    from sample import random_sample, stratified_sample, oversampling, undersampling

    logger.info("开始采样")
    logger.info(f"  输入文件: {args.input_file}")
    logger.info(f"  采样类型: {args.sample_type}")

    # 1. 读取数据
    df = pd.read_csv(args.input_file)
    logger.info(f"  原始数据行数: {len(df)}")

    # 2. 执行采样
    replace = args.replace.lower() == 'true'
    sample_type_func = {
        'random': lambda: random_sample(df, args.sample_rate, args.random_seed, replace),
        'stratified': lambda: stratified_sample(df, args.stratify_column, args.sample_rate, args.random_seed),
        'oversampling': lambda: oversampling(df, args.stratify_column, args.sample_rate, args.random_seed),
        'undersampling': lambda: undersampling(df, args.stratify_column, args.sample_rate, args.random_seed),
    }
    result = sample_type_func[args.sample_type]()

    # 3. 输出结果
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    result.to_csv(args.output_file, index=False)
    logger.info(f"采样完成，结果保存到: {args.output_file}")
    logger.info(f"采样后数据行数: {len(result)}")


def handle_union_join(args):
    """数据合并：union（行级拼接）、join（列级关联）"""
    import pandas as pd
    from union_join import merge_data

    logger.info("开始数据合并")
    logger.info(f"  输入文件A: {args.input_file_a}")
    logger.info(f"  输入文件B: {args.input_file_b}")
    logger.info(f"  合并类型: {args.merge_type}")

    # 读取数据
    if not os.path.exists(args.input_file_a):
        logger.error(f"文件不存在: {args.input_file_a}")
        sys.exit(1)
    if not os.path.exists(args.input_file_b):
        logger.error(f"文件不存在: {args.input_file_b}")
        sys.exit(1)

    df_a = pd.read_csv(args.input_file_a)
    df_b = pd.read_csv(args.input_file_b)
    logger.info(f"  文件A行数: {len(df_a)}, 文件B行数: {len(df_b)}")

    # 解析 join_on（支持逗号分隔的多列关联）
    join_on = None
    if args.join_on:
        join_on = [col.strip() for col in args.join_on.split(",")]

    # 执行合并
    result = merge_data(df_a, df_b, args.merge_type, join_on=join_on, join_how=args.join_how)

    # 输出结果
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    result.to_csv(args.output_file, index=False)
    logger.info(f"合并完成，结果保存到: {args.output_file}")
    logger.info(f"合并后数据行数: {len(result)}")


def handle_drop_duplicates(args):
    """去除重复样本：精确去重、模糊去重"""
    import pandas as pd
    from drop_duplicates import drop_duplicates

    logger.info("开始去除重复样本")
    logger.info(f"  输入文件: {args.input_file}")
    logger.info(f"  去重方法: {args.dedup_method}")

    if not os.path.exists(args.input_file):
        logger.error(f"文件不存在: {args.input_file}")
        sys.exit(1)

    df = pd.read_csv(args.input_file)
    logger.info(f"  原始数据行数: {len(df)}")

    # 解析 subset
    subset = None
    if args.dedup_subset:
        subset = [col.strip() for col in args.dedup_subset.split(",")]

    # 解析 keep
    keep = args.dedup_keep
    if keep and keep.lower() == 'false':
        keep = False

    result = drop_duplicates(df, method=args.dedup_method, subset=subset,
                             keep=keep, threshold=args.dedup_threshold)

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    result.to_csv(args.output_file, index=False)
    logger.info(f"去重完成，结果保存到: {args.output_file}")
    logger.info(f"去重后数据行数: {len(result)}")


def handle_data_transform(args):
    """数据变换：boxcox/二值化/类型转换/dct/函数转换/移动平均/多项式展开"""
    import numpy as np
    import pandas as pd
    from data_transform import TRANSFORM_FUNCTIONS

    logger.info("开始数据变换")
    logger.info(f"  输入文件: {args.input_file}")
    logger.info(f"  变换类型: {args.transform_type}")

    if not os.path.exists(args.input_file):
        logger.error(f"文件不存在: {args.input_file}")
        sys.exit(1)

    df = pd.read_csv(args.input_file)
    logger.info(f"  原始数据: {len(df)} 行, {len(df.columns)} 列")

    # 解析要变换的列
    if args.transform_columns:
        columns = [c.strip() for c in args.transform_columns.split(",")]
    else:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        logger.info(f"  未指定列，自动选择数值列: {columns}")

    # 获取变换函数并执行
    transform_func = TRANSFORM_FUNCTIONS[args.transform_type]

    kwargs = {}
    if args.transform_type == 'boxcox':
        kwargs['lmbda'] = args.boxcox_lambda
    elif args.transform_type == 'binarize':
        kwargs['threshold'] = args.binarize_threshold
    elif args.transform_type == 'type_convert':
        kwargs['target_type'] = args.target_type
    elif args.transform_type == 'dct':
        kwargs['n_components'] = args.dct_n_components
    elif args.transform_type == 'apply_function':
        kwargs['func_expr'] = args.func_expr
    elif args.transform_type == 'moving_average':
        kwargs['window'] = args.ma_window
        kwargs['center'] = args.ma_center.lower() == 'true'
    elif args.transform_type == 'polynomial_expand':
        kwargs['degree'] = args.poly_degree
        kwargs['interaction_only'] = args.poly_interaction_only.lower() == 'true'

    result = transform_func(df, columns, **kwargs)

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    result.to_csv(args.output_file, index=False)
    logger.info(f"数据变换完成，结果保存到: {args.output_file}")
    logger.info(f"变换后数据: {len(result)} 行, {len(result.columns)} 列")


def handle_objective_process(args):
    """非数值型变量处理：hash编码/频率编码/目标编码/One-Hot编码"""
    import pandas as pd
    from objective_process import ENCODE_FUNCTIONS

    logger.info("开始非数值型变量处理")
    logger.info(f"  输入文件: {args.input_file}")
    logger.info(f"  编码类型: {args.encode_type}")

    if not os.path.exists(args.input_file):
        logger.error(f"文件不存在: {args.input_file}")
        sys.exit(1)

    df = pd.read_csv(args.input_file)
    logger.info(f"  原始数据: {len(df)} 行, {len(df.columns)} 列")

    # 解析要编码的列
    if args.encode_columns:
        columns = [c.strip() for c in args.encode_columns.split(",")]
    else:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not columns:
            logger.warning("未找到非数值列，将使用全部列")
            columns = df.columns.tolist()
        logger.info(f"  未指定列，自动选择非数值列: {columns}")

    if not columns:
        logger.error("没有可编码的列")
        sys.exit(1)

    encode_func = ENCODE_FUNCTIONS[args.encode_type]

    kwargs = {}
    if args.encode_type == 'hash':
        kwargs['n_components'] = args.hash_n_components
    elif args.encode_type == 'frequency':
        # frequency 不需要额外参数
        pass
    elif args.encode_type == 'target':
        kwargs['target_column'] = args.target_column
        kwargs['smoothing'] = args.target_smoothing
    elif args.encode_type == 'one_hot':
        kwargs['drop_first'] = args.oh_drop_first.lower() == 'true'
        kwargs['max_categories'] = args.oh_max_categories

    result = encode_func(df, columns, **kwargs)

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    result.to_csv(args.output_file, index=False)
    logger.info(f"编码完成，结果保存到: {args.output_file}")
    logger.info(f"编码后数据: {len(result)} 行, {len(result.columns)} 列")


def handle_outlier_detection(args):
    """异常值检测：Z-Score / IQR / Isolation Forest / Percentile"""
    import numpy as np
    import pandas as pd
    from outlier_detection import DETECT_FUNCTIONS

    logger.info("开始异常值检测")
    logger.info(f"  输入文件: {args.input_file}")
    logger.info(f"  检测方法: {args.od_method}")

    if not os.path.exists(args.input_file):
        logger.error(f"文件不存在: {args.input_file}")
        sys.exit(1)

    df = pd.read_csv(args.input_file)
    logger.info(f"  原始数据: {len(df)} 行, {len(df.columns)} 列")

    # 解析要检测的列
    if args.od_columns:
        columns = [c.strip() for c in args.od_columns.split(",")]
    else:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        logger.info(f"  未指定列，自动选择数值列: {columns}")

    if not columns:
        logger.error("没有可检测的列")
        sys.exit(1)

    detect_func = DETECT_FUNCTIONS[args.od_method]

    kwargs = {}
    if args.od_method == 'zscore':
        kwargs['threshold'] = args.od_zscore_threshold
    elif args.od_method == 'iqr':
        kwargs['multiplier'] = args.od_iqr_multiplier
    elif args.od_method == 'isolation_forest':
        kwargs['contamination'] = args.od_contamination
        kwargs['random_seed'] = args.random_seed
    elif args.od_method == 'percentile':
        kwargs['lower_pct'] = args.od_lower_pct
        kwargs['upper_pct'] = args.od_upper_pct

    result = detect_func(df, columns, **kwargs)

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    result.to_csv(args.output_file, index=False)
    logger.info(f"异常值检测完成，结果保存到: {args.output_file}")
    logger.info(f"检测后数据: {len(result)} 行, {len(result.columns)} 列")


def handle_calculate_metric(args):
    """统计量计算：基础统计/分布统计/缺失值统计/全量统计"""
    import numpy as np
    import pandas as pd
    from calculate_metric import METRIC_FUNCTIONS

    logger.info("开始统计量计算")
    logger.info(f"  输入文件: {args.input_file}")
    logger.info(f"  统计类型: {args.metric_type}")

    if not os.path.exists(args.input_file):
        logger.error(f"文件不存在: {args.input_file}")
        sys.exit(1)

    df = pd.read_csv(args.input_file)
    logger.info(f"  原始数据: {len(df)} 行, {len(df.columns)} 列")

    # 解析要统计的列
    if args.metric_columns:
        columns = [c.strip() for c in args.metric_columns.split(",")]
    else:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        logger.info(f"  未指定列，自动选择数值列: {columns}")

    if not columns:
        logger.error("没有可统计的列")
        sys.exit(1)

    metric_func = METRIC_FUNCTIONS[args.metric_type]
    result = metric_func(df, columns)

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    result.to_csv(args.output_file)
    logger.info(f"统计量计算完成，结果保存到: {args.output_file}")
    logger.info(f"统计结果: {len(result)} 项指标, {len(result.columns)} 列")


def handle_drop_stablize(args):
    """去除值过于单一的变量：缺失率/主导值/唯一值/方差"""
    import pandas as pd
    from drop_stablize import STABLIZE_FUNCTIONS

    logger.info("开始去除值过于单一的变量")
    logger.info(f"  输入文件: {args.input_file}")
    logger.info(f"  检测策略: {args.stablize_method}")

    if not os.path.exists(args.input_file):
        logger.error(f"文件不存在: {args.input_file}")
        sys.exit(1)

    df = pd.read_csv(args.input_file)
    logger.info(f"  原始数据: {len(df)} 行, {len(df.columns)} 列")

    # 解析列
    columns = None
    if args.stablize_columns:
        columns = [c.strip() for c in args.stablize_columns.split(",")]

    stablize_func = STABLIZE_FUNCTIONS[args.stablize_method]

    if args.stablize_method == 'all':
        result = stablize_func(df, columns=columns,
                               missing_threshold=args.stablize_missing_threshold,
                               dominant_threshold=args.stablize_dominant_threshold,
                               min_unique=args.stablize_min_unique,
                               variance_threshold=args.stablize_variance_threshold)
        # drop_stablize returns just the DataFrame
    elif args.stablize_method == 'missing_rate':
        result, _ = stablize_func(df, columns=columns, threshold=args.stablize_missing_threshold)
    elif args.stablize_method == 'dominant':
        result, _ = stablize_func(df, columns=columns, threshold=args.stablize_dominant_threshold)
    elif args.stablize_method == 'unique':
        result, _ = stablize_func(df, columns=columns, min_unique=args.stablize_min_unique)
    elif args.stablize_method == 'variance':
        result, _ = stablize_func(df, columns=columns, threshold=args.stablize_variance_threshold)

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    result.to_csv(args.output_file, index=False)
    logger.info(f"去稳定完成，结果保存到: {args.output_file}")
    logger.info(f"处理后数据: {len(result)} 行, {len(result.columns)} 列")


def handle_drop_high_missing(args):
    """删除缺失率过高的值：按列删 / 按行删 / 综合"""
    import pandas as pd
    from drop_high_missing import MISSING_FUNCTIONS, drop_columns_by_missing, drop_rows_by_missing

    logger.info("开始删除缺失率过高的值")
    logger.info(f"  输入文件: {args.input_file}")
    logger.info(f"  删除维度: {args.drop_axis}")

    if not os.path.exists(args.input_file):
        logger.error(f"文件不存在: {args.input_file}")
        sys.exit(1)

    df = pd.read_csv(args.input_file)
    logger.info(f"  原始数据: {len(df)} 行, {len(df.columns)} 列")

    columns = None
    if args.drop_missing_columns:
        columns = [c.strip() for c in args.drop_missing_columns.split(",")]

    if args.drop_axis in ('column', 'both'):
        result, _ = drop_columns_by_missing(df, args.col_missing_threshold, columns)
        if args.drop_axis == 'row':
            result = df  # 仅删行时保持原 df
    else:
        result = df

    if args.drop_axis in ('row', 'both'):
        result, _ = drop_rows_by_missing(result, args.row_missing_threshold, columns)

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    result.to_csv(args.output_file, index=False)
    logger.info(f"缺失删除完成，结果保存到: {args.output_file}")
    logger.info(f"处理后数据: {len(result)} 行, {len(result.columns)} 列")


def handle_fill_missing(args):
    """填充缺失值：均值/中位数/众数/常量/前后向/插值"""
    import pandas as pd
    from fill_missing import FILL_FUNCTIONS

    logger.info("开始填充缺失值")
    logger.info(f"  输入文件: {args.input_file}")
    logger.info(f"  填充方法: {args.fill_method}")

    if not os.path.exists(args.input_file):
        logger.error(f"文件不存在: {args.input_file}")
        sys.exit(1)

    df = pd.read_csv(args.input_file)
    total_missing = df.isna().sum().sum()
    logger.info(f"  原始数据: {len(df)} 行, {len(df.columns)} 列, {total_missing} 个缺失值")

    columns = None
    if args.fill_columns:
        columns = [c.strip() for c in args.fill_columns.split(",")]

    fill_func = FILL_FUNCTIONS[args.fill_method]

    if args.fill_method == 'constant':
        result = fill_func(df, columns if columns else df.columns.tolist(),
                          value=args.fill_constant_value)
    else:
        result = fill_func(df, columns if columns else df.columns.tolist())

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    result.to_csv(args.output_file, index=False)
    logger.info(f"填充完成，结果保存到: {args.output_file}")
    logger.info(f"填充后数据: {len(result)} 行, {len(result.columns)} 列")


def handle_discretization(args):
    """数据离散化：等宽/等频/KMeans聚类/自定义切点"""
    import numpy as np
    import pandas as pd
    from discretization import DISCRETIZE_FUNCTIONS

    logger.info("开始数据离散化")
    logger.info(f"  输入文件: {args.input_file}")
    logger.info(f"  离散化方法: {args.disc_method}")

    if not os.path.exists(args.input_file):
        logger.error(f"文件不存在: {args.input_file}")
        sys.exit(1)

    df = pd.read_csv(args.input_file)
    logger.info(f"  原始数据: {len(df)} 行, {len(df.columns)} 列")

    if args.disc_columns:
        columns = [c.strip() for c in args.disc_columns.split(",")]
    else:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        logger.info(f"  未指定列，自动选择数值列: {columns}")

    if not columns:
        logger.error("没有可离散化的列")
        sys.exit(1)

    disc_func = DISCRETIZE_FUNCTIONS[args.disc_method]

    if args.disc_method == 'custom':
        result = disc_func(df, columns, args.disc_cut_points)
    else:
        kwargs = {'n_bins': args.disc_n_bins}
        if args.disc_method == 'kmeans':
            kwargs['random_seed'] = args.random_seed
        result = disc_func(df, columns, **kwargs)

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    result.to_csv(args.output_file, index=False)
    logger.info(f"离散化完成，结果保存到: {args.output_file}")
    logger.info(f"处理后数据: {len(result)} 行, {len(result.columns)} 列")


def handle_standardize_normalize(args):
    """标准化/归一化：Z-Score/MinMax/MaxAbs/Robust/L2"""
    import numpy as np
    import pandas as pd
    from standardize_normalize import NORMALIZE_FUNCTIONS

    logger.info("开始标准化/归一化")
    logger.info(f"  输入文件: {args.input_file}")
    logger.info(f"  归一化方法: {args.norm_method}")

    if not os.path.exists(args.input_file):
        logger.error(f"文件不存在: {args.input_file}")
        sys.exit(1)

    df = pd.read_csv(args.input_file)
    logger.info(f"  原始数据: {len(df)} 行, {len(df.columns)} 列")

    if args.norm_columns:
        columns = [c.strip() for c in args.norm_columns.split(",")]
    else:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        logger.info(f"  未指定列，自动选择数值列: {columns}")

    if not columns:
        logger.error("没有可处理的列")
        sys.exit(1)

    norm_func = NORMALIZE_FUNCTIONS[args.norm_method]
    if args.norm_method == 'minmax':
        result = norm_func(df, columns, feature_range=(args.norm_min, args.norm_max))
    else:
        result = norm_func(df, columns)

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    result.to_csv(args.output_file, index=False)
    logger.info(f"归一化完成，结果保存到: {args.output_file}")
    logger.info(f"处理后数据: {len(result)} 行, {len(result.columns)} 列")


def handle_index_process(args):
    """索引处理：增加索引/设置索引/索引转列/按索引重命名/重排列"""
    import pandas as pd
    from index_process import INDEX_FUNCTIONS, add_index_column, set_column_as_index, reset_index_to_column, rename_columns_by_index, reorder_columns

    logger.info("开始索引处理")
    logger.info(f"  输入文件: {args.input_file}")
    logger.info(f"  操作类型: {args.index_op}")

    if not os.path.exists(args.input_file):
        logger.error(f"文件不存在: {args.input_file}")
        sys.exit(1)

    df = pd.read_csv(args.input_file)
    logger.info(f"  原始数据: {len(df)} 行, {len(df.columns)} 列")

    if args.index_op == 'add_index':
        result = add_index_column(df, args.index_name, args.index_start, args.index_step)
    elif args.index_op == 'set_index':
        result = set_column_as_index(df, args.index_column)
    elif args.index_op == 'reset_index':
        result = reset_index_to_column(df, args.index_name)
    elif args.index_op == 'rename_by_index':
        result = rename_columns_by_index(df, args.index_name_map, args.index_inplace.lower() == 'true')
    elif args.index_op == 'reorder':
        result = reorder_columns(df, args.index_column_order)

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    result.to_csv(args.output_file, index=args.index_op != 'set_index' or args.index_keep_index.lower() == 'true')
    logger.info(f"索引处理完成，结果保存到: {args.output_file}")
    logger.info(f"处理后数据: {len(result)} 行, {len(result.columns)} 列")


def handle_sort(args):
    """排序：单列/多列排序，升序/降序"""
    import pandas as pd
    from sort import sort_data, sort_by_multiple

    logger.info("开始排序")
    logger.info(f"  输入文件: {args.input_file}")

    if not os.path.exists(args.input_file):
        logger.error(f"文件不存在: {args.input_file}")
        sys.exit(1)

    df = pd.read_csv(args.input_file)
    logger.info(f"  原始数据: {len(df)} 行, {len(df.columns)} 列")

    if args.sort_orders:
        # 多列分别升降序
        result = sort_by_multiple(df, args.sort_columns, args.sort_orders)
    else:
        # 单一升降序
        columns = None
        if args.sort_columns:
            columns = [c.strip() for c in args.sort_columns.split(",")]
        result = sort_data(df, columns, args.sort_ascending.lower() == 'true')

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    result.to_csv(args.output_file, index=False)
    logger.info(f"排序完成，结果保存到: {args.output_file}")


def handle_run_sql(args):
    """执行 SQL：将 CSV 加载为 SQLite 表并执行查询"""
    import pandas as pd
    from run_sql import execute_sql, execute_sql_multi_table

    logger.info("开始执行 SQL")
    logger.info(f"  输入文件: {args.input_file}")

    if not os.path.exists(args.input_file):
        logger.error(f"文件不存在: {args.input_file}")
        sys.exit(1)

    if not args.sql_query or not args.sql_query.strip():
        logger.error("SQL 查询语句不能为空")
        sys.exit(1)

    # 检查是否有多表输入
    if args.sql_extra_files:
        file_map = {args.sql_table_name: args.input_file}
        for mapping in args.sql_extra_files.split(","):
            mapping = mapping.strip()
            if not mapping:
                continue
            parts = mapping.split(":")
            if len(parts) == 2:
                file_map[parts[0].strip()] = parts[1].strip()
            else:
                logger.warning(f"格式错误: {mapping}，应为 '表名:文件路径'")
        result = execute_sql_multi_table(file_map, args.sql_query)
    else:
        result = execute_sql(args.input_file, args.sql_query, args.sql_table_name)

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    result.to_csv(args.output_file, index=False)
    logger.info(f"SQL 执行完成，结果保存到: {args.output_file}")
    logger.info(f"查询结果: {len(result)} 行, {len(result.columns)} 列")


def handle_hadamard_multiply(args):
    """Hadamard 乘积：两个向量/矩阵逐元素相乘"""
    import pandas as pd
    import numpy as np
    from hadamard_multiply import (
        hadamard_multiply_vectors, hadamard_multiply_dataframes, load_vector
    )

    logger.info("开始 Hadamard 乘积")
    logger.info(f"  输入文件A: {args.input_file_a}")
    logger.info(f"  输入文件B: {args.input_file_b}")
    logger.info(f"  输入类型: {args.hadamard_type}")

    if args.hadamard_type == 'vector':
        vec_a = load_vector(args.input_file_a)
        vec_b = load_vector(args.input_file_b)
        logger.info(f"  向量A长度: {len(vec_a)}, 向量B长度: {len(vec_b)}")

        if len(vec_a) != len(vec_b):
            logger.error(f"向量长度不一致: {len(vec_a)} vs {len(vec_b)}")
            sys.exit(1)

        result_arr = hadamard_multiply_vectors(vec_a, vec_b)
        result = pd.DataFrame({'hadamard_result': result_arr})
    else:
        df_a = pd.read_csv(args.input_file_a)
        df_b = pd.read_csv(args.input_file_b)
        logger.info(f"  DataFrameA: {df_a.shape}, DataFrameB: {df_b.shape}")
        result = hadamard_multiply_dataframes(df_a, df_b)

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    result.to_csv(args.output_file, index=False)
    logger.info(f"Hadamard 乘积完成，结果保存到: {args.output_file}")
    logger.info(f"结果: {len(result)} 行, {len(result.columns)} 列")


def handle_feature_combine(args):
    """特征组合：四则运算/多项式/自定义表达式/列对组合"""
    import numpy as np
    import pandas as pd
    from feature_combine import COMBINE_FUNCTIONS, arithmetic_combine, polynomial_combine, custom_combine, add_subtract_all_pairs

    logger.info("开始特征组合")
    logger.info(f"  输入文件: {args.input_file}")
    logger.info(f"  组合类型: {args.combine_type}")

    if not os.path.exists(args.input_file):
        logger.error(f"文件不存在: {args.input_file}")
        sys.exit(1)

    df = pd.read_csv(args.input_file)
    logger.info(f"  原始数据: {len(df)} 行, {len(df.columns)} 列")

    if args.combine_columns:
        columns = [c.strip() for c in args.combine_columns.split(",")]
    else:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        logger.info(f"  未指定列，自动选择数值列: {columns}")

    if args.combine_type == 'arithmetic':
        if len(columns) < 2:
            logger.error("四则运算至少需要 2 列")
            sys.exit(1)
        result = arithmetic_combine(df, columns[0], columns[1],
                                     args.combine_arith_op, args.combine_new_col)
    elif args.combine_type == 'polynomial':
        poly_df = polynomial_combine(df, columns, args.combine_poly_degree,
                                      args.combine_poly_interaction.lower() == 'true')
        result = pd.concat([df, poly_df], axis=1)
    elif args.combine_type == 'custom':
        result = custom_combine(df, columns, args.combine_expr, args.combine_new_col)
    elif args.combine_type == 'all_pairs':
        result = add_subtract_all_pairs(df, columns)

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    result.to_csv(args.output_file, index=False)
    logger.info(f"特征组合完成，结果保存到: {args.output_file}")
    logger.info(f"组合后数据: {len(result)} 行, {len(result.columns)} 列")


def handle_dimension_reduction(args):
    """降维：PCA / 卡方特征选择"""
    import numpy as np
    import pandas as pd
    from dimension_reduction import REDUCTION_FUNCTIONS, pca_reduce, chi2_select

    logger.info("开始降维")
    logger.info(f"  输入文件: {args.input_file}")
    logger.info(f"  降维方法: {args.dr_method}")

    if not os.path.exists(args.input_file):
        logger.error(f"文件不存在: {args.input_file}")
        sys.exit(1)

    df = pd.read_csv(args.input_file)
    logger.info(f"  原始数据: {len(df)} 行, {len(df.columns)} 列")

    if args.dr_columns:
        columns = [c.strip() for c in args.dr_columns.split(",")]
    else:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if args.dr_method == 'pca':
        n_components = args.dr_n_components
        if args.dr_variance_ratio:
            n_components = min(args.dr_variance_ratio, 1.0)
        result = pca_reduce(df, columns, n_components, args.dr_keep_original.lower() == 'true')
    elif args.dr_method == 'chi2':
        result, scores = chi2_select(df, columns,
                                     args.dr_target_column,
                                     args.dr_n_components)
        # 保存特征得分
        scores_dir = os.path.dirname(args.output_file)
        scores_path = os.path.join(scores_dir or '.',
                                    f"chi2_scores_{os.path.basename(args.output_file)}")
        scores.to_csv(scores_path, index=False)
        logger.info(f"卡方特征得分保存到: {scores_path}")

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    result.to_csv(args.output_file, index=False)
    logger.info(f"降维完成，结果保存到: {args.output_file}")
    logger.info(f"降维后数据: {len(result)} 行, {len(result.columns)} 列")


def handle_feature_importance(args):
    """特征重要性：随机森林/方差/互信息/相关性/IV值"""
    import numpy as np
    import pandas as pd
    from feature_importance import IMPORTANCE_FUNCTIONS

    logger.info("开始特征重要性计算")
    logger.info(f"  输入文件: {args.input_file}")
    logger.info(f"  重要性方法: {args.fi_method}")

    if not os.path.exists(args.input_file):
        logger.error(f"文件不存在: {args.input_file}")
        sys.exit(1)

    df = pd.read_csv(args.input_file)
    logger.info(f"  原始数据: {len(df)} 行, {len(df.columns)} 列")

    if args.fi_columns:
        columns = [c.strip() for c in args.fi_columns.split(",")]
    else:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    fi_func = IMPORTANCE_FUNCTIONS[args.fi_method]

    if args.fi_method == 'variance':
        result = fi_func(df, columns)
    else:
        if not args.fi_target_column:
            logger.error(f"方法 '{args.fi_method}' 需要指定目标列 (--fi_target_column)")
            sys.exit(1)
        result = fi_func(df, columns, args.fi_target_column)

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    result.to_csv(args.output_file, index=False)
    logger.info(f"特征重要性计算完成，结果保存到: {args.output_file}")
    logger.info(f"共 {len(result)} 个特征的得分")


def handle_data_split(args):
    """数据拆分：列内拆分/列间拆分/行间拆分/SVD分解"""
    import pandas as pd
    from data_split import SPLIT_FUNCTIONS

    logger.info("开始数据拆分")
    logger.info(f"  输入文件: {args.input_file}")
    logger.info(f"  拆分类型: {args.split_type}")

    if not os.path.exists(args.input_file):
        logger.error(f"文件不存在: {args.input_file}")
        sys.exit(1)

    df = pd.read_csv(args.input_file)
    logger.info(f"  原始数据: {len(df)} 行, {len(df.columns)} 列")

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if args.split_type == 'column_split':
        result = split_column(df, args.split_column, args.split_delimiter,
                              args.split_new_columns)
        result.to_csv(args.output_file, index=False)
    elif args.split_type == 'column_group':
        result = split_columns_by_index(df, args.split_group_indices)
        result.to_csv(args.output_file, index=False)
    elif args.split_type == 'row_split':
        results = split_rows(df, args.split_ratios, args.random_seed,
                             args.split_stratify_column)
        base_name = os.path.splitext(args.output_file)[0]
        for name, sub_df in results:
            fname = f"{base_name}_{name}.csv"
            sub_df.to_csv(fname, index=False)
            logger.info(f"  保存 {name}: {len(sub_df)} 行 -> {fname}")
    elif args.split_type == 'svd':
        if args.split_columns:
            columns = [c.strip() for c in args.split_columns.split(",")]
        else:
            columns = None
        result = svd_decompose(df, columns, args.split_n_components)
        result.to_csv(args.output_file, index=False)

    logger.info(f"数据拆分完成")


HANDLERS = {
    "calculate_correlation": handle_calculate_correlation,
    "calculate_metric": handle_calculate_metric,
    "data_split": handle_data_split,
    "data_transform": handle_data_transform,
    "dimension_reduction": handle_dimension_reduction,
    "discretization": handle_discretization,
    "drop_duplicates": handle_drop_duplicates,
    "drop_high_missing": handle_drop_high_missing,
    "drop_stablize": handle_drop_stablize,
    "feature_combine": handle_feature_combine,
    "feature_importance": handle_feature_importance,
    "fill_missing": handle_fill_missing,
    "hadamard_multiply": handle_hadamard_multiply,
    "index_process": handle_index_process,
    "objective_process": handle_objective_process,
    "outlier_detection": handle_outlier_detection,
    "run_sql": handle_run_sql,
    "sample": handle_sample,
    "sort": handle_sort,
    "standardize_normalize": handle_standardize_normalize,
    "union_join": handle_union_join,
}


def parse_args():
    parser = argparse.ArgumentParser(description='特征处理统一入口')
    parser.add_argument('--process_type', type=str, required=True,
                        choices=['calculate_correlation', 'calculate_metric', 'data_split',
                                 'data_transform', 'dimension_reduction', 'discretization',
                                 'drop_duplicates', 'drop_high_missing', 'drop_stablize',
                                 'feature_combine', 'feature_importance', 'fill_missing',
                                 'hadamard_multiply', 'index_process', 'objective_process',
                                 'outlier_detection', 'run_sql', 'sample', 'sort',
                                 'standardize_normalize', 'union_join'],
                        help='特征处理类型')
    # calculate_correlation 专用参数
    parser.add_argument('--input_file_a', type=str, help='特征向量A文件路径')
    parser.add_argument('--input_file_b', type=str, help='特征向量B文件路径')
    parser.add_argument('--output_file', type=str, help='结果输出路径')
    # sample 专用参数
    parser.add_argument('--input_file', type=str, help='输入数据路径')
    parser.add_argument('--sample_type', type=str,
                        choices=['random', 'stratified', 'oversampling', 'undersampling'],
                        help='采样类型')
    parser.add_argument('--sample_rate', type=float, default=0.5, help='采样比例/数量')
    parser.add_argument('--stratify_column', type=str, default='', help='分层/标签列')
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子')
    parser.add_argument('--replace', type=str, nargs='?', const='true', default='false', help='是否放回采样')
    # union_join 专用参数
    parser.add_argument('--merge_type', type=str,
                        choices=['union', 'join'],
                        help='合并类型：union(行拼接) / join(列关联)')
    parser.add_argument('--join_on', type=str, default='',
                        help='join 关联键列名，多列用逗号分隔')
    parser.add_argument('--join_how', type=str, default='inner',
                        choices=['inner', 'left', 'right', 'outer'],
                        help='join 方式，默认 inner')
    # drop_duplicates 专用参数
    parser.add_argument('--dedup_method', type=str, default='exact',
                        choices=['exact', 'fuzzy'],
                        help='去重方法：exact(精确) / fuzzy(模糊)')
    parser.add_argument('--dedup_subset', type=str, default='',
                        help='去重依据列名，多列用逗号分隔，默认全部列')
    parser.add_argument('--dedup_keep', type=str, default='first',
                        choices=['first', 'last', 'false'],
                        help='保留策略：first(首个)/last(末个)/false(全删)')
    parser.add_argument('--dedup_threshold', type=float, default=0.95,
                        help='模糊去重相似度阈值(0~1)，默认 0.95')
    # data_transform 专用参数
    parser.add_argument('--transform_type', type=str,
                        choices=['boxcox', 'binarize', 'type_convert', 'dct',
                                 'apply_function', 'moving_average', 'polynomial_expand'],
                        help='数据变换类型')
    parser.add_argument('--transform_columns', type=str, default='',
                        help='要变换的列名，多列用逗号分隔，默认所有数值列')
    parser.add_argument('--boxcox_lambda', type=float, default=None,
                        help='Box-Cox 变换的 lambda 值，默认自动搜索')
    parser.add_argument('--binarize_threshold', type=float, default=0.0,
                        help='二值化阈值，默认 0.0')
    parser.add_argument('--target_type', type=str, default='float64',
                        choices=['float64', 'int64', 'str', 'category'],
                        help='类型转换目标类型，默认 float64')
    parser.add_argument('--dct_n_components', type=int, default=None,
                        help='DCT 保留分量数，默认保留全部')
    parser.add_argument('--func_expr', type=str, default='x ** 2',
                        help='函数转换表达式，如 "x**2"、"np.log(x+1)"，默认 x**2')
    parser.add_argument('--ma_window', type=int, default=3,
                        help='移动平均窗口大小，默认 3')
    parser.add_argument('--ma_center', type=str, nargs='?', const='true', default='true',
                        help='移动平均是否居中窗口，默认 true')
    parser.add_argument('--poly_degree', type=int, default=2,
                        help='多项式展开阶数，默认 2')
    parser.add_argument('--poly_interaction_only', type=str, nargs='?', const='true', default='false',
                        help='多项式展开是否仅交互项，默认 false')
    # objective_process 专用参数
    parser.add_argument('--encode_type', type=str,
                        choices=['hash', 'frequency', 'target', 'one_hot'],
                        help='编码类型')
    parser.add_argument('--encode_columns', type=str, default='',
                        help='要编码的列名，多列用逗号分隔，默认所有非数值列')
    parser.add_argument('--hash_n_components', type=int, default=8,
                        help='Hash 编码输出维度，默认 8')
    parser.add_argument('--target_column', type=str, default='',
                        help='目标编码的目标变量列名')
    parser.add_argument('--target_smoothing', type=float, default=1.0,
                        help='目标编码平滑系数，默认 1.0')
    parser.add_argument('--oh_drop_first', type=str, nargs='?', const='true', default='false',
                        help='One-Hot 是否丢弃第一个类别，默认 false')
    parser.add_argument('--oh_max_categories', type=int, default=50,
                        help='One-Hot 每列最多保留类别数，默认 50')
    # outlier_detection 专用参数
    parser.add_argument('--od_method', type=str,
                        choices=['zscore', 'iqr', 'isolation_forest', 'percentile'],
                        help='异常值检测方法')
    parser.add_argument('--od_columns', type=str, default='',
                        help='要检测的列名，多列用逗号分隔，默认所有数值列')
    parser.add_argument('--od_zscore_threshold', type=float, default=3.0,
                        help='Z-Score 阈值，默认 3.0')
    parser.add_argument('--od_iqr_multiplier', type=float, default=1.5,
                        help='IQR 倍数，默认 1.5')
    parser.add_argument('--od_contamination', type=float, default=0.1,
                        help='IsolationForest 预期异常比例，默认 0.1')
    parser.add_argument('--od_lower_pct', type=float, default=1.0,
                        help='Percentile 下界百分位，默认 1.0')
    parser.add_argument('--od_upper_pct', type=float, default=99.0,
                        help='Percentile 上界百分位，默认 99.0')
    # calculate_metric 专用参数
    parser.add_argument('--metric_type', type=str,
                        choices=['basic', 'distribution', 'missing', 'full'],
                        help='统计量类型：basic(基础)/distribution(分布)/missing(缺失)/full(全部)')
    parser.add_argument('--metric_columns', type=str, default='',
                        help='要统计的列名，多列用逗号分隔，默认所有数值列')
    # drop_stablize 专用参数
    parser.add_argument('--stablize_method', type=str,
                        choices=['missing_rate', 'dominant', 'unique', 'variance', 'all'],
                        default='all',
                        help='去稳定策略：missing_rate/dominant/unique/variance/all(综合)')
    parser.add_argument('--stablize_columns', type=str, default='',
                        help='要检查的列名，多列用逗号分隔，默认所有列')
    parser.add_argument('--stablize_missing_threshold', type=float, default=0.5,
                        help='缺失率阈值，超过则删除，默认 0.5')
    parser.add_argument('--stablize_dominant_threshold', type=float, default=0.95,
                        help='主导值比例阈值，超过则删除，默认 0.95')
    parser.add_argument('--stablize_min_unique', type=int, default=2,
                        help='最少唯一值数，低于此值则删除，默认 2')
    parser.add_argument('--stablize_variance_threshold', type=float, default=1e-10,
                        help='方差阈值，低于此值则删除，默认 1e-10')
    # drop_high_missing 专用参数
    parser.add_argument('--drop_axis', type=str, default='both',
                        choices=['column', 'row', 'both'],
                        help='缺失删除维度：column(删列)/row(删行)/both(先列后行)')
    parser.add_argument('--drop_missing_columns', type=str, default='',
                        help='参与计算的列名，多列用逗号分隔，默认所有列')
    parser.add_argument('--col_missing_threshold', type=float, default=0.5,
                        help='列缺失率阈值，超过即删该列，默认 0.5')
    parser.add_argument('--row_missing_threshold', type=float, default=0.5,
                        help='行缺失率阈值，超过即删该行，默认 0.5')
    # fill_missing 专用参数
    parser.add_argument('--fill_method', type=str, default='mean',
                        choices=['mean', 'median', 'mode', 'constant',
                                 'ffill', 'bfill', 'interpolate'],
                        help='填充方法：mean/median/mode/constant/ffill/bfill/interpolate')
    parser.add_argument('--fill_columns', type=str, default='',
                        help='要填充的列名，多列用逗号分隔，默认所有列')
    parser.add_argument('--fill_constant_value', type=str, default='0',
                        help='常量填充值，默认 0')
    # discretization 专用参数
    parser.add_argument('--disc_method', type=str, default='equal_width',
                        choices=['equal_width', 'equal_freq', 'kmeans', 'custom'],
                        help='离散化方法')
    parser.add_argument('--disc_columns', type=str, default='',
                        help='要离散化的列名，多列用逗号分隔，默认所有数值列')
    parser.add_argument('--disc_n_bins', type=int, default=5,
                        help='分箱数，默认 5')
    parser.add_argument('--disc_cut_points', type=str, default='',
                        help='自定义切点，如 "0,10,50,100"；仅 disc_method=custom 时生效')
    # standardize_normalize 专用参数
    parser.add_argument('--norm_method', type=str, default='zscore',
                        choices=['zscore', 'minmax', 'maxabs', 'robust', 'l2'],
                        help='归一化方法')
    parser.add_argument('--norm_columns', type=str, default='',
                        help='要处理的列名，多列用逗号分隔，默认所有数值列')
    parser.add_argument('--norm_min', type=float, default=0.0,
                        help='MinMax 目标下界，默认 0.0')
    parser.add_argument('--norm_max', type=float, default=1.0,
                        help='MinMax 目标上界，默认 1.0')
    # index_process 专用参数
    parser.add_argument('--index_op', type=str, default='add_index',
                        choices=['add_index', 'set_index', 'reset_index',
                                 'rename_by_index', 'reorder'],
                        help='索引操作类型')
    parser.add_argument('--index_name', type=str, default='index',
                        help='索引列名（add_index/reset_index 时生效），默认 index')
    parser.add_argument('--index_start', type=int, default=0,
                        help='增加索引的起始值，默认 0')
    parser.add_argument('--index_step', type=int, default=1,
                        help='增加索引的步长，默认 1')
    parser.add_argument('--index_column', type=str, default='',
                        help='设为索引的列名（set_index 时生效）')
    parser.add_argument('--index_name_map', type=str, default='',
                        help='索引->新列名映射，如 "0:new_a,2:new_c"（rename_by_index 时生效）')
    parser.add_argument('--index_inplace', type=str, nargs='?', const='true', default='false',
                        help='按索引重命名是否就地替换（rename_by_index 时生效）')
    parser.add_argument('--index_column_order', type=str, default='',
                        help='列顺序，如 "col2,col1"（reorder 时生效）')
    parser.add_argument('--index_keep_index', type=str, nargs='?', const='true', default='false',
                        help='输出时是否保留索引列（set_index 时生效）')
    # sort 专用参数
    parser.add_argument('--sort_columns', type=str, default='',
                        help='排序列名，多列用逗号分隔，默认所有列')
    parser.add_argument('--sort_ascending', type=str, nargs='?', const='true', default='true',
                        help='是否升序，默认 true')
    parser.add_argument('--sort_orders', type=str, default='',
                        help='多列分别升降序，如 "asc,desc"（与 sort_columns 一一对应，留空则统一用 ascending）')
    # run_sql 专用参数
    parser.add_argument('--sql_query', type=str, default='',
                        help='SQL 查询语句，如 "SELECT * FROM data WHERE col1 > 0"')
    parser.add_argument('--sql_table_name', type=str, default='data',
                        help='CSV 数据加载到的表名，默认 data')
    parser.add_argument('--sql_extra_files', type=str, default='',
                        help='多表映射，如 "table_a:/path/a.csv,table_b:/path/b.csv"')
    # hadamard_multiply 专用参数
    parser.add_argument('--hadamard_type', type=str, default='vector',
                        choices=['vector', 'dataframe'],
                        help='Hadamard 乘积类型：vector(向量) / dataframe(矩阵)')
    # feature_combine 专用参数
    parser.add_argument('--combine_type', type=str, default='arithmetic',
                        choices=['arithmetic', 'polynomial', 'custom', 'all_pairs'],
                        help='特征组合类型')
    parser.add_argument('--combine_columns', type=str, default='',
                        help='参与组合的列名，多列用逗号分隔')
    parser.add_argument('--combine_arith_op', type=str, default='add',
                        choices=['add', 'sub', 'mul', 'div'],
                        help='四则运算运算符（arithmetic 时生效）')
    parser.add_argument('--combine_new_col', type=str, default='',
                        help='新列名，默认自动生成（arithmetic/custom 时生效）')
    parser.add_argument('--combine_expr', type=str, default='x0 + x1',
                        help='表达式，如 "x0 + x1 * 2"，x0=第1列（custom 时生效）')
    parser.add_argument('--combine_poly_degree', type=int, default=2,
                        help='多项式阶数（polynomial 时生效），默认 2')
    parser.add_argument('--combine_poly_interaction', type=str, nargs='?', const='true', default='false',
                        help='是否仅生成交互项（polynomial 时生效），默认 false')
    # dimension_reduction 专用参数
    parser.add_argument('--dr_method', type=str, default='pca',
                        choices=['pca', 'chi2'],
                        help='降维方法')
    parser.add_argument('--dr_columns', type=str, default='',
                        help='参与降维的列名，多列用逗号分隔，默认所有数值列')
    parser.add_argument('--dr_n_components', type=int, default=2,
                        help='保留的维度数，默认 2')
    parser.add_argument('--dr_variance_ratio', type=float, default=0.0,
                        help='PCA 目标解释方差比例(0~1)，设置后覆盖 n_components')
    parser.add_argument('--dr_keep_original', type=str, nargs='?', const='true', default='false',
                        help='是否保留原始列（PCA 时生效），默认 false')
    parser.add_argument('--dr_target_column', type=str, default='',
                        help='目标列名（卡方选择时生效）')
    # feature_importance 专用参数
    parser.add_argument('--fi_method', type=str, default='random_forest',
                        choices=['random_forest', 'variance', 'mutual_info',
                                 'correlation', 'iv'],
                        help='特征重要性方法')
    parser.add_argument('--fi_columns', type=str, default='',
                        help='特征列名，多列逗号分隔，默认所有数值列')
    parser.add_argument('--fi_target_column', type=str, default='',
                        help='目标变量列名（variance 不需要，其余方法需要）')
    # data_split 专用参数
    parser.add_argument('--split_type', type=str, default='row_split',
                        choices=['column_split', 'column_group', 'row_split', 'svd'],
                        help='拆分类型')
    parser.add_argument('--split_column', type=str, default='',
                        help='要拆分的列名（column_split 时生效）')
    parser.add_argument('--split_delimiter', type=str, default=',',
                        help='分隔符（column_split 时生效），默认逗号')
    parser.add_argument('--split_new_columns', type=str, default='',
                        help='新列名逗号分隔（column_split 时生效），默认自动命名')
    parser.add_argument('--split_group_indices', type=str, default='',
                        help='分组索引（column_group 时生效），如 "0,1|2,3"')
    parser.add_argument('--split_ratios', type=str, default='0.7,0.15,0.15',
                        help='拆分比例逗号分隔（row_split 时生效），默认 0.7,0.15,0.15')
    parser.add_argument('--split_stratify_column', type=str, default='',
                        help='分层列名（row_split 时生效）')
    parser.add_argument('--split_columns', type=str, default='',
                        help='参与拆分的列名（svd 时生效），默认所有数值列')
    parser.add_argument('--split_n_components', type=int, default=2,
                        help='SVD 保留分量数，默认 2')
    return parser.parse_args()


def main():
    args = parse_args()
    handler = HANDLERS[args.process_type]
    handler(args)


if __name__ == '__main__':
    main()
