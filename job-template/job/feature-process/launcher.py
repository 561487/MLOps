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
    sample_type_func = {
        'random': lambda: random_sample(df, args.sample_rate, args.random_seed, args.replace),
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


HANDLERS = {
    "calculate_correlation": handle_calculate_correlation,
    "sample": handle_sample,
}


def parse_args():
    parser = argparse.ArgumentParser(description='特征处理统一入口')
    parser.add_argument('--process_type', type=str, required=True,
                        choices=sorted(HANDLERS.keys()),
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
    parser.add_argument('--replace', type=bool, default=False, help='是否放回采样')
    return parser.parse_args()


def main():
    args = parse_args()
    handler = HANDLERS[args.process_type]
    handler(args)


if __name__ == '__main__':
    main()
