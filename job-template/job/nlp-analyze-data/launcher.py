"""
数据统计节点：自动识别文本字段，输出统计报告
可以独立运行，也可以作为 Pipeline 节点连接使用
"""
import os
import json
import argparse
import re
from collections import Counter


# ============================================================
# 公共工具函数（内联，每个节点自包含）
# ============================================================

def find_files(input_dir):
    """递归查找目录下所有 jsonl / json / parquet 文件"""
    files = []
    for root, dirs, filenames in os.walk(input_dir):
        for f in filenames:
            if f.endswith(('.jsonl', '.json', '.parquet')):
                files.append(os.path.join(root, f))
    return sorted(files)


def load_records(filepath):
    """自动识别格式并加载数据记录"""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.parquet':
        import pandas as pd
        df = pd.read_parquet(filepath)
        return df.to_dict('records')
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                data = json.load(f)
                return data if isinstance(data, list) else [data]
            records = []
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            return records


def _flatten_value(val):
    """将任意值展平为纯文本字符串（处理嵌套 JSON）"""
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return _flatten_value(parsed)
        except (json.JSONDecodeError, TypeError):
            return val

    if isinstance(val, list):
        parts = []
        for item in val:
            t = _flatten_value(item)
            if t:
                parts.append(t)
        return '\n'.join(parts)

    if isinstance(val, dict):
        parts = []
        for k, v in val.items():
            if k in ('content', 'text', 'value', 'message'):
                parts.append(_flatten_value(v))
        if not parts:
            parts = [_flatten_value(v) for v in val.values()]
        return '\n'.join(p for p in parts if p)

    if val is None:
        return ''
    return str(val)


def _auto_detect(record):
    """在单条记录中自动找最可能是文本的值"""
    best_text = ''
    best_len = 0
    for val in record.values():
        text = _flatten_value(val)
        if len(text) > best_len:
            best_text = text
            best_len = len(text)
    return best_text


def extract_text_from_record(record, text_field=None):
    """从一条记录中提取文本内容，支持自动检测"""
    if not isinstance(record, dict):
        return str(record)

    if text_field and text_field in record:
        val = record[text_field]
        return _flatten_value(val)

    if text_field and text_field not in record:
        print(f'  WARNING: field "{text_field}" not found, auto-detecting...')

    return _auto_detect(record)


def auto_detect_text_field(records, sample_size=50):
    """从一批记录中自动检测最可能是文本的字段名"""
    if not records or not isinstance(records[0], dict):
        return None

    sample = records[:sample_size]
    candidates = {}

    for col in sample[0].keys():
        lengths = []
        for r in sample:
            val = r.get(col, '')
            if isinstance(val, str):
                lengths.append(len(val))
            elif isinstance(val, (list, dict)):
                text = _flatten_value(val)
                lengths.append(len(text))
            else:
                lengths.append(len(str(val)) if val else 0)

        avg_len = sum(lengths) / len(lengths) if lengths else 0
        candidates[col] = avg_len

    if candidates:
        best = max(candidates, key=candidates.get)
        if candidates[best] > 10:
            print(f'  Auto-detected text field: "{best}" (avg length: {candidates[best]:.0f} chars)')
            return best
    return None


# ============================================================
# 统计分析逻辑
# ============================================================

def analyze_data(input_dir, output_dir, text_field='text'):
    """
    数据统计主函数：
    - 自动识别文本字段
    - 统计总记录数、总字符数、总 token 数
    - 计算平均值
    - 输出长度分布直方图
    - 按文件输出详细统计，汇总写入 stats.json
    """
    os.makedirs(output_dir, exist_ok=True)
    all_files = find_files(input_dir)

    if not all_files:
        print(f'ERROR: No data files found under {input_dir}')
        if os.path.exists(input_dir):
            print(f'Top-level contents: {os.listdir(input_dir)}')
        return 1

    stats = {
        'total_records': 0,
        'total_chars': 0,
        'total_tokens': 0,
        'length_dist': Counter(),
        'empty': 0,
        'per_file': [],
    }

    for input_path in all_files:
        rel_path = os.path.relpath(input_path, input_dir)
        fs = {
            'filename': rel_path,
            'records': 0,
            'chars': 0,
            'tokens': 0,
            'empty': 0,
        }

        records = load_records(input_path)
        if not records:
            continue

        # 自动检测文本字段
        detected = auto_detect_text_field(records)
        use_field = text_field if (text_field != 'text' or not detected) else detected
        if use_field == 'text' and detected and detected != 'text':
            use_field = detected
        print(f'[{rel_path}] using field: "{use_field}"')

        for r in records:
            fs['records'] += 1
            stats['total_records'] += 1
            text = extract_text_from_record(r, use_field)

            if not text.strip():
                fs['empty'] += 1
                stats['empty'] += 1
                continue

            fs['chars'] += len(text)
            stats['total_chars'] += len(text)

            tokens = len(re.findall(r'\w+', text))
            fs['tokens'] += tokens
            stats['total_tokens'] += tokens

            # 长度分布桶（每 100 字符）
            bucket_start = len(text) // 100 * 100
            bucket = f'{bucket_start}-{bucket_start + 99}'
            stats['length_dist'][bucket] += 1

        stats['per_file'].append(fs)
        print(f'[{rel_path}] records={fs["records"]}, chars={fs["chars"]}, tokens={fs["tokens"]}, empty={fs["empty"]}')

    # 汇总
    total = max(stats['total_records'], 1)
    summary = {
        'total_records': stats['total_records'],
        'total_chars': stats['total_chars'],
        'avg_chars': round(stats['total_chars'] / total, 2),
        'total_tokens': stats['total_tokens'],
        'avg_tokens': round(stats['total_tokens'] / total, 2),
        'empty': stats['empty'],
        'length_distribution': dict(stats['length_dist'].most_common(20)),
        'per_file': stats['per_file'],
    }

    output_path = os.path.join(output_dir, 'stats.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f'\nAnalyzed {stats["total_records"]} records from {len(all_files)} files → {output_path}')
    print(f'  Avg chars: {summary["avg_chars"]}, Avg tokens: {summary["avg_tokens"]}')
    print(f'  Empty: {summary["empty"]}, Length buckets: {len(summary["length_distribution"])}')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser('nlp-analyze-data')
    parser.add_argument('--input_dir', type=str, required=True, help='输入目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--text_field', type=str, default='text', help='文本字段名（留空则自动检测）')
    args = parser.parse_args()

    exit(analyze_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        text_field=args.text_field,
    ))
