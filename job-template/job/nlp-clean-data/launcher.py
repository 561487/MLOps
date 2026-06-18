"""
数据清洗节点：自动识别文本字段，按长度过滤、去空、去重
可以独立运行，也可以作为 Pipeline 节点连接使用
"""
import os
import json
import argparse


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
# 清洗逻辑
# ============================================================

def clean_data(input_dir, output_dir, text_field='text',
               min_length=0, max_length=0,
               drop_empty=True, drop_duplicates=True):
    """
    数据清洗主函数：
    - 自动识别文本字段
    - 去空值/空白行
    - 按长度范围过滤
    - 去重
    - 输出为标准 JSONL
    """
    os.makedirs(output_dir, exist_ok=True)
    all_files = find_files(input_dir)

    if not all_files:
        print(f'ERROR: No data files found under {input_dir}')
        if os.path.exists(input_dir):
            print(f'Top-level contents: {os.listdir(input_dir)}')
        return 1

    for input_path in all_files:
        rel_path = os.path.relpath(input_path, input_dir)
        ext = os.path.splitext(input_path)[1]
        output_path = os.path.join(output_dir, os.path.basename(input_path).replace(ext, '.jsonl'))

        records = load_records(input_path)
        if not records:
            print(f'[{rel_path}] no records')
            continue

        # 自动检测文本字段
        detected_field = auto_detect_text_field(records)
        use_field = text_field if (text_field != 'text' or not detected_field) else detected_field
        if use_field == 'text' and detected_field and detected_field != 'text':
            use_field = detected_field
        print(f'[{rel_path}] using field: "{use_field}"')

        seen = set()
        total = kept = 0

        with open(output_path, 'w', encoding='utf-8') as fout:
            for record in records:
                total += 1
                text = extract_text_from_record(record, use_field)

                if drop_empty and (not text or not text.strip()):
                    continue
                if min_length > 0 and len(text) < min_length:
                    continue
                if max_length > 0 and len(text) > max_length:
                    continue
                if drop_duplicates:
                    key = json.dumps(record, ensure_ascii=False, sort_keys=True) if isinstance(record, dict) else text
                    if key in seen:
                        continue
                    seen.add(key)

                # 写入时保留原始记录，确保 text 字段有值
                if isinstance(record, dict) and 'text' not in record:
                    record = {**record, 'text': text}
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                kept += 1

        print(f'[{rel_path}] total={total}, kept={kept}, dropped={total - kept}')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser('nlp-clean-data')
    parser.add_argument('--input_dir', type=str, required=True, help='输入目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--text_field', type=str, default='text', help='文本字段名（留空则自动检测）')
    parser.add_argument('--min_length', type=int, default=0, help='最小文本长度')
    parser.add_argument('--max_length', type=int, default=0, help='最大文本长度')
    parser.add_argument('--drop_empty', type=str, default='true', help='是否去除空值 (true/false)')
    parser.add_argument('--drop_duplicates', type=str, default='true', help='是否去重 (true/false)')
    args = parser.parse_args()

    exit(clean_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        text_field=args.text_field,
        min_length=int(args.min_length) if args.min_length else 0,
        max_length=int(args.max_length) if args.max_length else 0,
        drop_empty=args.drop_empty.lower() == 'true',
        drop_duplicates=args.drop_duplicates.lower() == 'true',
    ))
