"""
数据过滤节点：自动识别文本字段，按语言/质量分数/长度过滤
可以独立运行，也可以作为 Pipeline 节点连接使用
"""
import os
import json
import argparse
import re


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
# 过滤逻辑
# ============================================================

# CJK 统一汉字区间
CJK_RANGES = [
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
]


def cjk_ratio(text):
    """计算文本中 CJK 字符的占比"""
    if not text:
        return 0.0
    cjk_count = 0
    for ch in text:
        code = ord(ch)
        if any(lo <= code <= hi for lo, hi in CJK_RANGES):
            cjk_count += 1
    return cjk_count / len(text)


def _is_cjk_char(ch):
    """判断单个字符是否为 CJK 字符（与 cjk_ratio 使用相同区间）"""
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in CJK_RANGES)


def quality_score(text):
    """对文本质量打分（0-100）。只检查特殊字符占比，长度过滤由清洗节点负责。"""
    if not text or not text.strip():
        return 0
    score = 100
    # 特殊字符占比惩罚：非字母数字、非空白、非 CJK 的字符视为特殊字符
    special_count = sum(1 for ch in text
                        if not (ch.isalnum() or ch == '_' or ch.isspace() or _is_cjk_char(ch)))
    special_ratio = special_count / max(len(text), 1)
    if special_ratio > 0.3:
        score -= int(special_ratio * 100)
    return max(0, score)


def filter_data(input_dir, output_dir, text_field='text',
                target_lang='zh', min_score=60):
    """
    数据过滤主函数：
    - 自动识别文本字段
    - 语言检测（CJK 字符占比）
    - 质量评分过滤（特殊字符占比）
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
        detected = auto_detect_text_field(records)
        use_field = text_field if (text_field != 'text' or not detected) else detected
        if use_field == 'text' and detected and detected != 'text':
            use_field = detected
        print(f'[{rel_path}] using field: "{use_field}"')

        total = kept = 0
        with open(output_path, 'w', encoding='utf-8') as fout:
            for record in records:
                total += 1
                text = extract_text_from_record(record, use_field)

                # 语言过滤
                if target_lang == 'zh' and cjk_ratio(text) < 0.3:
                    continue
                # 质量过滤（特殊字符占比）
                if quality_score(text) < min_score:
                    continue

                # 确保 text 字段存在
                if isinstance(record, dict) and 'text' not in record:
                    record = {**record, 'text': text}
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                kept += 1

        print(f'[{rel_path}] total={total}, kept={kept}, dropped={total - kept}')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser('nlp-filter-data')
    parser.add_argument('--input_dir', type=str, required=True, help='输入目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--text_field', type=str, default='text', help='文本字段名（留空则自动检测）')
    parser.add_argument('--target_lang', type=str, default='zh', help='目标语言代码，默认 zh')
    parser.add_argument('--min_score', type=int, default=60, help='质量及格线 (0-100)')
    args = parser.parse_args()

    exit(filter_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        text_field=args.text_field,
        target_lang=args.target_lang,
        min_score=int(args.min_score),
    ))
