#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解析 OpenCompass 评测结果，保存为结构化文件。

OpenCompass 输出目录结构大致如下（--work-dir 指定的目录）：
  {work_dir}/
    ├── summary/
    │   ├── {timestamp}_summary.csv
    │   └── {timestamp}_summary.txt
    └── predictions/
         └── ...

本脚本解析 summary CSV，输出到用户指定的 output_path。
"""

import argparse
import csv
import glob
import json
import os


def find_summary_file(opencompass_dir: str) -> str:
    """在 opencompass_dir 中查找 summary CSV 文件。

    优先选择 CSV 格式；如果存在多个同名 summary，取最新修改的文件。
    """
    patterns = [
        os.path.join(opencompass_dir, '**', '*summary*.csv'),
        os.path.join(opencompass_dir, '**', '*summary*.txt'),
    ]
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        if files:
            if len(files) > 1:
                # 取最新修改的文件，保证可复现
                files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
                print(f'[INFO] 找到 {len(files)} 个 summary 文件，取最新: {files[0]}')
            return files[0]
    return ''


def parse_summary_csv(summary_path: str) -> dict:
    """解析 OpenCompass summary CSV 文件。

    典型 CSV 格式（OpenCompass 不同版本列数不同）：
      v1: dataset,version,metric,score
      v2: dataset,version,metric,mode,model_score

    返回:
        { "ceval": {"accuracy": 0.65}, "gsm8k": {"accuracy": 0.58} }
    """
    results = {}
    with open(summary_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        first_row = next(reader, None)
        if first_row is None:
            print(f'[WARN] CSV 文件为空: {summary_path}')
            return results

        # 检测第一行是否为表头
        first_cells = [c.strip().lower() for c in first_row if c.strip()]
        is_header = (
            'dataset' in first_cells or
            'metric' in first_cells or
            'score' in first_cells
        )
        if not is_header:
            reader = [first_row] + list(reader)

        for row in reader:
            row = [col.strip() for col in row if col.strip()]
            if len(row) < 4:
                continue
            dataset = row[0]
            metric = row[2]
            if metric.lower() == 'metric' or dataset.lower() == 'dataset':
                continue
            # 尝试找到数值型 score：优先取最后一列，如果不是数值则逐个往前找
            score = None
            for col in reversed(row[3:]):
                try:
                    score = float(col)
                    break
                except ValueError:
                    continue
            if score is None:
                score = row[3]  # fallback to raw string
            if dataset not in results:
                results[dataset] = {}
            results[dataset][metric] = score
    return results


def parse_summary_txt(summary_path: str) -> dict:
    """解析 OpenCompass summary TXT 文件（备用格式）。

    OpenCompass TXT summary 可能包含以下非数据行：
      - 分隔线（含 '─' / '-' / '=' / '+' 等字符）
      - 标题行（以字母或中文开头但不含逗号）
      - 空行
      - 注释信息行
    本函数会先过滤这些行，再按逗号分隔解析。
    """
    results = {}
    # 非数据行的特征：分隔线（同一分隔符连续出现 3 次及以上）
    SEPARATOR_CHARS = {'─', '━', '═', '-', '=', '+', '|'}
    with open(summary_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 空行
            if not line:
                continue
            # 分隔线：由同一字符重复 3 次以上组成
            if line[0] in SEPARATOR_CHARS and len(line) >= 3:
                if all(c == line[0] for c in line):
                    continue
            # 不含逗号的行无法解析
            if ',' not in line:
                continue
            parts = [p.strip() for p in line.split(',')]
            # 需要至少 4 个字段: dataset, version, metric, score
            if len(parts) < 4:
                continue
            dataset = parts[0]
            metric = parts[2]
            # 跳过可能的表头（metric 列名本身是 metric 的情况极少，但做个防护）
            if metric.lower() == 'metric' or dataset.lower() == 'dataset':
                continue
            try:
                score = float(parts[3])
            except ValueError:
                score = parts[3]
            if dataset not in results:
                results[dataset] = {}
            results[dataset][metric] = score
    return results


def parse_results(opencompass_dir: str) -> dict:
    """自动识别并解析 summary 文件。"""
    summary_path = find_summary_file(opencompass_dir)
    if not summary_path:
        print(f"[WARN] 未找到 summary 文件，目录: {opencompass_dir}")
        return {}

    print(f"[INFO] 解析 summary: {summary_path}")
    if summary_path.endswith('.csv'):
        return parse_summary_csv(summary_path)
    else:
        return parse_summary_txt(summary_path)


def compute_overall(results: dict) -> float:
    """计算综合得分：对每个数据集优先取 accuracy，否则取第一个数值型指标。"""
    # 常见指标的优先级（按通用性降序）
    METRIC_PRIORITY = ['accuracy', 'acc', 'exact_match', 'f1', 'bleu', 'rouge']

    # OpenCompass 实际输出的 metric 名归一化映射
    # （bleu_score -> bleu, rouge_1/rouge_l -> rouge, accuracy_score -> accuracy 等）
    METRIC_ALIAS = {
        'accuracy_score': 'accuracy',
        'acc_score': 'acc',
        'exact': 'exact_match',
        'exact_match_score': 'exact_match',
        'bleu_score': 'bleu',
        'bleu': 'bleu',
        'rouge_1': 'rouge',
        'rouge_2': 'rouge',
        'rouge_l': 'rouge',
        'rouge': 'rouge',
        'f1_score': 'f1',
    }

    def _normalize(name: str) -> str:
        return METRIC_ALIAS.get(name.lower(), name.lower())

    scores = []
    for ds, metrics in results.items():
        if not metrics:
            continue

        chosen_metric = None
        chosen_value = None

        # 按优先级寻找可用的指标
        for preferred in METRIC_PRIORITY:
            for name, value in metrics.items():
                if _normalize(name) == preferred and isinstance(value, (int, float)):
                    chosen_metric = name
                    chosen_value = value
                    break
            if chosen_metric:
                break

        # 如果没有命中优先级列表，取第一个数值型指标
        if chosen_metric is None:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    chosen_metric = name
                    chosen_value = value
                    break

        if chosen_value is not None:
            scores.append((ds, chosen_metric, chosen_value))

    if not scores:
        return 0.0

    avg = round(sum(s[2] for s in scores) / len(scores), 4)
    metric_summary = ', '.join(f'{s[0]}={s[1]}:{s[2]}' for s in scores)
    print(f'[INFO] 综合得分基于: {metric_summary} -> overall={avg}')
    return avg


def save_metric_json(output_path: str, results: dict, metadata: dict):
    """写入 metric.json（平台自动采集）。"""
    data = {
        'model_name': metadata.get('model_name', ''),
        'model_version': metadata.get('model_version', ''),
        'model_path': metadata.get('model_path', ''),
        'datasets': metadata.get('datasets', []),
        'eval_results': results,
        'overall_score': compute_overall(results),
    }
    metric_path = os.path.join(output_path, 'metric.json')
    with open(metric_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[OK] 写入 metric.json: {metric_path}")


def save_summary_json(output_path: str, results: dict):
    """写入 eval_summary.json（简洁摘要）。"""
    summary_path = os.path.join(output_path, 'eval_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[OK] 写入 eval_summary.json: {summary_path}")


def save_report_csv(output_path: str, results: dict):
    """写入 eval_report.csv（可读表格）。"""
    csv_path = os.path.join(output_path, 'eval_report.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'metric', 'score'])
        for ds in sorted(results.keys()):
            metrics = results[ds]
            for metric in sorted(metrics.keys()):
                writer.writerow([ds, metric, metrics[metric]])
    print(f"[OK] 写入 eval_report.csv: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='解析 OpenCompass 评测结果')
    parser.add_argument('--opencompass-dir', required=True,
                        help='OpenCompass --work-dir 输出目录')
    parser.add_argument('--output-path', required=True,
                        help='评测结果输出根目录')
    parser.add_argument('--model-name', default='', help='模型名称')
    parser.add_argument('--model-version', default='', help='模型版本号')
    parser.add_argument('--model-path', default='', help='模型文件路径')
    parser.add_argument('--datasets', default='', help='评测数据集列表（逗号分隔）')
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    # 解析 OpenCompass 结果
    results = parse_results(args.opencompass_dir)

    # 元数据
    metadata = {
        'model_name': args.model_name,
        'model_version': args.model_version,
        'model_path': args.model_path,
        'datasets': [d.strip() for d in args.datasets.split(',') if d.strip()],
    }

    # 输出结构化文件
    save_metric_json(args.output_path, results, metadata)
    save_summary_json(args.output_path, results)
    save_report_csv(args.output_path, results)

    # 打印摘要
    print('\n========== 评测结果摘要 ==========')
    for ds in sorted(results.keys()):
        metrics_str = ', '.join(
            f'{k}={v}' for k, v in results[ds].items()
        )
        print(f'  {ds}: {metrics_str}')
    print(f'  综合得分: {compute_overall(results)}')
    print(f'  输出目录: {args.output_path}')
    print('==================================\n')


if __name__ == '__main__':
    main()
