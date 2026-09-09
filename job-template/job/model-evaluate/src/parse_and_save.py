#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解析 OpenCompass 评测结果，保存为结构化文件（多 benchmark 聚合版）。

执行模型（run_evaluation.py）：用户一次选择多个内置数据集时，每个数据集
单独启动一次 OpenCompass run，每次 run 使用独立 work_dir：

  {output_path}/opencompass_results/<dataset>/{YYYYMMDD_HHMMSS}/
      ├── summary/summary_<timestamp>.csv
      └── results/<model>_hf/*.json

dataset -> run_dir -> summary_file 的映射由 run_evaluation.py 写入
{output_path}/dataset_status.json 的 runs 字段（路径相对 opencompass_results）。

本脚本不再"全局找最新的一个 summary"来代表整个 job，而是按 dataset_status.runs
逐数据集读取其自己的 summary，然后做 benchmark / subset 两层聚合：

  - 单结果 benchmark（gsm8k）：summary 一行 gsm8k -> benchmark score
  - 多 subset benchmark（ceval_gen / bbh_gen）：summary 每行一个 subset
    （ceval-xxx / bbh-xxx）-> details；benchmark score = 有效 subset 等权平均
    （与 OpenCompass 官方 leaderboard 口径一致）
  - overall_score = 成功 benchmark 的 benchmark-level score 等权平均
    （每个 benchmark 权重相同，不按 subset 行数加权）

输出三个文件共享同一聚合结果对象 report：
  metric.json        -> eval_results = report（平台采集）
  eval_summary.json  -> report（结构化摘要）
  eval_report.csv    -> report 展开成 benchmark/subset 行

兼容性回退：当 dataset_status.json 缺失或没有 runs 字段（旧版本产物 / 自定义
数据集单次运行），退化为解析整个 opencompass_results 中最新的一个 summary
（旧行为，仅用于历史产物兜底）。
"""

import argparse
import csv
import glob
import json
import os

# 常见指标的优先级（按通用性降序）。注意：OpenCompass 对 BBH 的 summary
# 中 metric 名为 'score'，CEval/GSM8K 为 'accuracy'，因此不能写死 metric。
METRIC_PRIORITY = ['accuracy', 'acc', 'score', 'exact_match', 'f1', 'bleu', 'rouge']

# OpenCompass 实际输出的 metric 名归一化映射
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


# --------------------------------------------------------------------------
# summary 文件解析（单文件粒度，兼容旧格式）
# --------------------------------------------------------------------------

def find_summary_file(opencompass_dir: str) -> str:
    """在 opencompass_dir 中查找 summary CSV 文件（兼容回退用）。

    仅用于旧布局/自定义数据集等"无 dataset_status.runs 映射"的场景；
    新布局下聚合走 dataset -> run -> summary 的确定性路径，不调用本函数。
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
        { "ceval-accountant": {"accuracy": 44.9}, "gsm8k": {"accuracy": 70.96} }
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
    """自动识别并解析 summary 文件（兼容回退：取整个目录最新的一个）。"""
    summary_path = find_summary_file(opencompass_dir)
    if not summary_path:
        print(f"[WARN] 未找到 summary 文件，目录: {opencompass_dir}")
        return {}

    print(f"[INFO] 解析 summary: {summary_path}")
    if summary_path.endswith('.csv'):
        return parse_summary_csv(summary_path)
    else:
        return parse_summary_txt(summary_path)


# --------------------------------------------------------------------------
# benchmark / subset 两层聚合
# --------------------------------------------------------------------------

def benchmark_root_name(dataset_arg: str) -> str:
    """平台数据集名 -> OpenCompass summary 中的数据集根名。

    'ceval_gen'/'ceval_ppl' -> 'ceval'；'gsm8k_gen' -> 'gsm8k'。
    OpenCompass summary 行中：单结果 benchmark 的行名 == 根名（gsm8k），
    多 subset benchmark 的行名 == '{根名}-{subset}'（ceval-accountant）。
    """
    for suffix in ('_gen', '_ppl'):
        if dataset_arg.endswith(suffix):
            return dataset_arg[:-len(suffix)]
    return dataset_arg


def _pick_metric_value(metrics: dict):
    """从某 summary 行（{metric: value}）中按优先级挑选 (metric名, 数值)。

    返回 (metric_name, value)，metric_name 为 summary 中的原名（如 'accuracy'
    或 'score'）；无任何数值型指标返回 None。
    """
    for preferred in METRIC_PRIORITY:
        for name, value in metrics.items():
            norm = METRIC_ALIAS.get(name.lower(), name.lower())
            if norm == preferred and isinstance(value, (int, float)):
                return name, value
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            return name, value
    return None


def aggregate_rows_into(report: dict, benchmark: str, rows: dict):
    """把一个 benchmark 自己的 summary 行聚合进 report（benchmark/subset 分层）。

    rows: {summary 行 dataset 名: {metric: 数值}}，全部来自该 benchmark 的 run。

    规则（不使用字符串猜测以外的启发式，见 benchmark_root_name）：
      - 行名 == 根名（如 gsm8k）→ 显式根行，其 score 直接作为 benchmark score；
      - 行名 == '{根名}-{subset}'（如 ceval-accountant / bbh-boolean_expressions）
        → 进入 details，benchmark score = 全部有效 subset 值等权平均；
      - 行名与根名无关（不应出现；自定义数据集场景由调用方走 flat 路径）
        → 各自独立成条，不进 details。

    多 subset benchmark 同时存在显式根行与 subset 行时（个别 benchmark 的
    summary 两者都有），benchmark score 优先取显式根行。
    """
    root = benchmark_root_name(benchmark)
    subsets = {}       # subset 名 -> (metric, value)
    explicit = None    # 显式根行 (metric, value)
    independent = {}   # 与根名无关的行

    for row_name, metrics in rows.items():
        picked = _pick_metric_value(metrics)
        if picked is None:
            print(f'[WARN] {benchmark}: 行 {row_name} 无数值型指标，跳过: {metrics}')
            continue
        metric, value = picked
        if row_name == root:
            explicit = (metric, value)
        elif root and row_name.startswith(root + '-'):
            subsets[row_name[len(root) + 1:]] = (metric, value)
        else:
            independent[row_name] = (metric, value)

    # ---- benchmark 主条目 ----
    used_metrics = {m for m, _ in subsets.values()}
    if explicit is not None:
        used_metrics.add(explicit[0])
    metric_label = explicit[0] if explicit is not None else (
        sorted(used_metrics)[0] if len(used_metrics) == 1 else '/'.join(sorted(used_metrics)))

    if explicit is not None or subsets:
        if explicit is not None:
            score = explicit[1]
        else:
            score = round(sum(v for _, v in subsets.values()) / len(subsets), 4)
        entry = {
            'status': 'succeeded',
            'metric': metric_label,
            'score': score,
        }
        if subsets:
            entry['details'] = {name: value for name, (_, value) in subsets.items()}
        report[benchmark] = entry

    # ---- 与根名无关的行：各自独立成条（正常只出现在 flat/兼容路径） ----
    for name, (metric, value) in independent.items():
        print(f'[WARN] {benchmark}: 行 {name} 与数据集根名 {root} 不匹配，'
              f'按独立结果保存')
        report[name] = {
            'status': 'succeeded',
            'metric': metric,
            'score': value,
        }


def load_dataset_status(output_path: str, explicit_path: str = '') -> dict:
    """读取 dataset_status.json（含 runs 映射）。文件缺失/损坏返回 {}。"""
    status_path = explicit_path or os.path.join(output_path, 'dataset_status.json')
    if not os.path.exists(status_path):
        return {}
    try:
        with open(status_path, 'r', encoding='utf-8') as f:
            status = json.load(f)
        if not isinstance(status, dict):
            print(f'[WARN] {status_path} 顶层不是 JSON 对象，忽略')
            return {}
        return status
    except (json.JSONDecodeError, OSError) as e:
        print(f'[WARN] 读取 {status_path} 失败: {e}')
        return {}


def _status_of(ds: str, succeeded: list, skipped: list) -> dict | None:
    """在 succeeded/skipped 列表中查找某数据集的旧式状态描述。"""
    if ds in succeeded:
        return {'status': 'succeeded'}
    for item in skipped:
        name = item.get('dataset', item) if isinstance(item, dict) else item
        if name == ds:
            if isinstance(item, dict):
                return {'status': 'failed', **item}
            return {'status': 'failed'}
    return None


def _entry_from_failed(ds: str, run: dict | None, item: dict | None) -> dict:
    """构造失败/跳过数据集的 metric.json 条目。"""
    entry = {'status': 'failed', 'metric': None, 'score': None}
    src = run or item or {}
    if src.get('exit_code') is not None:
        entry['exit_code'] = src['exit_code']
    if src.get('reason'):
        entry['status'] = 'skipped'
        entry['reason'] = src['reason']
    error = src.get('error')
    if error:
        entry['error'] = error
    return entry


def aggregate_job_results(opencompass_dir: str, output_path: str,
                          datasets: list, succeeded: list, skipped: list,
                          status: dict) -> dict:
    """多 benchmark job 结果聚合：逐数据集 → benchmark-level 条目。

    report: {平台数据集名(或 summary 行名): {status, metric, score, details?...}}
    顺序与用户选择的 datasets 一致；flat 场景多余行按 summary 顺序追加。
    """
    report: dict = {}
    runs = status.get('runs') if isinstance(status, dict) else None

    if runs:
        # ---- 新布局：dataset_status.runs 提供 dataset -> run -> summary 映射 ----
        run_by_ds = {}
        for run in runs:
            if isinstance(run, dict) and run.get('dataset'):
                run_by_ds[run['dataset']] = run

        for ds in datasets:
            run = run_by_ds.get(ds)
            if run is None:
                # runs 缺失该数据集（不应发生）：用 succeeded/skipped 兜底
                item = _status_of(ds, succeeded, skipped)
                print(f'[WARN] dataset_status.runs 缺少数据集 {ds}'
                      f'{"" if item is None else "，按状态列表兜底"}')
                if item is None:
                    continue
                run = {'dataset': ds, **item}

            run_status = run.get('status')
            if run_status == 'succeeded':
                rows = {}
                summary_rel = run.get('summary_file')
                if summary_rel:
                    summary_path = os.path.join(opencompass_dir, summary_rel)
                    if os.path.exists(summary_path):
                        rows = parse_summary_csv(summary_path)
                    else:
                        print(f'[WARN] 数据集 {ds} 的 summary 文件不存在: '
                              f'{summary_path}，跳过其结果解析')
                aggregate_rows_into(report, ds, rows)
                if ds not in report:
                    # summary 为空（运行成功但无 summary 的异常场景）
                    report[ds] = {'status': 'succeeded', 'metric': None,
                                  'score': None}
                    print(f'[WARN] 数据集 {ds} 的 summary 无任何可解析行')
            elif run_status == 'failed':
                report[ds] = _entry_from_failed(ds, run, None)
            else:  # skipped（预跳过）等
                entry = {'status': run_status or 'skipped', 'metric': None,
                         'score': None}
                if run.get('reason'):
                    entry['reason'] = run['reason']
                report[ds] = entry
        return report

    # ---- 兼容回退：无 runs 映射（旧版本产物 / 自定义数据集单次运行） ----
    print('[INFO] dataset_status.json 无 runs 映射，按旧布局解析整个目录中最新的 summary')
    rows = parse_results(opencompass_dir)
    covered = set()

    for ds in datasets:
        root = benchmark_root_name(ds)
        matched = {n: v for n, v in rows.items()
                   if n == root or (root and n.startswith(root + '-'))}
        if matched:
            aggregate_rows_into(report, ds, matched)
            covered.update(matched.keys())
        else:
            item = _status_of(ds, succeeded, skipped)
            if item is not None and item['status'] != 'succeeded':
                report[ds] = _entry_from_failed(ds, None, item)
            elif ds in succeeded:
                report[ds] = {'status': 'succeeded', 'metric': None, 'score': None}
                print(f'[WARN] 数据集 {ds} 在 summary 中无对应行，无法给出分数')

    for name, metrics in rows.items():
        if name in covered:
            continue
        picked = _pick_metric_value(metrics)
        if picked is None:
            continue
        metric, value = picked
        report[name] = {'status': 'succeeded', 'metric': metric, 'score': value}

    return report


def compute_overall(results: dict) -> float:
    """综合得分：成功 benchmark 的 benchmark-level score 等权平均。

    每个 benchmark（无论含多少 subset）只贡献一个 score，杜绝多 subset
    benchmark 因行数多而被过度加权的问题。无成功分数时返回 0.0。
    """
    scores = []
    for entry in results.values():
        if not isinstance(entry, dict):
            continue
        if entry.get('status') != 'succeeded':
            continue
        score = entry.get('score')
        if isinstance(score, (int, float)):
            scores.append(score)
    if not scores:
        return 0.0
    avg = round(sum(scores) / len(scores), 4)
    detail = ', '.join(f'{k}={v.get("score")}' for k, v in results.items()
                       if isinstance(v, dict) and isinstance(v.get('score'), (int, float)))
    print(f'[INFO] 综合得分基于 {len(scores)} 个成功 benchmark'
          f'（等权平均）: {detail} -> overall={avg}')
    return avg


# --------------------------------------------------------------------------
# 输出文件
# --------------------------------------------------------------------------

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
    if metadata.get('succeeded_datasets') is not None:
        data['succeeded_datasets'] = metadata['succeeded_datasets']
    if metadata.get('skipped_datasets') is not None:
        data['skipped_datasets'] = metadata['skipped_datasets']
    metric_path = os.path.join(output_path, 'metric.json')
    with open(metric_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[OK] 写入 metric.json: {metric_path}")


def save_summary_json(output_path: str, results: dict):
    """写入 eval_summary.json（与 metric.json.eval_results 同源）。"""
    summary_path = os.path.join(output_path, 'eval_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[OK] 写入 eval_summary.json: {summary_path}")


def save_report_csv(output_path: str, results: dict):
    """写入 eval_report.csv（benchmark 行 + subset 明细行）。

    表头: benchmark,subset,status,metric,score,exit_code
      ceval_gen,,succeeded,accuracy,51.84,
      ceval_gen,accountant,succeeded,accuracy,44.9,
      ...
      commonsenseqa_gen,,failed,,,1
    """
    csv_path = os.path.join(output_path, 'eval_report.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['benchmark', 'subset', 'status', 'metric', 'score',
                         'exit_code'])
        for bench in results:
            entry = results[bench]
            if not isinstance(entry, dict):
                continue
            status = entry.get('status', 'succeeded')
            metric = entry.get('metric') or ''
            score = entry.get('score')
            exit_code = entry.get('exit_code') if entry.get('exit_code') is not None else ''
            writer.writerow([bench, '', status, metric,
                             '' if score is None else score, exit_code])
            details = entry.get('details') or {}
            for subset in sorted(details.keys()):
                v = details[subset]
                writer.writerow([bench, subset, status, metric,
                                 '' if v is None else v, ''])
    print(f"[OK] 写入 eval_report.csv: {csv_path}")


def print_report(report: dict):
    """控制台打印聚合结果摘要。"""
    print('\n========== 评测结果摘要（benchmark 层） ==========')
    for name in report:
        entry = report[name]
        if not isinstance(entry, dict):
            continue
        status = entry.get('status', 'succeeded')
        if status == 'succeeded' and entry.get('score') is not None:
            details = entry.get('details') or {}
            suffix = f' ({len(details)} 个 subset)' if details else ''
            print(f'  {name}: [{status}] {entry.get("metric")}={entry["score"]}{suffix}')
        elif status == 'succeeded':
            print(f'  {name}: [{status}] 无分数')
        else:
            extras = []
            if entry.get('exit_code') is not None:
                extras.append(f'exit_code={entry["exit_code"]}')
            if entry.get('reason'):
                extras.append(f'reason={entry["reason"]}')
            if entry.get('error'):
                extras.append(f'error={entry["error"][:120]}')
            print(f'  {name}: [{status}] '
                  + (', '.join(extras) if extras else '无详情'))
    print('==================================================\n')


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
    parser.add_argument('--succeeded-datasets', default='',
                        help='成功完成的数据集（逗号分隔）')
    parser.add_argument('--skipped-datasets', default='',
                        help='跳过的数据集（JSON 数组）')
    parser.add_argument('--status-file', default='',
                        help='dataset_status.json 路径（缺省取 output-path 下同名文件）')
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    # 数据集状态：优先用 dataset_status.json 的 runs 映射；无则退回命令行参数
    status = load_dataset_status(args.output_path, args.status_file)
    succeeded_arg = [d.strip() for d in args.succeeded_datasets.split(',')
                     if d.strip()] if args.succeeded_datasets else []
    skipped_arg = []
    if args.skipped_datasets:
        try:
            skipped_arg = json.loads(args.skipped_datasets)
        except json.JSONDecodeError:
            print(f'[WARN] skipped-datasets 不是合法 JSON，已忽略: '
                  f'{args.skipped_datasets}')
    if not succeeded_arg and not skipped_arg and status:
        succeeded_arg = status.get('succeeded', [])
        skipped_arg = status.get('skipped', [])

    datasets_arg = [d.strip() for d in args.datasets.split(',') if d.strip()]

    # 聚合：核心入口（替代旧的"只取最新一个 summary"）
    report = aggregate_job_results(
        args.opencompass_dir, args.output_path,
        datasets=datasets_arg,
        succeeded=succeeded_arg,
        skipped=skipped_arg,
        status=status,
    )

    # 元数据
    metadata = {
        'model_name': args.model_name,
        'model_version': args.model_version,
        'model_path': args.model_path,
        'datasets': datasets_arg,
    }
    if succeeded_arg:
        metadata['succeeded_datasets'] = succeeded_arg
    if skipped_arg:
        metadata['skipped_datasets'] = skipped_arg

    # 输出结构化文件（三者同源于 report）
    save_metric_json(args.output_path, report, metadata)
    save_summary_json(args.output_path, report)
    save_report_csv(args.output_path, report)

    # 打印摘要
    print_report(report)
    print(f'  综合得分: {compute_overall(report)}')
    if metadata.get('succeeded_datasets') is not None:
        print(f'  成功数据集: {", ".join(metadata["succeeded_datasets"]) or "无"}')
    if metadata.get('skipped_datasets'):
        skipped_names = [
            item.get('dataset', str(item)) for item in metadata['skipped_datasets']
        ]
        print(f'  跳过数据集: {", ".join(skipped_names) or "无"}')
    print(f'  输出目录: {args.output_path}')
    print('==================================\n')


if __name__ == '__main__':
    main()
