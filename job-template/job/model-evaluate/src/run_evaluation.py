#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cube Studio Job Template: model-evaluate-opencompass

基于 OpenCompass 的 LLM 模型评测入口脚本。
被容器 ENTRYPOINT 调用，接收用户在 pipeline 编辑器中配置的参数。

参数通过 argparse 从命令行传入（Cube Studio 平台传递机制）。
"""

import argparse
import json
import os
import subprocess
import sys
import textwrap
import time

# ---- Monkey-patch: 为 MsDataset.load 自动注入 trust_remote_code=True ----
# C-Eval 等数据集需要执行远程脚本，OpenCompass 内置的 dataset config 没有传这个参数
_original_msdataset_load = None

def _patch_msdataset():
    global _original_msdataset_load
    try:
        from modelscope import MsDataset
        _original_msdataset_load = MsDataset.load
        def _patched_load(*args, **kwargs):
            kwargs.setdefault('trust_remote_code', True)
            return _original_msdataset_load(*args, **kwargs)
        MsDataset.load = staticmethod(_patched_load)
        print('[INFO] MsDataset.load 已 patch: trust_remote_code=True')
    except ImportError:
        pass

_patch_msdataset()

# ---- 平台注入的环境变量 ----
KFJ_CREATOR = os.getenv('KFJ_CREATOR', 'admin')
KFJ_RUN_ID = os.getenv('KFJ_RUN_ID', '')
KFJ_PIPELINE_ID = os.getenv('KFJ_PIPELINE_ID', '0')
KFJ_TASK_PROJECT_NAME = os.getenv('KFJ_TASK_PROJECT_NAME', 'public')


DEFAULT_MODEL_KWARGS = {
    'device_map': 'auto',
    'torch_dtype': 'bfloat16',
}


def resolve_model_kwargs(model_kwargs_str: str) -> dict:
    """合并默认 model_kwargs 与用户覆盖项（用户优先）。"""
    kwargs = dict(DEFAULT_MODEL_KWARGS)
    if not model_kwargs_str:
        return kwargs
    try:
        user_kwargs = json.loads(model_kwargs_str)
        if isinstance(user_kwargs, dict):
            kwargs.update(user_kwargs)
        else:
            print(f'[WARN] model_kwargs 不是 JSON 对象，使用默认值: {model_kwargs_str}')
    except json.JSONDecodeError:
        print(f'[WARN] model_kwargs 不是合法 JSON，使用默认值: {model_kwargs_str}')
    return kwargs


def resolve_num_gpus(explicit=None):
    """Use pipeline allocation; retain the legacy CLI override for old jobs."""
    if explicit is not None:
        if explicit < 1:
            raise ValueError('num_gpus 必须大于 0')
        return explicit
    import math
    import re
    raw = os.environ.get('KFJ_TASK_RESOURCE_GPU', '').strip()
    amount = re.split(r'[（(]', raw)[0].strip()
    if amount and ',' not in amount:
        try:
            requested = float(amount)
        except ValueError:
            raise ValueError('无法解析流水线 GPU 申请: %s' % raw)
        if not math.isfinite(requested) or requested <= 0:
            raise ValueError('请在公共资源参数中申请 GPU')
        return math.ceil(requested)
    import torch
    visible = torch.cuda.device_count()
    if visible < 1:
        raise ValueError('未检测到分配的 GPU，请检查公共资源配置')
    return visible


def apply_num_gpus_visibility(num_gpus: int):
    """限制进程可见 GPU，使 device_map=auto 与 --num_gpus 一致。

    device_map=auto 会使用当前进程可见的全部 GPU（CUDA_VISIBLE_DEVICES），
    不会读取 --num_gpus；因此在启动前按 num_gpus 裁剪可见设备列表。
    """
    if num_gpus <= 0:
        return
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if cuda_visible:
        devices = [d for d in cuda_visible.split(',') if d.strip() != '']
        if len(devices) > num_gpus:
            limited = ','.join(devices[:num_gpus])
            os.environ['CUDA_VISIBLE_DEVICES'] = limited
            print(f'[INFO] CUDA_VISIBLE_DEVICES 已限制为前 {num_gpus} 卡: {limited}')
        elif len(devices) < num_gpus:
            print(f'[WARN] CUDA_VISIBLE_DEVICES={cuda_visible} 仅 {len(devices)} 卡，'
                  f'但 --num_gpus={num_gpus}')
    else:
        limited = ','.join(str(i) for i in range(num_gpus))
        os.environ['CUDA_VISIBLE_DEVICES'] = limited
        print(f'[INFO] CUDA_VISIBLE_DEVICES 未设置，按 --num_gpus={num_gpus} 设为: {limited}')


def check_opencompass():
    """检查 OpenCompass CLI 是否可用。"""
    try:
        result = subprocess.run(
            ['opencompass', '--help'],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            print(f"[ERROR] opencompass CLI 不可用: {result.stderr}")
            sys.exit(1)
        print(f"[INFO] opencompass CLI 可用")
    except FileNotFoundError:
        print("[ERROR] opencompass 命令未找到，请确认镜像中已安装 opencompass")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        # opencompass CLI 启动需 import torch/transformers，冷缓存时可超过 10s+
        # 能启动到超时说明命令存在，仅首次加载慢，不视为致命错误
        print("[WARN] opencompass --help 响应慢(120s 超时)，CLI 存在，继续执行")


def check_gpu_available(num_gpus: int = 1):
    """检查 GPU 可用性和显存，输出诊断信息。"""
    try:
        import torch
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            print(f'[ERROR] 未检测到可用 GPU！'
                  f' CUDA_VISIBLE_DEVICES={cuda_visible}')
            print('[HINT] 检查：1) nvidia-smi 是否正常 2) CUDA_VISIBLE_DEVICES 是否指向有效 GPU')
            sys.exit(1)
        print(f'[INFO] 检测到 {gpu_count} 个可用 GPU'
              f' (CUDA_VISIBLE_DEVICES={cuda_visible})')
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            # 兼容新旧版 torch: 遍历所有可能的内存属性名
            total_mem = 0
            for attr in ('total_memory', 'total_mem', 'totalGlobalMem'):
                try:
                    val = getattr(props, attr, None)
                    if val is not None and val > 0:
                        total_mem = val
                        break
                except Exception:
                    continue
            mem_total_gb = total_mem / (1024 ** 3)
            mem_allocated_gb = torch.cuda.memory_allocated(i) / (1024 ** 3)
            mem_free_gb = mem_total_gb - mem_allocated_gb
            print(f'  GPU {i}: {props.name}, '
                  f'显存 {mem_total_gb:.1f}GB, 已用 {mem_allocated_gb:.1f}GB, '
                  f'可用 ~{mem_free_gb:.1f}GB')
        if gpu_count < num_gpus:
            print(f'[ERROR] 需要 {num_gpus} 个 GPU，但仅检测到 {gpu_count} 个')
            sys.exit(1)
    except ImportError:
        print('[WARN] 无法导入 torch，跳过 GPU 检查')
    except Exception as e:
        print(f'[WARN] GPU 检查失败: {e}')


def _resolve_evaluator_type(metric: str) -> str:
    """把 metric 名映射为 OpenCompass Evaluator 类的完整 type 路径。

    会用 importlib 探测目标类是否存在；探测失败则 fallback 到 AccEvaluator，
    避免 config 加载时因类名错误直接崩溃。
    """
    METRIC_EVALUATOR_MAP = {
        'accuracy': 'opencompass.openicl.icl_evaluator.AccEvaluator',
        'exact_match': 'opencompass.openicl.icl_evaluator.ExactMatchEvaluator',
        'bleu': 'opencompass.openicl.icl_evaluator.BleuScoreEvaluator',
        'rouge': 'opencompass.openicl.icl_evaluator.RougeEvaluator',
    }
    type_str = METRIC_EVALUATOR_MAP.get(metric, METRIC_EVALUATOR_MAP['accuracy'])
    try:
        mod_path, cls_name = type_str.rsplit('.', 1)
        import importlib
        mod = importlib.import_module(mod_path)
        if not hasattr(mod, cls_name):
            print(f'[WARN] Evaluator 类 {cls_name} 不存在，fallback 到 AccEvaluator')
            return METRIC_EVALUATOR_MAP['accuracy']
    except Exception as e:
        print(f'[WARN] 探测 {type_str} 失败 ({e})，fallback 到 AccEvaluator')
        return METRIC_EVALUATOR_MAP['accuracy']
    return type_str


def _read_first_record(file_path: str) -> dict:
    """读取 jsonl/json/csv 文件的第一条记录，返回字段名列表。"""
    import csv as _csv
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.csv':
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = _csv.reader(f)
                header = next(reader, None)
                return {col: '' for col in (header or [])}
        else:
            # jsonl / json
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if not first_line:
                    return {}
                return json.loads(first_line)
    except Exception as e:
        print(f'[WARN] 读取文件首行失败 {file_path}: {e}')
        return {}


def infer_dataset_schema(file_path: str) -> dict:
    """读首行数据，自动推断列名/题型/指标。

    返回 dict:
      input_col: 问题列名
      output_col: 答案列名
      choice_cols: 选项列名列表（空=生成式）
      metric: 推断的指标
      fmt: 'json' 或 'csv'
    """
    record = _read_first_record(file_path)
    fields = list(record.keys()) if record else []

    # ---- 格式 ----
    ext = os.path.splitext(file_path)[1].lower()
    fmt = 'csv' if ext == '.csv' else 'json'

    # ---- 答案列 ----
    ANSWER_CANDIDATES = ['target', 'answer', 'label', 'gold', 'gt']
    output_col = ''
    for cand in ANSWER_CANDIDATES:
        if cand in fields:
            output_col = cand
            break
    if not output_col and fields:
        output_col = fields[-1]  # fallback: 最后一列

    # ---- 选项列(MCQ) ----
    # 策略1: A,B,C,D 同时存在
    choice_cols = []
    upper_letters = [c for c in fields if len(c) == 1 and c.isupper()]
    if len(upper_letters) >= 2:
        choice_cols = sorted(upper_letters)
    else:
        # 策略2: option_a/option_b/... 或 option1/option2/...
        option_prefix = [f for f in fields if f.lower().startswith('option')]
        if len(option_prefix) >= 2:
            choice_cols = sorted(option_prefix)

    is_mcq = len(choice_cols) > 0

    # ---- 问题列 ----
    INPUT_CANDIDATES = ['input', 'question', 'query', 'prompt']
    input_col = ''
    for cand in INPUT_CANDIDATES:
        if cand in fields:
            input_col = cand
            break
    if not input_col:
        # 去掉答案列和选项列后取第一个
        remaining = [f for f in fields if f != output_col and f not in choice_cols]
        input_col = remaining[0] if remaining else (fields[0] if fields else 'input')

    # ---- 指标 ----
    metric = 'accuracy' if is_mcq else 'exact_match'

    # ---- 打印推断结果 ----
    task_type = 'MCQ(选择题)' if is_mcq else '生成式问答'
    print(f'[INFO] 自动推断: 问题列={input_col}, 答案列={output_col}, '
          f'选项列={choice_cols or "无"}, 题型={task_type}, 指标={metric}')

    if not fields:
        print(f'[WARN] 无法读取数据字段，使用默认值: input/target/无选项/accuracy')
        return {
            'input_col': 'input', 'output_col': 'target',
            'choice_cols': [], 'metric': 'accuracy', 'fmt': fmt,
        }

    return {
        'input_col': input_col,
        'output_col': output_col,
        'choice_cols': choice_cols,
        'metric': metric,
        'fmt': fmt,
    }


def _find_list_format_fields(record: dict):
    """检测「列表打包」式 MCQ 数据格式。

    这种格式把选项文本和正确答案标记以列表形式存在单条记录里，例如：
      {question:..., options:[str,...], correct:[bool,...], id:int, ...}
    而非标准 MCQ 的「每个选项一列」(A/B/C/D 独立列)。

    Returns:
      (question_col, options_col, correct_col) 三元组；非列表格式返回 None。
    """
    if not isinstance(record, dict):
        return None
    # 收集 list[str] 候选（选项文本列）和 list[bool] 候选（答案标记列）
    str_list_cols = []  # [(colname, value)]
    correct_col = None
    for k, v in record.items():
        if isinstance(v, list) and len(v) >= 2:
            if all(isinstance(x, bool) for x in v):
                correct_col = k
            elif all(isinstance(x, str) for x in v):
                str_list_cols.append((k, v))
    if not str_list_cols or not correct_col:
        return None
    # 从 list[str] 候选里挑真正的「选项文本列」（排除 option_ids 这类短 id 列）
    kl = lambda name: name.lower()
    if any(k == 'options' for k, _ in str_list_cols):
        options_col = 'options'
    elif any('option' in kl(k) and 'id' not in kl(k) for k, _ in str_list_cols):
        options_col = next(k for k, _ in str_list_cols
                           if 'option' in kl(k) and 'id' not in kl(k))
    elif any('text' in kl(k) or 'choice' in kl(k) for k, _ in str_list_cols):
        options_col = next(k for k, _ in str_list_cols
                           if 'text' in kl(k) or 'choice' in kl(k))
    else:
        # 兜底: 取元素平均长度最长的列（选项文本通常比短 id 长很多）
        options_col = max(str_list_cols,
                          key=lambda kv: sum(len(x) for x in kv[1]) / len(kv[1]))[0]
    for cand in ['question', 'input', 'query', 'prompt']:
        if cand in record and isinstance(record[cand], str):
            return (cand, options_col, correct_col)
    for k, v in record.items():
        if isinstance(v, str) and k != options_col:
            return (k, options_col, correct_col)
    return None


def _normalize_list_format_file(input_file: str, output_dir: str):
    """把列表打包式 MCQ 文件转成标准 MCQ 格式。

    标准格式：{question:..., A:opt0, B:opt1, ..., answer:'C'}
    这样现有 infer_dataset_schema 能正确推断（选项列=A/B/C...，答案列=answer，
    字母匹配 answer_pattern）。原数据的 option_ids(P/Q/R/S/T) 统一映射成 A/B/C...。

    Returns:
      标准化后的文件路径；若 input_file 不是列表格式，返回 None（调用方用原文件）。
    """
    import string
    records = []
    fields = None
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if fields is None:
                fields = _find_list_format_fields(rec)
                if not fields:
                    return None
            q_col, opt_col, ans_col = fields
            options_list = rec.get(opt_col, [])
            correct_list = rec.get(ans_col, [])
            if len(options_list) != len(correct_list) or not options_list:
                continue
            true_idx = next((i for i, c in enumerate(correct_list) if c), None)
            if true_idx is None:
                continue
            new_rec = {q_col: rec[q_col], 'answer': string.ascii_uppercase[true_idx]}
            for i, opt_text in enumerate(options_list):
                new_rec[string.ascii_uppercase[i]] = opt_text
            records.append(new_rec)
    if not records:
        return None
    base = os.path.splitext(os.path.basename(input_file))[0]
    out_file = os.path.join(output_dir, f'_normalized_{base}.jsonl')
    os.makedirs(output_dir, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    n_opts = sum(1 for k in records[0] if len(k) == 1 and k.isupper())
    print(f'[INFO] 列表式数据已标准化: {os.path.basename(input_file)} -> '
          f'{out_file} ({len(records)} 条, {n_opts} 选项)')
    return out_file


def build_custom_config(args: argparse.Namespace) -> str:
    """为自定义数据集生成 OpenCompass config 文件，返回文件路径。

    支持:
      - 单文件 / 单目录（目录内每个文件作为独立数据集）
      - 自动推断列名/题型/指标（读首行数据）
      - --custom_columns JSON 覆盖列名（推断失败时救场）
      - --custom_metric 覆盖指标
      - --custom_prompt_template 覆盖 prompt
    输出: {output_path}/_custom_config.py
    """
    import glob as _glob

    path = args.custom_dataset_path
    if not os.path.exists(path):
        raise FileNotFoundError(f'自定义数据集路径不存在: {path}')

    # ---- 1. 收集数据文件列表 ----
    json_exts = {'.json', '.jsonl'}
    csv_exts = {'.csv'}
    if os.path.isdir(path):
        all_files = []
        for f in sorted(os.listdir(path)):
            ext = os.path.splitext(f)[1].lower()
            if ext in json_exts or ext in csv_exts:
                all_files.append(os.path.join(path, f))
        if not all_files:
            raise FileNotFoundError(f'目录 {path} 下未找到 .json/.jsonl/.csv 文件')
        # 检查格式统一
        exts_found = set(os.path.splitext(f)[1].lower() for f in all_files)
        has_json = bool(exts_found & json_exts)
        has_csv = bool(exts_found & csv_exts)
        if has_json and has_csv:
            raise ValueError(
                f'目录 {path} 包含混合格式 {sorted(exts_found)}，'
                '要求统一为 .json/.jsonl 或 .csv')
        data_files = all_files
    else:
        data_files = [path]

    # ---- 1.5 列表打包式 MCQ 预处理 ----
    # FailureSensorIQ 等数据集把选项文本/正确答案标记以列表形式存在单条记录里
    # （options:[str,...], correct:[bool,...]），infer_dataset_schema 会误把列表列
    # 当独立选项、把题目序号 id 当答案。这里先展开成标准 MCQ 格式
    # （A/B/C... 独立列 + answer 字母列），让后续推断与 prompt 模板正确工作。
    normalized_files = []
    for f in data_files:
        nf = _normalize_list_format_file(f, args.output_path)
        normalized_files.append(nf if nf else f)
    data_files = normalized_files

    # ---- 2. 推断 schema（用第一个文件） ----
    schema = infer_dataset_schema(data_files[0])

    # ---- 3. 用户覆盖 ----
    # --custom_columns JSON 覆盖列名
    if args.custom_columns:
        try:
            cols_override = json.loads(args.custom_columns)
            if isinstance(cols_override, dict):
                if 'input' in cols_override:
                    schema['input_col'] = cols_override['input']
                if 'target' in cols_override:
                    schema['output_col'] = cols_override['target']
                if 'choices' in cols_override:
                    raw = cols_override['choices']
                    if isinstance(raw, str):
                        schema['choice_cols'] = [c.strip() for c in raw.split(',') if c.strip()]
                    elif isinstance(raw, list):
                        schema['choice_cols'] = [str(c).strip() for c in raw if str(c).strip()]
                    else:
                        print(f'[WARN] --custom_columns choices 格式不支持({type(raw)})，已忽略')
                    # 覆盖 choices 后按新 choice_cols 重算 metric，保持题型与指标一致
                    schema['metric'] = 'accuracy' if schema['choice_cols'] else 'exact_match'
                print(f'[INFO] 列名已覆盖: input={schema["input_col"]}, '
                      f'target={schema["output_col"]}, choices={schema["choice_cols"]}, '
                      f'metric={schema["metric"]}')
        except json.JSONDecodeError:
            print(f'[WARN] --custom_columns 不是合法 JSON，已忽略: {args.custom_columns}')

    # --custom_metric 覆盖指标
    metric = (args.custom_metric or schema['metric']).lower()

    # ---- 4. 构建公共配置 ----
    is_mcq = len(schema['choice_cols']) > 0
    input_columns = [schema['input_col']] + schema['choice_cols']
    output_column = schema['output_col']
    evaluator_type = _resolve_evaluator_type(metric)
    fmt = schema['fmt']

    # prompt 模板
    if args.custom_prompt_template:
        prompt_str = args.custom_prompt_template
    elif is_mcq:
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        options_block = '\n'.join(
            f'{letters[i]}) {{{col}}}' for i, col in enumerate(schema['choice_cols']))
        last_letter = letters[len(schema['choice_cols']) - 1]
        prompt_str = (
            "Answer the following multiple choice question. The last line of your "
            "response should be of the following format: 'ANSWER: $LETTER' "
            f"(without quotes) where LETTER is one of A-{last_letter}. "
            "Think step by step before answering.\n\n"
            f"{{{schema['input_col']}}}\n\n{options_block}"
        )
    else:
        prompt_str = '{' + schema['input_col'] + '}'

    # ---- 5. 生成 models 配置 ----
    # OpenCompass get_config_from_arg 分支1: 当 args.config 有值时，
    # 加载 config 后直接 return，完全忽略 --hf-path 等 CLI 参数。
    # 因此 config 必须自带 models，否则 partitioner 取 cfg['models'] 会 KeyError。
    is_chat = (args.model_type == 'hf_chat')
    model_cls = 'HuggingFacewithChatTemplate' if is_chat else 'HuggingFaceBaseModel'
    model_type_str = f'opencompass.models.huggingface_above_v4_33.{model_cls}'
    model_abbr = os.path.basename(args.model_path.rstrip('/')) + '_hf'
    model_kwargs = resolve_model_kwargs(args.model_kwargs)

    # ---- 6. 生成 config（目录场景：每文件一个 dataset） ----
    config_path = os.path.join(args.output_path, '_custom_config.py')
    lines = []
    lines.append('# Auto-generated OpenCompass config for custom dataset')
    # models 块：与 OpenCompass cli/main.py 的 --hf-path 分支保持一致的字段集
    lines.append('models = [')
    lines.append('    dict(')
    lines.append(f'        type={repr(model_type_str)},')
    lines.append(f'        abbr={repr(model_abbr)},')
    lines.append(f'        path={repr(args.model_path)},')
    lines.append(f'        model_kwargs={repr(model_kwargs)},')
    lines.append(f'        max_seq_len={args.max_seq_len},')
    lines.append(f'        max_out_len={args.max_out_len},')
    lines.append(f'        batch_size={args.batch_size},')
    lines.append(f'        run_cfg=dict(num_gpus={args.num_gpus}),')
    lines.append('    ),')
    lines.append(']')
    lines.append('datasets = [')

    for data_file in data_files:
        # abbr: 文件名去扩展名
        basename = os.path.splitext(os.path.basename(data_file))[0]
        abbr = basename

        lines.append('    dict(')
        lines.append(f'        abbr={repr(abbr)},')
        lines.append("        type='opencompass.datasets.HFDataset',")
        lines.append(f'        path={repr(fmt)},')
        # HFDataset.load 把 data_files 传给 get_data_path(期望 str)，必须用 str 不能用 list
        lines.append(f'        data_files={repr(data_file)},')
        lines.append('        reader_cfg=dict(')
        lines.append(f'            input_columns={repr(input_columns)},')
        lines.append(f'            output_column={repr(output_column)},')
        lines.append("            train_split='train',")
        lines.append("            test_split='train',")
        lines.append('        ),')
        lines.append('        infer_cfg=dict(')
        lines.append('            prompt_template=dict(')
        lines.append("                type='opencompass.openicl.icl_prompt_template.PromptTemplate',")
        lines.append('                template=dict(')
        lines.append('                    round=[')
        lines.append(f'                        dict(prompt={repr(prompt_str)}, role="HUMAN"),')
        lines.append('                    ],')
        lines.append('                ),')
        lines.append('            ),')
        lines.append("            retriever=dict(type='opencompass.openicl.icl_retriever.ZeroRetriever'),")
        lines.append("            inferencer=dict(type='opencompass.openicl.icl_inferencer.GenInferencer'),")
        lines.append('        ),')
        lines.append('        eval_cfg=dict(')
        if is_mcq:
            last_letter = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[len(schema['choice_cols']) - 1]
            answer_pattern = f'(?i)ANSWER\\s*:\\s*([A-{last_letter}])'
            lines.append('            pred_postprocessor=dict(')
            lines.append(f'                answer_pattern={repr(answer_pattern)},')
            lines.append("                type='opencompass.utils.text_postprocessors.match_answer_pattern'),")
        lines.append(f'            evaluator=dict(type={repr(evaluator_type)}),')
        lines.append('        ),')
        lines.append('    ),')

    lines.append(']')
    config_code = '\n'.join(lines)

    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_code + '\n')

    print(f'[INFO] 自定义数据集 config 已生成: {config_path}')
    print(f'[INFO] 数据文件({len(data_files)}个): {data_files}')
    print(f'[INFO] 格式: {fmt}, MCQ: {is_mcq}, 指标: {metric} -> {evaluator_type}')
    print(f'[INFO] 模型: {model_abbr}, type={model_type_str}, '
          f'num_gpus={args.num_gpus}, batch_size={args.batch_size}, '
          f'max_seq_len={args.max_seq_len}, max_out_len={args.max_out_len}')
    return config_path


def list_exp_dirs(work_dir: str) -> list:
    """列出 work_dir 下已有的 OpenCompass 时间戳运行目录（如 20260902_100358）。

    只按目录名排序，不依赖 mtime。"""
    if not os.path.isdir(work_dir):
        return []
    return sorted(name for name in os.listdir(work_dir)
                  if os.path.isdir(os.path.join(work_dir, name)))


def find_run_summary_csv(run_dir: str) -> str | None:
    """在单个 OpenCompass run 目录内定位 summary CSV（确定性规则）。

    OpenCompass 每次运行在 {run_dir}/summary/ 下写 summary_<run_name>.csv，
    run_name 即时间戳目录名。优先精确匹配 run_name；匹配不到时若目录内
    只有一个 csv 则取它。不依赖 mtime。找不到返回 None。
    """
    summary_dir = os.path.join(run_dir, 'summary')
    if not os.path.isdir(summary_dir):
        return None
    run_name = os.path.basename(run_dir.rstrip('/'))
    csvs = sorted(
        f for f in os.listdir(summary_dir)
        if f.endswith('.csv') and f.startswith('summary')
    )
    if not csvs:
        # 兜底：目录里没有任何 summary_*.csv 时，接受任意 csv
        csvs = sorted(f for f in os.listdir(summary_dir) if f.endswith('.csv'))
    if not csvs:
        return None
    for f in csvs:
        if f.startswith(run_name):
            return os.path.join(summary_dir, f)
    if len(csvs) == 1:
        return os.path.join(summary_dir, csvs[0])
    # 同 run 目录内多份 csv：文件名排序取第一个并告警（本 run 一次只应产出一份）
    print(f'[WARN] {summary_dir} 存在多个 summary csv，取 {csvs[0]}')
    return os.path.join(summary_dir, csvs[0])


def extract_failure_summary(log_path: str, start_offset: int,
                            max_chars: int = 800) -> str:
    """从 eval_run.log 的指定区段（一次 OpenCompass 运行的输出）提取可读错误摘要。

    start_offset 为该次运行写入日志前的文件字节偏移。从区段末尾向前找
    Traceback/Error 等特征行并取其后的文本；找不到特征行时取区段末尾。
    """
    try:
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            f.seek(start_offset)
            chunk = f.read()
    except OSError as e:
        return f'无法读取运行日志({log_path}): {e}'
    if not chunk.strip():
        return ''
    marks = ('Traceback', 'Error', 'ERROR', 'Exception', 'raise ',
             'FileNotFoundError', 'No valid url', 'AssertionError',
             'KeyError', 'ValueError', 'RuntimeError')
    lines = chunk.splitlines()
    hit = None
    for i in range(len(lines) - 1, -1, -1):
        if any(m in lines[i] for m in marks):
            hit = i
            break
    if hit is not None:
        text = '\n'.join(lines[max(0, hit - 2):])
    else:
        text = chunk[-2000:]
    text = text.strip()
    if len(text) > max_chars:
        text = '...(truncated)\n' + text[-max_chars:].lstrip()
    return text


def get_dataset_skip_reason(dataset: str, model_type: str) -> str | None:
    """返回数据集与模型类型不兼容时的跳过原因；兼容则返回 None。"""
    if model_type == 'hf_chat' and dataset.endswith('_ppl'):
        return 'hf_chat 不支持 _ppl 数据集'
    return None


def build_opencompass_cmd(args: argparse.Namespace,
                          dataset_list: list | None = None,
                          work_dir: str | None = None) -> list:
    """构建 OpenCompass 执行命令。

    由于 C-Eval 等数据集需要 trust_remote_code=True 才能加载，
    而 opencompass 内置的 dataset config 并未传递此参数，
    这里生成一个 wrapper 脚本，在调用 opencompass 入口前先 monkey-patch
    MsDataset.load，确保 trust_remote_code=True 在子进程中也生效。

    数据集来源二选一：
      - --custom_dataset_path：生成自定义 config，作为位置参数传给 OpenCompass
      - --datasets：用 OpenCompass 内置数据集名，走 --datasets

    work_dir：本次运行的 OpenCompass --work-dir（OpenCompass 会在其下生成
    {YYYYMMDD_HHMMSS} 运行子目录）。缺省时用 {output_path}/opencompass_results
    （自定义数据集等单次运行场景）。多内置数据集逐数据集运行时由调用方传入
    各数据集独立的 work_dir，保证 dataset -> run_dir 一一对应。
    """
    if work_dir is None:
        work_dir = os.path.join(args.output_path, 'opencompass_results')
    os.makedirs(work_dir, exist_ok=True)

    # ---- 构建 opencompass CLI 参数（不含可执行文件） ----
    # 数据集参数：custom 路径用 config 位置参数（OpenCompass 的 config 是位置参数，
    # 不能用 --config 前缀，否则 argparse 会撞 --config-dir/--config-verbose 报歧义），
    # 内置数据集用 --datasets
    if args.custom_dataset_path:
        config_path = build_custom_config(args)
        print(f'[INFO] 使用自定义数据集: {args.custom_dataset_path}')
        opencompass_args = [config_path]
    else:
        # datasets 参数：平台传逗号分隔字符串，需拆分为空格分隔的多个参数
        if dataset_list is None:
            dataset_list = [d.strip() for d in args.datasets.split(',') if d.strip()]

        # ---- 约束：hf_chat 只能选 _gen 数据集 ----
        # HuggingFacewithChatTemplate 不支持 PPL 评测(NotImplementedError)
        incompatible = [
            d for d in dataset_list
            if get_dataset_skip_reason(d, args.model_type)
        ]
        if incompatible:
            if len(dataset_list) == 1:
                reason = get_dataset_skip_reason(dataset_list[0], args.model_type)
                print(f'[ERROR] 模型类型为 hf_chat 时只能选择 _gen 数据集，'
                      f'检测到非法的 _ppl 数据集: {dataset_list[0]}')
                print('[HINT] 二选一修改：'
                      '1) 从评测数据集中去掉 _ppl 项；'
                      '2) 将模型类型改为 hf_base(base 支持 _ppl 和 _gen)')
                sys.exit(1)
            # 多数据集逐条运行时由 run_builtin_datasets_with_skip 跳过，此处仅过滤
            dataset_list = [
                d for d in dataset_list
                if not get_dataset_skip_reason(d, args.model_type)
            ]
            if not dataset_list:
                print('[ERROR] 过滤 _ppl 数据集后无可评测项')
                sys.exit(1)

        print(f'[INFO] 数据集列表(内置): {dataset_list}')
        opencompass_args = ['--datasets'] + dataset_list
    opencompass_args += [
        '--hf-path', args.model_path,
        '--hf-num-gpus', str(args.num_gpus),
        '--batch-size', str(args.batch_size),
        '--max-seq-len', str(args.max_seq_len),
        '--max-out-len', str(args.max_out_len),
        '--work-dir', work_dir,
        '--debug',
    ]

    # 模型类型
    if args.model_type == 'hf_chat':
        opencompass_args.extend(['--hf-type', 'chat'])
    elif args.model_type == 'hf_base':
        opencompass_args.extend(['--hf-type', 'base'])
    else:
        opencompass_args.extend(['--hf-type', 'chat'])

    # Few-shot
    if args.few_shot > 0:
        os.environ['OPENCOMPASS_FEW_SHOT'] = str(args.few_shot)
        print(f'[INFO] Few-shot 设置为 {args.few_shot}，'
              '请确保数据集配置中引用了该环境变量')

    # ---- DATASET_SOURCE 处理：交给 sitecustomize 的智能选源补丁逐数据集决策 ----
    # 优先级（每个数据集独立判断）：
    #   1. 本地缓存已存在 → 直接用
    #   2. OSS 直链可用（DATASETS_URL 覆盖 piqa/gaokao/gsm8k/bbh/ceval 等 95 个）→ 自动下载
    #   3. 无 OSS 直链（如 nq/xsum/obqa/siqa）→ 临时启用 ModelScope repo 加载
    #   4. 都不可用 → 报错并指引离线数据包
    if 'DATASET_SOURCE' in os.environ:
        print(f'[INFO] 清除全局 DATASET_SOURCE={os.environ["DATASET_SOURCE"]}，'
              f'改由智能选源补丁逐数据集决策')
        os.environ.pop('DATASET_SOURCE', None)

    # 数据集缓存目录 → 直接设 environment，确保在 Python 启动前生效
    if args.datasets_cache_dir:
        # 防呆告警：缓存目录不应是自定义数据集目录本身
        if args.custom_dataset_path and (
            args.custom_dataset_path == args.datasets_cache_dir
            or args.datasets_cache_dir.startswith(
                args.custom_dataset_path.rstrip('/') + '/')
            or args.custom_dataset_path.startswith(
                args.datasets_cache_dir.rstrip('/') + '/')
        ):
            print(f'[WARN] datasets_cache_dir={args.datasets_cache_dir} 与 '
                  f'custom_dataset_path={args.custom_dataset_path} 重叠或嵌套，'
                  f'可能导致 OpenCompass 在自定义数据集目录下查找 ./data/<内置数据集>/ 子目录。'
                  f'建议：datasets_cache_dir 使用独立的缓存目录（如 '
                  f'/mnt/storage/models-share-volume/opencompass_data）')
        os.makedirs(args.datasets_cache_dir, exist_ok=True)
        os.environ['MODELSCOPE_CACHE'] = args.datasets_cache_dir
        os.environ['HF_DATASETS_CACHE'] = args.datasets_cache_dir
        # get_data_path 实际读的是 COMPASS_DATA_CACHE（前面两个对内置数据集无效）
        os.environ['COMPASS_DATA_CACHE'] = args.datasets_cache_dir
        # 诊断：检查缓存目录是否已有数据
        existing = []
        for root, dirs, files in os.walk(args.datasets_cache_dir):
            for f in files:
                existing.append(os.path.join(root, f))
        cache_hit = len(existing) > 0
        print(f'[INFO] 数据集缓存目录: {args.datasets_cache_dir} (已有{len(existing)}个文件)' if cache_hit
              else f'[INFO] 数据集缓存目录: {args.datasets_cache_dir} (空，将下载)')
    else:
        # 无缓存目录时，设默认值（get_data_path 走 else 分支时会用到）
        os.environ.setdefault('COMPASS_DATA_CACHE',
                              os.path.expanduser('~/.cache/opencompass'))

    # 样本数限制 → 通过环境变量传给 wrapper
    if args.max_samples > 0:
        print(f'[INFO] 样本数限制: 每个数据集最多取 {args.max_samples} 条')
        os.environ['OC_MAX_SAMPLES'] = str(args.max_samples)

    # 模型加载参数（默认 device_map=auto + torch_dtype=bfloat16，用户可覆盖）
    model_kwargs = resolve_model_kwargs(args.model_kwargs)
    print(f'[INFO] model_kwargs: {model_kwargs}')
    for k, v in model_kwargs.items():
        opencompass_args.extend(['--model-kwargs', f'{k}={v}'])

    # ---- 生成 wrapper 脚本（monkey-patch MsDataset.load → 调用 opencompass 入口） ----
    wrapper_path = os.path.join(args.output_path, '_opencompass_wrapper.py')
    wrapper_code = textwrap.dedent('''\
        #!/usr/bin/env python3
        """Auto-generated: monkey-patch MsDataset.load BEFORE OpenCompass starts."""
        import os
        import sys

        # 确保 HuggingFace datasets 和 modelscope 都允许执行远程脚本
        os.environ.setdefault('HF_DATASETS_TRUST_REMOTE_CODE', '1')

        # 显式传播 CUDA_VISIBLE_DEVICES（防止 OpenCompass 内部覆盖）
        _cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if _cuda_devices:
            os.environ['CUDA_VISIBLE_DEVICES'] = _cuda_devices
            print(f'[wrapper] CUDA_VISIBLE_DEVICES={_cuda_devices}')

        # ---- 样本数限制 ----
        _max_samples = int(os.environ.get('OC_MAX_SAMPLES', '0'))
        if _max_samples > 0:
            print(f'[wrapper] 样本数限制: 每个数据集最多 {_max_samples} 条')

        # ---- Monkey-patch MsDataset.load ----
        from modelscope import MsDataset
        _ms_original_load = MsDataset.load

        def _ms_patched_load(*pa, **kw):
            kw.setdefault('trust_remote_code', True)
            ds = _ms_original_load(*pa, **kw)
            if _max_samples > 0:
                ds = _truncate_dataset(ds, _max_samples)
            return ds

        MsDataset.load = staticmethod(_ms_patched_load)
        if hasattr(MsDataset, '_load'):
            _ms_original_load2 = MsDataset._load
            @staticmethod
            def _ms_patched_load2(*pa, **kw):
                kw.setdefault('trust_remote_code', True)
                ds = _ms_original_load2(*pa, **kw)
                if _max_samples > 0:
                    ds = _truncate_dataset(ds, _max_samples)
                return ds
            MsDataset._load = _ms_patched_load2

        print('[wrapper] MsDataset.load patched: trust_remote_code=True')

        # ---- Monkey-patch HuggingFace load_dataset ----
        try:
            import datasets as _hf_datasets
            _hf_original = _hf_datasets.load_dataset
            def _hf_patched(*pa, **kw):
                ds = _hf_original(*pa, **kw)
                if _max_samples > 0:
                    ds = _truncate_dataset(ds, _max_samples)
                return ds
            _hf_datasets.load_dataset = _hf_patched
            print('[wrapper] HuggingFace load_dataset patched')
        except ImportError:
            pass

        # ---- 截断工具函数 ----
        def _truncate_dataset(ds, limit):
            """将 dataset 截断到最多 limit 条，不够则全取。"""
            if isinstance(ds, dict):
                return {k: _truncate_dataset(v, limit) for k, v in ds.items()}
            if hasattr(ds, '__len__') and hasattr(ds, 'select'):
                n = len(ds)
                if n > limit:
                    return ds.select(range(limit))
            return ds

        # ---- 将控制权交给 OpenCompass CLI ----
        from opencompass.cli.main import main
        main()
    ''').strip()

    with open(wrapper_path, 'w', encoding='utf-8') as f:
        f.write(wrapper_code + '\n')

    print(f'[INFO] 已生成 wrapper 脚本: {wrapper_path}')

    # 返回: python /path/to/wrapper.py [opencompass_args...]
    return [sys.executable, wrapper_path] + opencompass_args


def run_opencompass(cmd: list, log_path: str, *,
                    fail_on_error: bool = True,
                    append_log: bool = False) -> int:
    """执行 OpenCompass 命令，实时输出日志。"""
    print(f'[INFO] 执行命令: {" ".join(cmd)}')
    print(f'[INFO] 日志输出: {log_path}')
    print('=' * 60)

    process = None
    start_time = time.time()

    try:
        env = os.environ.copy()
        env.setdefault('HF_DATASETS_TRUST_REMOTE_CODE', '1')
        env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        log_mode = 'a' if append_log else 'w'
        with open(log_path, log_mode, encoding='utf-8') as log_f:
            if append_log:
                sep = f'\n{"=" * 60}\n[{time.strftime("%Y-%m-%d %H:%M:%S")}] 继续评测下一个数据集\n{"=" * 60}\n'
                print(sep, end='')
                log_f.write(sep)
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                env=env,
            )

            # 实时读取输出
            for line in iter(process.stdout.readline, ''):
                print(line, end='', flush=True)
                log_f.write(line)

            process.wait()

    except FileNotFoundError:
        print(f'[ERROR] 命令未找到: {cmd[0]}，请确认 OpenCompass 已正确安装')
        with open(log_path, 'a', encoding='utf-8') as log_f:
            log_f.write(f'\n[FATAL] 命令未找到: {cmd[0]}\n')
        if fail_on_error:
            sys.exit(1)
        return 127
    except OSError as e:
        print(f'[ERROR] 系统调用失败: {e}')
        with open(log_path, 'a', encoding='utf-8') as log_f:
            log_f.write(f'\n[FATAL] 系统调用失败: {e}\n')
        if fail_on_error:
            sys.exit(1)
        return 1
    except Exception as e:
        print(f'[ERROR] 执行过程中发生未知错误: {e}')
        with open(log_path, 'a', encoding='utf-8') as log_f:
            log_f.write(f'\n[FATAL] 未知错误: {e}\n')
        if fail_on_error:
            sys.exit(1)
        return 1

    elapsed = time.time() - start_time
    print('=' * 60)
    print(f'[INFO] 总耗时: {elapsed:.1f}s')

    if process is None:
        print('[ERROR] 进程未能启动，无法获取退出码')
        if fail_on_error:
            sys.exit(1)
        return 1

    returncode = process.returncode

    if returncode == -9:
        print(f'[ERROR] OpenCompass 被 OOM Killer 强制终止（SIGKILL）')
        if fail_on_error:
            sys.exit(1)
        return returncode

    if returncode != 0:
        print(f'[ERROR] OpenCompass 运行失败，退出码: {returncode}')
        if fail_on_error:
            sys.exit(returncode)
        return returncode

    print(f'[OK] OpenCompass 运行完成')
    return 0


def save_dataset_status(output_path: str, succeeded: list, skipped: list,
                        runs: list | None = None):
    """写入 dataset_status.json，记录各数据集运行状态。

    runs（可选）：每次 OpenCompass 运行的明细，与用户选择顺序一致，字段：
      dataset / status(succeeded|failed|skipped) /
      run_dir / summary_file / exit_code / error / reason
    其中 run_dir、summary_file 为相对 {output_path}/opencompass_results 的路径。
    """
    status_path = os.path.join(output_path, 'dataset_status.json')
    payload = {
        'succeeded': succeeded,
        'skipped': skipped,
        'total': len(succeeded) + len(skipped),
    }
    if runs is not None:
        payload['runs'] = runs
    with open(status_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f'[OK] 数据集运行状态已写入: {status_path}')
    if runs:
        print('[INFO] 运行目录映射:')
        for r in runs:
            ds = r['dataset']
            run_dir = r.get('run_dir') or '-'
            summary = r.get('summary_file') or '-'
            print(f'  - {ds}: status={r.get("status")}, '
                  f'run_dir={run_dir}, summary={summary}')
            for k in ('exit_code', 'reason'):
                if r.get(k) is not None:
                    print(f'      {k}={r[k]}')
            if r.get('error'):
                err = r['error'].replace('\n', ' ')[:160]
                print(f'      error={err}')


def print_dataset_summary(succeeded: list, skipped: list):
    """打印数据集运行汇总。"""
    print('\n========== 数据集运行汇总 ==========')
    print(f'  成功 ({len(succeeded)}): {", ".join(succeeded) if succeeded else "无"}')
    if skipped:
        skipped_desc = []
        for item in skipped:
            ds = item.get('dataset', item)
            code = item.get('exit_code')
            reason = item.get('reason', '')
            if reason:
                skipped_desc.append(f'{ds} ({reason})')
            elif code is not None:
                skipped_desc.append(f'{ds} (exit={code})')
            else:
                skipped_desc.append(str(ds))
        print(f'  跳过 ({len(skipped)}): {", ".join(skipped_desc)}')
    else:
        print('  跳过 (0): 无')
    print('====================================\n')


def run_builtin_datasets_with_skip(args: argparse.Namespace) -> tuple[list, list]:
    """逐数据集运行 OpenCompass；单个失败时跳过并继续。

    每个数据集使用独立的 --work-dir: {output_path}/opencompass_results/<dataset>/
    OpenCompass 在其中再生成 {YYYYMMDD_HHMMSS} 运行子目录。运行前后对
    work_dir 做目录快照差集，即可确定 dataset -> run_dir -> summary_file 的
    明确对应关系（不依赖全局 mtime / latest 目录猜测）。

    每次运行的 run_dir/summary_file 连同 status/exit_code/error 写入
    dataset_status.json 的 runs 字段，供 parse_and_save.py 逐数据集聚合。
    """
    dataset_list = [d.strip() for d in args.datasets.split(',') if d.strip()]
    if not dataset_list:
        print('[ERROR] --datasets 为空')
        sys.exit(1)

    log_path = os.path.join(args.output_path, 'eval_run.log')
    opencompass_root = os.path.join(args.output_path, 'opencompass_results')

    pre_skipped = []
    runnable = []
    for ds in dataset_list:
        reason = get_dataset_skip_reason(ds, args.model_type)
        if reason:
            pre_skipped.append({'dataset': ds, 'reason': reason})
        else:
            runnable.append(ds)

    if pre_skipped:
        names = ', '.join(item['dataset'] for item in pre_skipped)
        print(f'[WARN] 以下数据集与 model_type={args.model_type} 不兼容，将自动跳过: {names}')
        if args.model_type == 'hf_chat':
            print('[HINT] hf_chat 仅支持 _gen 数据集；如需 _ppl 请改用 hf_base，或从列表中去掉 _ppl 项')

    print(f'[INFO] 共 {len(dataset_list)} 个内置数据集'
          f'（可运行 {len(runnable)}，预跳过 {len(pre_skipped)}），'
          f'运行失败时将自动跳过并继续')

    skipped = list(pre_skipped)
    succeeded = []
    runs_by_ds: dict[str, dict] = {}

    for idx, ds in enumerate(runnable):
        print(f'\n{"=" * 60}')
        print(f'[INFO] 评测数据集 ({idx + 1}/{len(runnable)}): {ds}')
        print(f'{"=" * 60}')

        # 每数据集独立 work_dir；运行前快照已有 run 目录
        work_dir = os.path.join(opencompass_root, ds)
        os.makedirs(work_dir, exist_ok=True)
        existed = set(list_exp_dirs(work_dir))

        # 记录本轮日志起点，失败时用于提取错误摘要
        log_offset = 0
        if os.path.exists(log_path):
            log_offset = os.path.getsize(log_path)

        cmd = build_opencompass_cmd(args, dataset_list=[ds], work_dir=work_dir)
        returncode = run_opencompass(
            cmd, log_path,
            fail_on_error=False,
            append_log=(idx > 0),
        )

        # 快照差集 = 本次运行新增的 OpenCompass 时间戳目录（一次调用应恰好 1 个）
        new_dirs = [d for d in list_exp_dirs(work_dir) if d not in existed]
        if len(new_dirs) > 1:
            print(f'[WARN] 数据集 {ds} 一次运行新增了 {len(new_dirs)} 个目录: '
                  f'{new_dirs}，取最后一个')
        run_name = new_dirs[-1] if new_dirs else None
        run_rel = f'{ds}/{run_name}' if run_name else None

        # 定位该 run 的 summary CSV
        summary_rel = None
        if run_name is not None:
            summary_abs = find_run_summary_csv(os.path.join(work_dir, run_name))
            if summary_abs:
                summary_rel = os.path.relpath(summary_abs, opencompass_root)

        if returncode == 0:
            succeeded.append(ds)
            runs_by_ds[ds] = {
                'dataset': ds,
                'status': 'succeeded',
                'run_dir': run_rel,
                'summary_file': summary_rel,
            }
            print(f'[OK] 数据集 {ds} 评测完成'
                  + (f' (run_dir={run_rel})' if run_rel else ''))
            if summary_rel is None:
                print(f'[WARN] 数据集 {ds} 未找到 summary 文件，其结果将无法被解析')
        else:
            skipped.append({'dataset': ds, 'exit_code': returncode})
            runs_by_ds[ds] = {
                'dataset': ds,
                'status': 'failed',
                'run_dir': run_rel,
                'summary_file': None,
                'exit_code': returncode,
                'error': extract_failure_summary(log_path, log_offset),
            }
            print(f'[WARN] 数据集 {ds} 评测失败(退出码 {returncode})，已跳过，继续下一个')

    # 预跳过项也补 runs 记录，保证每个用户选择的数据集都有明确状态
    for item in pre_skipped:
        ds = item['dataset']
        runs_by_ds[ds] = {
            'dataset': ds,
            'status': 'skipped',
            'run_dir': None,
            'summary_file': None,
            'reason': item.get('reason'),
        }

    # 按用户选择顺序输出 runs
    runs = [runs_by_ds[ds] for ds in dataset_list if ds in runs_by_ds]

    print_dataset_summary(succeeded, skipped)
    save_dataset_status(args.output_path, succeeded, skipped, runs=runs)

    if not succeeded:
        if skipped:
            print('[ERROR] 所有数据集均未能成功完成（含跳过/失败项）')
        else:
            print('[ERROR] 无可运行的数据集')
        sys.exit(1)

    return succeeded, skipped


def run_parse_and_save(opencompass_dir: str, output_path: str,
                       model_name: str, model_version: str,
                       model_path: str, datasets: str,
                       succeeded_datasets: list | None = None,
                       skipped_datasets: list | None = None):
    """调用 parse_and_save.py 解析结果。"""
    parse_script = os.path.join(os.path.dirname(__file__), 'parse_and_save.py')
    if not os.path.exists(parse_script):
        print(f'[WARN] parse_and_save.py 未找到: {parse_script}，跳过解析')
        return

    cmd = [
        sys.executable,
        parse_script,
        '--opencompass-dir', opencompass_dir,
        '--output-path', output_path,
        '--model-name', model_name,
        '--model-version', model_version,
        '--model-path', model_path,
        '--datasets', datasets,
    ]
    if succeeded_datasets is not None:
        cmd.extend(['--succeeded-datasets', ','.join(succeeded_datasets)])
    if skipped_datasets is not None:
        cmd.extend(['--skipped-datasets', json.dumps(skipped_datasets, ensure_ascii=False)])

    print(f'[INFO] 解析评测结果: {" ".join(cmd)}')
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f'[WARN] 结果解析异常: {result.stderr}')

    # 软链 metric.json 到容器根目录（平台采集用）
    metric_json_src = os.path.join(output_path, 'metric.json')
    metric_json_dst = '/metric.json'
    if os.path.exists(metric_json_src):
        if os.path.exists(metric_json_dst):
            os.remove(metric_json_dst)
        os.symlink(metric_json_src, metric_json_dst)
        print(f'[OK] 软链 /metric.json -> {metric_json_src}')


def main():
    parser = argparse.ArgumentParser(
        description='OpenCompass 模型评测（作为 Cube Studio Pipeline 节点）'
    )

    # ---- 模型信息 ----
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型路径（HuggingFace 本地路径或模型目录）')
    parser.add_argument('--model_name', type=str, default='model',
                        help='模型名称')
    parser.add_argument('--model_version', type=str, default='v1',
                        help='模型版本号')
    parser.add_argument('--model_type', type=str, default='hf_chat',
                        choices=['hf_chat', 'hf_base'],
                        help='模型类型: hf_chat（对话模型）/ hf_base（基座模型）')

    # ---- 评测配置 ----
    parser.add_argument('--datasets', type=str, required=False, default='',
                        help='内置数据集名称，逗号分隔，如: ceval_gen,gsm8k_gen。'
                             '与 --custom_dataset_path 互斥，二选一')
    parser.add_argument('--output_path', type=str, required=True,
                        help='评测结果输出根目录，所有文件写入此路径')

    # ---- 推理参数 ----
    parser.add_argument('--num_gpus', type=int, default=None,
                        help='使用的 GPU 数量，默认 1')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='推理 batch size，默认 64')
    parser.add_argument('--max_seq_len', type=int, default=2048,
                        help='模型最大输入长度，默认 2048')
    parser.add_argument('--max_out_len', type=int, default=512,
                        help='模型最大输出长度，默认 512')
    parser.add_argument('--few_shot', type=int, default=0,
                        help='Few-shot 样本数，默认 0')
    parser.add_argument('--model_kwargs', type=str, default='',
                        help='模型加载参数（JSON 字符串），如: {"device_map":"auto"}')

    # ---- 数据集控制 ----
    parser.add_argument('--max_samples', type=int, default=0,
                        help='每个数据集最多评测的样本数，0=全量。取前N条，不够则全取')
    parser.add_argument('--datasets_cache_dir', type=str, default='',
                        help='数据集缓存目录（挂载卷路径），有则复用，无则下载到此目录')

    # ---- 自定义数据集（与 --datasets 互斥） ----
    parser.add_argument('--custom_dataset_path', type=str, default='',
                        help='自定义数据集路径（容器内绝对路径，文件或目录）。'
                             '与 --datasets 互斥；提供时走自定义评测路径。'
                             '列名/题型/指标自动推断，目录内每个文件作为独立数据集')
    parser.add_argument('--custom_columns', type=str, default='',
                        help='列名覆盖(JSON)，推断失败时救场。'
                             '如 {"input":"q","target":"ans","choices":"opt1,opt2,opt3,opt4"}。'
                             '空=自动推断')
    parser.add_argument('--custom_metric', type=str, default='',
                        choices=['', 'accuracy', 'exact_match', 'bleu', 'rouge'],
                        help='评测指标覆盖。空=自动(MCQ→accuracy / 生成式→exact_match)')
    parser.add_argument('--custom_prompt_template', type=str, default='',
                        help='自定义 prompt 模板（含 {input}/{A} 等占位符）。'
                             '空=自动(MCQ 用 ANSWER 模板 / 生成式用 {input})')

    args = parser.parse_args()
    args.num_gpus = resolve_num_gpus(args.num_gpus)

    # ---- 打印参数 ----
    print('========== OpenCompass 评测配置 ==========')
    for k, v in vars(args).items():
        print(f'  {k}: {v}')
    print(f'  KFJ_CREATOR: {KFJ_CREATOR}')
    print(f'  KFJ_RUN_ID: {KFJ_RUN_ID}')
    print(f'  KFJ_PIPELINE_ID: {KFJ_PIPELINE_ID}')
    print('==========================================\n')

    # ---- 校验 ----
    # custom_dataset_path 与 datasets 互斥，二选一
    if args.custom_dataset_path and args.datasets:
        print('[ERROR] --custom_dataset_path 与 --datasets 不能同时指定，二选一')
        sys.exit(1)
    if not args.custom_dataset_path and not args.datasets:
        print('[ERROR] 必须指定 --datasets 或 --custom_dataset_path 之一')
        sys.exit(1)

    from evaluation_bridge import run_standard_evaluation, prepare_public_adapter
    if args.custom_dataset_path and run_standard_evaluation(args):
        return
    # Preserve the master OpenCompass path for public and legacy custom datasets.
    prepare_public_adapter(args)

    check_opencompass()
    apply_num_gpus_visibility(args.num_gpus)
    check_gpu_available(args.num_gpus)
    # 仅对本地路径做存在性检查，HF 模型 ID（如 Qwen/Qwen2.5-0.5B-Instruct）由 OpenCompass 自行下载
    is_local_path = args.model_path.startswith('/') or args.model_path.startswith('./')
    if is_local_path:
        if os.path.exists(args.model_path):
            print(f'[INFO] 模型路径存在: {args.model_path}')
        else:
            print(f'[WARN] 模型路径不存在: {args.model_path}')
            # 诊断：检查上级目录和挂载点
            parent = args.model_path
            for _ in range(3):
                parent = os.path.dirname(parent)
                if not parent or parent == '/':
                    break
                if os.path.exists(parent):
                    print(f'[INFO] 上级目录存在: {parent}/')
                    try:
                        items = os.listdir(parent)
                        print(f'[INFO] 目录内容({len(items)}项): {items[:20]}')
                    except Exception as e:
                        print(f'[WARN] 无法列出目录: {e}')
                    break
                else:
                    print(f'[WARN] 上级目录也不存在: {parent}/')
            print('[INFO] 将继续尝试运行，由 OpenCompass 自行处理路径错误')
    if not is_local_path:
        print(f'[INFO] 检测到 HF 模型 ID，将由 OpenCompass 自动下载: {args.model_path}')

    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)

    # ---- 1. 运行 OpenCompass ----
    succeeded_datasets = None
    skipped_datasets = None
    if args.custom_dataset_path:
        cmd = build_opencompass_cmd(args)
        log_path = os.path.join(args.output_path, 'eval_run.log')
        run_opencompass(cmd, log_path)
    else:
        succeeded_datasets, skipped_datasets = run_builtin_datasets_with_skip(args)

    # ---- 2. 解析结果 ----
    opencompass_out = os.path.join(args.output_path, 'opencompass_results')
    run_parse_and_save(
        opencompass_dir=opencompass_out,
        output_path=args.output_path,
        model_name=args.model_name,
        model_version=args.model_version,
        model_path=args.model_path,
        datasets=args.datasets,
        succeeded_datasets=succeeded_datasets,
        skipped_datasets=skipped_datasets,
    )

    print(f'\n[OK] 评测完成！结果目录: {args.output_path}')
    print(f'  - metric.json:      平台自动采集的指标')
    print(f'  - eval_summary.json: 结构化评测摘要')
    print(f'  - eval_report.csv:   评测结果表格')
    print(f'  - eval_run.log:      OpenCompass 运行日志')
    print(f'  - opencompass_results/: OpenCompass 原始输出\n')


if __name__ == '__main__':
    main()
