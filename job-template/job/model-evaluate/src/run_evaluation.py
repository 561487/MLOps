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

    # ---- 5. 生成 config（目录场景：每文件一个 dataset） ----
    config_path = os.path.join(args.output_path, '_custom_config.py')
    lines = []
    lines.append('# Auto-generated OpenCompass config for custom dataset')
    lines.append('datasets = [')

    for data_file in data_files:
        # abbr: 文件名去扩展名
        basename = os.path.splitext(os.path.basename(data_file))[0]
        abbr = basename

        lines.append('    dict(')
        lines.append(f'        abbr={repr(abbr)},')
        lines.append("        type='opencompass.datasets.HFDataset',")
        lines.append(f'        path={repr(fmt)},')
        lines.append(f'        data_files={repr([data_file])},')
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
    return config_path


def build_opencompass_cmd(args: argparse.Namespace) -> list:
    """构建 OpenCompass 执行命令。

    由于 C-Eval 等数据集需要 trust_remote_code=True 才能加载，
    而 opencompass 内置的 dataset config 并未传递此参数，
    这里生成一个 wrapper 脚本，在调用 opencompass 入口前先 monkey-patch
    MsDataset.load，确保 trust_remote_code=True 在子进程中也生效。

    数据集来源二选一：
      - --custom_dataset_path：生成自定义 config，用 --config 传给 OpenCompass
      - --datasets：用 OpenCompass 内置数据集名，走 --datasets
    """
    work_dir = os.path.join(args.output_path, 'opencompass_results')
    os.makedirs(work_dir, exist_ok=True)

    # ---- 构建 opencompass CLI 参数（不含可执行文件） ----
    # 数据集参数：custom 路径用 --config，内置数据集用 --datasets
    if args.custom_dataset_path:
        config_path = build_custom_config(args)
        print(f'[INFO] 使用自定义数据集: {args.custom_dataset_path}')
        opencompass_args = ['--config', config_path]
    else:
        # datasets 参数：平台传逗号分隔字符串，需拆分为空格分隔的多个参数
        dataset_list = [d.strip() for d in args.datasets.split(',') if d.strip()]
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

    # 数据集缓存目录 → 直接设 environment，确保在 Python 启动前生效
    if args.datasets_cache_dir:
        os.makedirs(args.datasets_cache_dir, exist_ok=True)
        os.environ['MODELSCOPE_CACHE'] = args.datasets_cache_dir
        os.environ['HF_DATASETS_CACHE'] = args.datasets_cache_dir
        # 诊断：检查缓存目录是否已有数据
        existing = []
        for root, dirs, files in os.walk(args.datasets_cache_dir):
            for f in files:
                existing.append(os.path.join(root, f))
        cache_hit = len(existing) > 0
        print(f'[INFO] 数据集缓存目录: {args.datasets_cache_dir} (已有{len(existing)}个文件)' if cache_hit
              else f'[INFO] 数据集缓存目录: {args.datasets_cache_dir} (空，将下载)')

    # 样本数限制 → 通过环境变量传给 wrapper
    if args.max_samples > 0:
        print(f'[INFO] 样本数限制: 每个数据集最多取 {args.max_samples} 条')
        os.environ['OC_MAX_SAMPLES'] = str(args.max_samples)

    # 额外的模型加载参数
    if args.model_kwargs:
        try:
            model_kwargs = json.loads(args.model_kwargs)
            if isinstance(model_kwargs, dict):
                for k, v in model_kwargs.items():
                    opencompass_args.extend(['--model-kwargs', f'{k}={v}'])
            else:
                print(f'[WARN] model_kwargs 不是 JSON 对象，已跳过: {args.model_kwargs}')
        except json.JSONDecodeError:
            print(f'[WARN] model_kwargs 不是合法 JSON，已跳过: {args.model_kwargs}')

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


def run_opencompass(cmd: list, log_path: str):
    """执行 OpenCompass 命令，实时输出日志。"""
    print(f'[INFO] 执行命令: {" ".join(cmd)}')
    print(f'[INFO] 日志输出: {log_path}')
    print('=' * 60)

    process = None
    start_time = time.time()

    try:
        env = os.environ.copy()
        env.setdefault('HF_DATASETS_TRUST_REMOTE_CODE', '1')
        with open(log_path, 'w', encoding='utf-8') as log_f:
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
        sys.exit(1)
    except OSError as e:
        print(f'[ERROR] 系统调用失败: {e}')
        with open(log_path, 'a', encoding='utf-8') as log_f:
            log_f.write(f'\n[FATAL] 系统调用失败: {e}\n')
        sys.exit(1)
    except Exception as e:
        print(f'[ERROR] 执行过程中发生未知错误: {e}')
        with open(log_path, 'a', encoding='utf-8') as log_f:
            log_f.write(f'\n[FATAL] 未知错误: {e}\n')
        sys.exit(1)

    elapsed = time.time() - start_time
    print('=' * 60)
    print(f'[INFO] 总耗时: {elapsed:.1f}s')

    if process is None:
        print('[ERROR] 进程未能启动，无法获取退出码')
        sys.exit(1)

    returncode = process.returncode

    if returncode == -9:
        print(f'[ERROR] OpenCompass 被 OOM Killer 强制终止（SIGKILL）')
        sys.exit(1)

    if returncode != 0:
        print(f'[ERROR] OpenCompass 运行失败，退出码: {returncode}')
        sys.exit(returncode)

    print(f'[OK] OpenCompass 运行完成')


def run_parse_and_save(opencompass_dir: str, output_path: str,
                       model_name: str, model_version: str,
                       model_path: str, datasets: str):
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
    parser.add_argument('--num_gpus', type=int, default=1,
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

    check_opencompass()
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
    cmd = build_opencompass_cmd(args)
    log_path = os.path.join(args.output_path, 'eval_run.log')
    run_opencompass(cmd, log_path)

    # ---- 2. 解析结果 ----
    opencompass_out = os.path.join(args.output_path, 'opencompass_results')
    run_parse_and_save(
        opencompass_dir=opencompass_out,
        output_path=args.output_path,
        model_name=args.model_name,
        model_version=args.model_version,
        model_path=args.model_path,
        datasets=args.datasets,
    )

    print(f'\n[OK] 评测完成！结果目录: {args.output_path}')
    print(f'  - metric.json:      平台自动采集的指标')
    print(f'  - eval_summary.json: 结构化评测摘要')
    print(f'  - eval_report.csv:   评测结果表格')
    print(f'  - eval_run.log:      OpenCompass 运行日志')
    print(f'  - opencompass_results/: OpenCompass 原始输出\n')


if __name__ == '__main__':
    main()
