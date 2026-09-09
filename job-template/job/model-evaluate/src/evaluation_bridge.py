"""Add Convert eval_qa support without replacing master OpenCompass behavior."""
import json
import os
from pathlib import Path

from custom_evaluator import (discover_dataset_files, evaluate_custom_dataset,
                              _read_file, _write_json)
from model_artifact import resolve_model_artifact, configure_adapter_environment


def is_standard_dataset(dataset_path):
    path = Path(dataset_path).expanduser().resolve()
    if path.is_dir():
        for name in ('dataset_manifest.json', 'manifest.json'):
            manifest = path / name
            if manifest.is_file():
                info = json.loads(manifest.read_text(encoding='utf-8-sig'))
                schema = info.get('target_schema') or info.get('format_type')
                if schema != 'eval_qa':
                    raise ValueError('评测 manifest 须声明 eval_qa；messages 请先经过 DatasetConvert')
                discover_dataset_files(str(path))  # validate file boundary before GPU loading
                return True
    # A canonical file explicitly carries input + target + type. Legacy files
    # without a type keep the master OpenCompass path and custom overrides.
    files = discover_dataset_files(str(path))
    found = False
    for file in files:
        if file.suffix.lower() == '.jsonl':
            with file.open(encoding='utf-8-sig') as stream:
                first = next((json.loads(line) for line in stream if line.strip()), None)
        else:
            rows = _read_file(file)
            first = rows[0] if rows else None
        if isinstance(first, dict) and {'input','target','type'} <= set(first):
            found = True
    return found


def prepare_public_adapter(args):
    artifact = resolve_model_artifact(args.model_path)
    configure_adapter_environment(artifact)
    if artifact.artifact_type == 'lora_adapter':
        import peft  # fail before launching subprocesses if dependency is missing
        args.model_path = artifact.load_path
        print('[INFO] OpenCompass base=%s adapter=%s' % (artifact.load_path, artifact.adapter_path), flush=True)


def run_standard_evaluation(args):
    if not is_standard_dataset(args.custom_dataset_path):
        return False
    for field in ('custom_columns','custom_metric','custom_prompt_template'):
        if getattr(args, field, ''):
            raise ValueError('%s 不适用于 Convert 标准 eval_qa；请留空，题型决定评分方式' % field)
    if args.few_shot:
        raise ValueError('标准 eval_qa 当前仅支持 few_shot=0')
    if min(args.num_gpus, args.max_seq_len, args.max_out_len, args.batch_size) < 1 or args.max_samples < 0:
        raise ValueError('GPU/长度/batch_size 必须为正数，max_samples 不得为负')
    kwargs = json.loads(args.model_kwargs) if args.model_kwargs else {}
    if not isinstance(kwargs, dict):
        raise ValueError('model_kwargs 必须是 JSON 对象')
    # Reuse master's GPU selection and availability checks.
    from run_evaluation import apply_num_gpus_visibility, check_gpu_available
    apply_num_gpus_visibility(args.num_gpus)
    check_gpu_available(args.num_gpus)
    artifact = resolve_model_artifact(args.model_path)
    # Native loading applies PEFT itself, not the OpenCompass subprocess hook.
    os.environ.pop('MLOPS_LORA_ADAPTER_PATH', None)
    output = Path(args.output_path).expanduser().resolve()
    source = Path(args.custom_dataset_path).expanduser().resolve()
    if output == source or output in source.parents:
        raise ValueError('评测输出目录不能等于或包含输入数据路径')
    output.mkdir(parents=True, exist_ok=True)
    status = output / 'task_status.json'
    _write_json(status, {'status':'running','model':artifact.to_dict()})
    try:
        summary = evaluate_custom_dataset(
            args.custom_dataset_path, str(output), artifact,
            max_samples=args.max_samples, max_seq_len=args.max_seq_len,
            max_out_len=args.max_out_len, model_type=args.model_type, model_kwargs=kwargs,
            batch_size=args.batch_size)
        from parse_and_save import save_metric_json, save_summary_json, save_report_csv
        report = {'custom_eval': {'status':'succeeded', 'metric':'pass_rate',
                                  'score':round(summary['pass_rate']*100, 4),
                                  'subsets':{}}}
        metadata = {'model_name':args.model_name,'model_version':args.model_version,
                    'model_path':args.model_path,'datasets':['custom_eval']}
        save_metric_json(str(output), report, metadata)
        save_summary_json(str(output), report)
        save_report_csv(str(output), report)
        metric_link = Path('/metric.json')
        if metric_link.is_symlink():
            metric_link.unlink()
        if not metric_link.exists():
            metric_link.symlink_to(output/'metric.json')
        _write_json(status, {'status':'success','model':artifact.to_dict(), 'summary':summary})
        print('[OK] 标准 eval_qa 评测完成: %s' % output, flush=True)
        return True
    except Exception as exc:
        _write_json(status, {'status':'failed','error':str(exc),'model':artifact.to_dict()})
        raise
