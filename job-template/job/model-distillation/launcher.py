"""
模型蒸馏节点入口脚本
使用 EasyDistill train_only 模式（跳过 vllm Teacher 推理，直接训练）

白盒蒸馏: launcher 预计算 Teacher logits → easydistill kd_white_box_train_only
黑盒蒸馏: easydistill kd_black_box_train_only（纯 SFT）
"""
import os
# 禁用所有进度条（避免 ANSI 转义码污染在线日志界面）
for _key in ('TQDM_DISABLE', 'HF_HUB_DISABLE_PROGRESS_BARS', 'DATASETS_DISABLE_PROGRESS_BARS'):
    os.environ[_key] = '1'

import argparse
import json
import subprocess
import datetime
import sys
import glob

KFJ_CREATOR = os.getenv('KFJ_CREATOR', 'admin')
KFJ_RUN_ID = os.getenv('KFJ_RUN_ID', '')

SENTINEL_FILE = '.distillation_done'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOP_K_LOGITS = int(os.getenv('DISTILL_TOP_K', '1000'))  # 每个 token 保留的 top-k 概率


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def check_skip(output_path):
    sentinel_path = os.path.join(output_path, SENTINEL_FILE)
    if os.path.isfile(sentinel_path):
        print(f'蒸馏结果已存在: {output_path}')
        return True
    return False


# 常见 ModelScope 命名空间，用于从本地路径反查模型 ID
_MODELSCOPE_NAMESPACES = ['Qwen', 'qwen', 'deepseek-ai', 'damo', 'llava']


def resolve_model_path(model_name, cache_dir):
    """ModelScope ID → 本地路径；已是本地路径直接返回

    支持两种输入:
    1. ModelScope ID（如 Qwen/Qwen2.5-0.5B-Instruct）: 下载到 cache_dir
    2. 本地路径（如 /mnt/models/Qwen2.5-0.5B-Instruct）: 直接用；不存在则尝试反查 ModelScope ID
    """
    if os.path.isdir(model_name):
        print(f'使用本地模型: {model_name}')
        return model_name

    # 如果看起来像本地路径（以 / ./ ../ 开头）但目录不存在，提取模型名重试
    if model_name.startswith('/') or model_name.startswith('./') or model_name.startswith('../'):
        model_basename = os.path.basename(model_name.rstrip('/'))
        print(f'本地模型不存在: {model_name}，尝试从 ModelScope 下载')
        from modelscope import snapshot_download
        for ns in _MODELSCOPE_NAMESPACES:
            model_id = f'{ns}/{model_basename}'
            try:
                print(f'  尝试: {model_id}')
                local_dir = snapshot_download(model_id, cache_dir=cache_dir)
                print(f'模型下载完成: {local_dir}')
                return local_dir
            except Exception:
                continue
        raise FileNotFoundError(
            f'本地模型路径不存在且无法从 ModelScope 自动下载: {model_name}\n'
            f'请确保上游 "模型导入" 步骤已成功执行，或手动提供有效的 ModelScope ID。')

    # 纯 ModelScope ID（如 Qwen/Qwen2.5-0.5B-Instruct）
    print(f'从 ModelScope 下载模型: {model_name}')
    from modelscope import snapshot_download
    local_dir = snapshot_download(model_name, cache_dir=cache_dir)
    print(f'模型下载完成: {local_dir}')
    return local_dir


def resolve_data_path(data_path):
    """解析数据路径，支持目录（取第一个 jsonl/json/parquet/csv）"""
    if os.path.isdir(data_path):
        files = glob.glob(os.path.join(data_path, '*.jsonl')) + \
                glob.glob(os.path.join(data_path, '*.json')) + \
                glob.glob(os.path.join(data_path, '*.parquet')) + \
                glob.glob(os.path.join(data_path, '*.csv'))
        return files[0] if files else data_path
    return data_path


def detect_data_format(data_path):
    """根据扩展名推断数据格式"""
    ext = os.path.splitext(data_path)[1].lower()
    if ext in ('.jsonl', '.json'):
        return 'json'
    elif ext == '.parquet':
        return 'parquet'
    elif ext == '.csv':
        return 'csv'
    else:
        return 'json'  # 默认按 JSON 尝试


SYSTEM_PROMPT = "You are a helpful assistant."

# 支持的列名映射（优先匹配前面的）
INSTRUCTION_COLUMNS = ['instruction', 'question', 'prompt', 'input', 'query', 'text', 'content']
OUTPUT_COLUMNS = ['output', 'answer', 'response', 'completion', 'target', 'label', 'text_output']


def auto_detect_columns(dataset):
    """自动检测 instruction/output 对应的列名"""
    cols = list(dataset.features.keys())
    inst_col = next((c for c in INSTRUCTION_COLUMNS if c in cols), cols[0])
    # output 列：找一个不是 instruction 的匹配列
    out_col = next((c for c in OUTPUT_COLUMNS if c in cols and c != inst_col),
                   cols[1] if len(cols) > 1 else cols[0])
    print(f'列名映射: instruction → "{inst_col}", output → "{out_col}"')
    return inst_col, out_col


def format_sample_with_tokenizer(tokenizer, instruction, output):
    """使用 tokenizer 自带的 chat_template 格式化样本（自动适配 Qwen/Llama/ChatGLM 等）"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


# ---------------------------------------------------------------------------
# Teacher Logits 预计算（白盒蒸馏核心）
# ---------------------------------------------------------------------------

def generate_teacher_logits(teacher_path, data_path, logits_path, max_length=512, max_samples=0):
    """
    预计算 Teacher 模型的概率分布（替代 easydistill kd/infer.py，不依赖 vllm）

    使用 tokenizer.apply_chat_template() 自动适配 Qwen/Llama/ChatGLM 等架构

    对每条训练数据:
      1. tokenizer.apply_chat_template 格式化
      2. Teacher 模型前向推理 → logits
      3. softmax → 概率分布
      4. 保留 top-k + 重新归一化 → 稀疏格式写入 JSONL
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    import jsonlines

    print('===== 预计算 Teacher Logits =====')
    print(f'Teacher: {teacher_path}')
    print(f'数据:   {data_path}')
    print(f'max_length: {max_length}, top_k: {TOP_K_LOGITS}')

    # 加载 tokenizer（自带 chat_template）
    tokenizer = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f'Chat template: {tokenizer.chat_template[:80] if tokenizer.chat_template else "无（将用默认格式）"}...')

    # 加载 Teacher 模型 (bf16 省显存)
    print('加载 Teacher 模型...')
    model = AutoModelForCausalLM.from_pretrained(
        teacher_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    teacher_vocab_size = model.config.vocab_size
    print(f'Teacher vocab_size: {teacher_vocab_size}')

    # 加载数据（自动检测 JSON/JSONL/Parquet 格式）
    data_format = detect_data_format(data_path)
    dataset = load_dataset(data_format, data_files=data_path, split="train")
    if max_samples > 0 and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
    print(f'训练样本数: {len(dataset)}')
    inst_col, out_col = auto_detect_columns(dataset)

    # 逐条处理
    print(f'开始生成 logits → {logits_path}')
    import time as _time
    skipped = 0
    with jsonlines.open(logits_path, 'w') as writer:
        for idx, example in enumerate(dataset):
            try:
                instruction = example.get(inst_col, "")
                output = example.get(out_col, "")

                # 使用 tokenizer 自带的 chat_template 格式化（自动适配架构）
                text = format_sample_with_tokenizer(tokenizer, instruction, output)
                inputs = tokenizer(
                    text, return_tensors="pt",
                    max_length=max_length, truncation=True, padding=False,
                )
                seq_len = inputs["input_ids"].shape[1]

                # 进度提前打印（含 seq_len），方便定位卡死样本
                if (idx + 1) % 10 == 0 or idx == 0:
                    print(f'  进度: {idx + 1}/{len(dataset)} (seq_len={seq_len})')

                _t0 = _time.time()
                # 前向推理
                with torch.no_grad():
                    outputs = model(
                        input_ids=inputs["input_ids"].to(model.device),
                        attention_mask=inputs["attention_mask"].to(model.device),
                    )
                    logits = outputs.logits[0]  # [seq_len, vocab_size]

                # softmax → 概率，保留 top-k 并重新归一化（向量化：一次 topk 处理所有位置）
                probs = torch.softmax(logits.float(), dim=-1)  # [seq_len, vocab_size]
                k = min(TOP_K_LOGITS, teacher_vocab_size)
                topk_vals, topk_ids = torch.topk(probs, k=k, dim=-1)  # [seq_len, k]
                topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)  # 重新归一化
                sample_data = []
                for pos in range(seq_len):
                    pos_dict = {int(tid): float(tv) for tid, tv in zip(topk_ids[pos].cpu(), topk_vals[pos].cpu())}
                    sample_data.append(pos_dict)

                writer.write(sample_data)
                _elapsed = _time.time() - _t0
                if _elapsed > 30:
                    print(f'  [慢] 样本 {idx + 1}: seq_len={seq_len}, 耗时={_elapsed:.1f}s')

            except Exception as e:
                skipped += 1
                print(f'  [跳过] 样本 {idx + 1} 处理失败: {e}')
                continue

    # 清理 Teacher 显存
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f'Teacher logits 生成完成: {logits_path}')
    return teacher_vocab_size


# ---------------------------------------------------------------------------
# EasyDistill 配置构建
# ---------------------------------------------------------------------------

def build_config(args, teacher_path, student_path, data_path, work_dir, template_path,
                 inst_col='instruction', out_col='output', data_format='json'):
    """将命令行参数转换为 EasyDistill 配置 JSON"""

    distill_type = args.distill_type
    job_type = 'kd_white_box_train_only' if distill_type == 'whitebox' else 'kd_black_box_train_only'

    config = {
        "job_type": job_type,
        "models": {
            "teacher": teacher_path,
            "student": student_path,
        },
        "dataset": {
            "data_format": data_format,
            "instruction_path": data_path,
            "labeled_path": data_path,
            "template": template_path,
            "seed": 42,
            "max_samples": int(args.max_samples) if args.max_samples else 0,
            "inst_col": inst_col,
            "out_col": out_col,
        },
        "training": {
            "output_dir": args.output_path,
            "num_train_epochs": int(args.epochs),
            "per_device_train_batch_size": int(args.batch_size),
            "gradient_accumulation_steps": 8,
            "max_length": 512,
            "save_steps": 1000,
            "logging_steps": 1,
            "learning_rate": float(args.learning_rate),
            "weight_decay": 0.05,
            "warmup_ratio": 0.1,
            "lr_scheduler_type": "cosine",
        }
    }

    if distill_type == 'whitebox':
        logits_path = os.path.join(work_dir, 'teacher_logits.jsonl')
        config["dataset"]["logits_path"] = logits_path
        kd_ratio = round(1.0 - float(args.alpha), 2)
        config["distillation"] = {
            "kd_ratio": kd_ratio,
            "max_seq_length": 512,
            "distillation_type": "forward_kld",
        }

    if args.data_synthesis == 'true':
        config["dataset"]["enable_synthesis"] = True

    return config


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    arg_parser = argparse.ArgumentParser("model distillation launcher")
    arg_parser.add_argument('--teacher_model', type=str, required=True)
    arg_parser.add_argument('--student_model', type=str, required=True)
    arg_parser.add_argument('--output_path', type=str, required=True)
    arg_parser.add_argument('--distill_type', type=str, default='whitebox',
                            choices=['whitebox', 'blackbox'])
    arg_parser.add_argument('--data_path', type=str, required=True)
    arg_parser.add_argument('--max_samples', type=str, default='0')
    arg_parser.add_argument('--data_synthesis', type=str, default='false')
    arg_parser.add_argument('--temperature', type=str, default='4.0')
    arg_parser.add_argument('--alpha', type=str, default='0.5')
    arg_parser.add_argument('--epochs', type=str, default='3')
    arg_parser.add_argument('--batch_size', type=str, default='4')
    arg_parser.add_argument('--learning_rate', type=str, default='2e-5')

    args = arg_parser.parse_args()

    # ----- 1. 跳过检查 -----
    if check_skip(args.output_path):
        return

    try:
        output_dir = args.output_path.rstrip('/')
        cache_dir = os.path.join(os.path.dirname(output_dir), '.modelscope_cache')
        work_dir = os.path.join(os.path.dirname(output_dir), '.distill_work')
        os.makedirs(work_dir, exist_ok=True)
        template_path = os.path.join(SCRIPT_DIR, 'chat_template', 'chat_template_kd.jinja')

        # ----- 2. 下载/解析模型 + 数据（只做一次） -----
        print('===== 准备模型和数据 =====')
        teacher_path = resolve_model_path(args.teacher_model, cache_dir)
        student_path = resolve_model_path(args.student_model, cache_dir)
        data_path = resolve_data_path(args.data_path)
        print(f'Teacher: {teacher_path}')
        print(f'Student: {student_path}')
        print(f'Data:    {data_path}')

        # ----- 3. 先检测列名（加载一次数据集） -----
        from datasets import load_dataset as _load_ds
        _format = detect_data_format(data_path)
        _ds = _load_ds(_format, data_files=data_path, split="train")
        inst_col, out_col = auto_detect_columns(_ds)
        print(f'列名映射: instruction → "{inst_col}", output → "{out_col}"')

        # ----- 3.5 截断数据到 max_samples（logits 和训练共用） -----
        max_samples = int(args.max_samples) if args.max_samples else 0
        if max_samples > 0 and len(_ds) > max_samples:
            _ds = _ds.select(range(max_samples))
        limited_data_path = os.path.join(work_dir, 'limited_train.jsonl')
        _ds.to_json(limited_data_path, force_ascii=False)
        print(f'截断后样本数: {len(_ds)} → {limited_data_path}')
        del _ds

        # ----- 4. 白盒: 预计算 Teacher logits -----
        if args.distill_type == 'whitebox':
            logits_path = os.path.join(work_dir, 'teacher_logits.jsonl')
            generate_teacher_logits(
                teacher_path=teacher_path,
                data_path=limited_data_path,
                logits_path=logits_path,
                max_length=512,
                max_samples=0,  # 数据已在 limited_data_path 中截断
            )

        # ----- 5. 构建 EasyDistill 配置 -----
        config = build_config(args, teacher_path, student_path, limited_data_path, work_dir, template_path,
                              inst_col=inst_col, out_col=out_col, data_format='json')  # limited_train.jsonl 始终为 json 格式

        config_path = os.path.join(work_dir, 'easydistill_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f'EasyDistill 配置:\n{json.dumps(config, indent=2, ensure_ascii=False)}')

        # ----- 5. 执行蒸馏 -----
        cmd = ['easydistill', '--config', config_path]
        print(f'执行: {" ".join(cmd)}')

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
        exit_code = process.wait()

        if exit_code != 0:
            print(f'蒸馏失败，exit code={exit_code}')
            sys.exit(exit_code)

        # ----- 6. ModelScope 兼容 + 哨兵 -----
        config_json = os.path.join(output_dir, 'config.json')
        ms_config_json = os.path.join(output_dir, 'configuration.json')
        if os.path.exists(config_json):
            import shutil
            shutil.copy2(config_json, ms_config_json)

        sentinel_path = os.path.join(output_dir, SENTINEL_FILE)
        os.makedirs(output_dir, exist_ok=True)
        with open(sentinel_path, 'w') as f:
            f.write(f'distilled at {datetime.datetime.now().isoformat()}\n'
                    f'teacher={args.teacher_model}\nstudent={args.student_model}\n')
        print(f'蒸馏完成，模型保存至: {output_dir}')

    except Exception as e:
        print(f'蒸馏失败: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
