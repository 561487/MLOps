#!/usr/bin/env python3
"""
litgpt 模型预训练任务启动脚本
===============================
Pipeline 表单参数 → argparse 解析 → 生成 litgpt 训练配置 (YAML) → litgpt pretrain
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def has_cuda():
    """Check if CUDA GPU is available from within the container."""
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if devices in ("", "-1", "none", "None"):
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def detect_num_gpus():
    """Return number of *visible* GPUs (respects CUDA_VISIBLE_DEVICES)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except ImportError:
        pass
    return 1


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="litgpt 模型预训练任务")
    # ── 模型配置 ──
    p.add_argument("--model_name", type=str, default="",
                   help="litgpt 内置配置名 (如 tiny-llama-1.1b, llama-3.2-1b)")
    p.add_argument("--tokenizer_dir", type=str, default="",
                   help="tokenizer HF repo 或本地路径；留空则从 model_name 下载")
    p.add_argument("--precision", type=str, default="bf16-true",
                   choices=["bf16-true", "bf16-mixed", "32-true"],
                   help="bf16-true(推荐H100/A100), bf16-mixed, 32-true(慢)")

    # ── 数据配置 ──
    p.add_argument("--data_path", type=str, default="",
                   help="存放 .txt 文件的目录 (litgpt.data.TextFiles)")
    p.add_argument("--max_seq_length", type=int, default=4096,
                   help="最大序列长度 (token)，范围 512~131072")

    # ── 训练配置 ──
    p.add_argument("--max_tokens", type=str, default="100M",
                   help="最大训练 token 数 (如 100M, 1B, 10B)，litgpt 以此计算终止条件")
    p.add_argument("--learning_rate", type=str, default="4e-4",
                   help="学习率")
    p.add_argument("--micro_batch_size", type=int, default=4,
                   help="每 GPU 批次大小")
    p.add_argument("--global_batch_size", type=int, default=64,
                   help="全局批次大小，必须是 micro_batch_size × GPU 数的整数倍")
    p.add_argument("--weight_decay", type=str, default="1e-2",
                   help="AdamW 权重衰减")
    p.add_argument("--warmup_steps", type=int, default=2000,
                   help="学习率预热步数，必须 < max_tokens / global_batch_size / seq_len")
    p.add_argument("--max_norm", type=float, default=1.0,
                   help="梯度裁剪值，0=不裁剪")
    p.add_argument("--save_interval", type=int, default=1000,
                   help="checkpoint 保存间隔(步)")
    p.add_argument("--log_interval", type=int, default=10,
                   help="日志打印间隔(步)")

    # ── 输出配置 ──
    p.add_argument("--output_dir", type=str, default="",
                   help="输出目录，不填则自动生成")
    p.add_argument("--resume_from", type=str, default="",
                   help="恢复训练的 checkpoint 路径")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_output_dir(user_dir):
    if user_dir:
        os.makedirs(user_dir, exist_ok=True)
        return user_dir
    creator = os.environ.get("KFJ_CREATOR", "admin")
    pipe = os.environ.get("KFJ_PIPELINE_NAME", "default")
    task = os.environ.get("KFJ_TASK_NAME", "litgpt-pretrain")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = f"/mnt/{creator}/pipeline/{pipe}/{task}-{ts}"
    os.makedirs(d, exist_ok=True)
    return d


def _parse_token_count(s):
    """'100M' / '1B' / '10B' → int"""
    if not s or not s.strip():
        return None
    s = s.strip().upper()
    m = re.match(r'^([\d.]+)\s*([BMKT]?)$', s)
    if not m:
        return None
    val = float(m.group(1))
    mult = {"B": 1e9, "M": 1e6, "K": 1e3, "T": 1e12}.get(m.group(2), 1)
    return int(val * mult)


def _resolve_tokenizer(tokenizer_dir, model_name, cache_dir):
    """Return a local directory containing tokenizer files.
    Downloads from HF if a repo-id is given.
    """
    if not tokenizer_dir:
        # If model_name looks like an HF repo, use it for tokenizer download
        if model_name and "/" in model_name:
            tokenizer_dir = model_name
        elif model_name:
            # Resolve litgpt built-in name → HF repo (e.g. tiny-llama-1.1b → TinyLlama/TinyLlama-1.1B-…)
            try:
                from litgpt.config import name_to_config
                hf = name_to_config[model_name].get("hf_config", {})
                org, name = hf.get("org", ""), hf.get("name", "")
                tokenizer_dir = f"{org}/{name}" if org and name else None
            except (KeyError, ImportError):
                pass
        if not tokenizer_dir:
            return None

    # Already a local directory
    local = os.path.abspath(os.path.expanduser(tokenizer_dir))
    if os.path.isdir(local):
        return local

    # Looks like HF repo (contains '/') → download
    if "/" in tokenizer_dir:
        dest = os.path.join(cache_dir, "tokenizer")
        os.makedirs(dest, exist_ok=True)
        print(f"[start.py] Downloading tokenizer from {tokenizer_dir} …")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=tokenizer_dir,
                local_dir=dest,
                allow_patterns=[
                    "tokenizer.*", "vocab.*", "merges.*",
                    "special_tokens_map.json", "tokenizer_config.json",
                ],
                max_workers=4,
            )
            print(f"[start.py] Tokenizer saved → {dest}")
            return dest
        except Exception as e:
            print(f"[start.py] WARNING: tokenizer download failed ({e})")
            return tokenizer_dir  # let litgpt try its own resolution

    return tokenizer_dir


def _validate_batch_size(global_bs, micro_bs, num_gpus):
    """LitGPT requires global_batch_size to be divisible by micro_batch_size * num_gpus."""
    step = micro_bs * num_gpus
    if global_bs % step != 0:
        print(f"[start.py] WARNING: global_batch_size ({global_bs}) is not a multiple of "
              f"micro_batch_size ({micro_bs}) × GPU_count ({num_gpus}) = {step}")
        # Auto-fix: round up to nearest multiple
        new_bs = ((global_bs + step - 1) // step) * step
        print(f"[start.py] Auto-correcting global_batch_size → {new_bs}")
        return new_bs
    return global_bs


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def build_config(args, output_dir, num_gpus):
    tk = _resolve_tokenizer(
        args.tokenizer_dir.strip() if args.tokenizer_dir else "",
        args.model_name,
        output_dir,
    )

    # Validate & correct batch size
    gbs = _validate_batch_size(args.global_batch_size, args.micro_batch_size, num_gpus)

    max_tokens_val = _parse_token_count(args.max_tokens) or _parse_token_count("100M")

    train = {
        "max_seq_length": args.max_seq_length,
        "micro_batch_size": args.micro_batch_size,
        "global_batch_size": gbs,
        "max_tokens": max_tokens_val,
        "lr_warmup_steps": args.warmup_steps,
        "save_interval": args.save_interval,
        "log_interval": args.log_interval,
    }
    if args.max_norm > 0:
        train["max_norm"] = args.max_norm

    cfg = {
        "model_name": args.model_name or None,
        "tokenizer_dir": tk,
        "out_dir": output_dir,
        "precision": args.precision,
        "devices": num_gpus,
        "logger_name": "csv",
        "data": {
            "class_path": "litgpt.data.TextFiles",
            "init_args": {
                "train_data_path": args.data_path,
            },
        },
        "train": train,
        "optimizer": {
            "class_path": "torch.optim.AdamW",
            "init_args": {
                "lr": float(args.learning_rate),
                "weight_decay": float(args.weight_decay),
            },
        },
    }

    if args.resume_from:
        cfg["resume"] = args.resume_from

    # Strip None-valued keys recursively
    return _strip_nones(cfg)


def _strip_nones(obj):
    if isinstance(obj, dict):
        return {k: _strip_nones(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_nones(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def write_config_yaml(cfg, path):
    import yaml
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"[start.py] Config → {path}")
    with open(path) as f:
        print(f.read())


def run_training(config_path):
    cmd = ["litgpt", "pretrain", "--config", config_path]
    print(f"[start.py] {' '.join(cmd)}")
    sys.stdout.flush()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="", flush=True)
    return proc.wait()


def write_metadata(args, output_dir, exit_code):
    meta = {k: v for k, v in vars(args).items()}
    meta["exit_code"] = exit_code
    meta["timestamp"] = datetime.now().isoformat()
    path = os.path.join(output_dir, "meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)



def _ensure_data(data_path):
    """If data_path exists but is empty, create minimal test data so training can start."""
    if not data_path or not os.path.isdir(data_path):
        return
    txts = [f for f in os.listdir(data_path) if f.endswith(".txt")]
    if txts:
        return  # already has data
    # Create one .txt file with sample text for a minimal test run
    sample = os.path.join(data_path, "sample.txt")
    text = (
        "Deep learning uses neural networks with multiple layers to learn from data.\n"
        "Large language models are trained on vast corpora of text.\n"
        "The transformer architecture uses self-attention for NLP tasks.\n"
        "Pretraining learns representations from unlabeled data.\n"
        "Optimization algorithms like AdamW help training converge faster.\n"
        "Gradient descent updates parameters to minimize the loss function.\n"
        "Tokenization converts raw text into numerical tokens for models.\n"
        "Attention mechanisms focus on relevant parts of the input.\n"
        "Transfer learning adapts pretrained models to downstream tasks.\n"
        "Neural networks transform input data through nonlinear functions.\n"
        "Dropout regularization prevents overfitting during training.\n"
        "Layer normalization stabilizes training across features.\n"
        "Convolutional networks excel at processing images and videos.\n"
        "Recurrent networks process sequential data with hidden states.\n"
        "Positional encodings provide order information to transformers.\n"
        "Multi-head attention allows diverse representation subspaces.\n"
        "Subword tokenization handles rare words efficiently.\n"
        "Mixed precision training speeds up training with float16.\n"
        "Learning rate scheduling uses cosine decay or warmup strategies.\n"
        "Perplexity evaluates language model prediction quality.\n"
        "Temperature sampling controls randomness of generated text.\n"
        "Knowledge distillation transfers knowledge to smaller models.\n"
        "Model quantization reduces size by using fewer bits.\n"
        "Backpropagation computes gradients using the chain rule.\n"
        "Residual connections help gradient flow in deep networks.\n"
        "Reinforcement learning maximizes cumulative rewards.\n"
        "Unsupervised learning discovers patterns without labels.\n"
        "The softmax function converts outputs to probabilities.\n"
        "Cross-entropy measures difference between predictions and labels.\n"
        "The LLaMA model family includes various parameter sizes.\n"
    ) * 3  # repeat 3x for enough tokens
    with open(sample, "w") as f:
        f.write(text)
    print(f"[start.py] auto-created test data: {sample} ({os.path.getsize(sample)} bytes)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  litgpt 模型预训练 - 启动")
    print("=" * 60)

    # Monitor (optional)
    try:
        from common.training_monitor import TrainingMonitor
        monitor = TrainingMonitor()
    except ImportError:
        monitor = None

    final_status = "FAILED"
    exit_code = 1

    try:
        # Env
        for k in ("KFJ_CREATOR", "KFJ_PIPELINE_NAME", "KFJ_TASK_NAME",
                   "HF_ENDPOINT", "CUDA_VISIBLE_DEVICES"):
            print(f"[start.py] {k}: {os.environ.get(k, '')}")

        # Parse
        args = parse_args()
        for k, v in vars(args).items():
            print(f"    --{k}: {v}")

        if not args.model_name:
            sys.exit("[start.py] ERROR: --model_name is required")
        if not args.data_path:
            sys.exit("[start.py] ERROR: --data_path is required")

        # Ensure data dir isn't empty
        _ensure_data(args.data_path)

        # GPU
        cuda = has_cuda()
        ngpu = detect_num_gpus() if cuda else 1
        print(f"[start.py] CUDA: {cuda}   visible GPUs: {ngpu}")

        # Output dir
        out = resolve_output_dir(args.output_dir)
        print(f"[start.py] out_dir: {out}")

        # Config
        cfg = build_config(args, out, ngpu)
        cp = os.path.join(out, "litgpt_config.yaml")
        write_config_yaml(cfg, cp)

        # Run
        if monitor:
            monitor.start()
        exit_code = run_training(cp)

        final_status = "SUCCEEDED" if exit_code == 0 else "FAILED"
        write_metadata(args, out, exit_code)

        print("=" * 60)
        print(f"  {'SUCCEEDED ✅' if exit_code == 0 else 'FAILED ❌'}   out: {out}")
        print("=" * 60)

    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        if monitor:
            try:
                monitor.finish(final_status)
            except Exception:
                pass

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
