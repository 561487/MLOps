#!/usr/bin/env python3
"""
llama_factory LoRA 微调任务启动脚本
===================================

接收流水线表单传入的参数，生成 LLaMAFactory 训练配置（YAML），
调用 llamafactory-cli train 执行微调，模型输出保存到共享存储。

参数传递链路：
  流水线表单 → K8s Pod command args → argparse解析 → 生成config.yaml → llamafactory-cli train
"""

import argparse
import json
import os
import subprocess
import sys
import textwrap
from datetime import datetime


def has_cuda():
    """检测是否有可用的 CUDA GPU（不需要 import torch）"""
    import subprocess, os
    # 方式1：检查 nvidia-smi
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return True
    except Exception:
        pass
    # 方式2：检查环境变量
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        return True
    return False


def parse_args():
    """解析流水线表单传入的参数，与 init-job-template.json 的 job_template_args 一一对应"""
    parser = argparse.ArgumentParser(
        description="llama_factory LoRA 微调任务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            参数来源：流水线编辑器 → job_template_args
            框架文档：https://github.com/hiyouga/LLaMA-Factory
        """),
    )

    # === 模型参数 ===
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="预训练模型名称或路径（HuggingFace 模型名或本地路径）")

    # === 数据集参数 ===
    parser.add_argument("--dataset", type=str, default="",
                        help="训练数据集，支持多个用逗号分隔")
    parser.add_argument("--cutoff_len", type=int, default=2048,
                        help="截断长度，根据显存调整")

    # === LoRA 参数 ===
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA 秩，通常 8 或 16")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha，通常是 rank 的 2 倍")
    parser.add_argument("--lora_target", type=str, default="",
                        help="LoRA 应用的模块，如 q_proj,v_proj")

    # === 训练参数 ===
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="学习率")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="每设备批次大小")
    parser.add_argument("--num_train_epochs", type=float, default=3.0,
                        help="训练轮数")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="梯度累积步数")

    return parser.parse_args()


def resolve_output_dir():
    """解析模型输出目录，优先使用共享存储路径"""
    creator = os.environ.get("KFJ_CREATOR", "admin")
    pipeline_name = os.environ.get("KFJ_PIPELINE_NAME", "default")
    task_name = os.environ.get("KFJ_TASK_NAME", "qwen3-lora")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"/mnt/{creator}/pipeline/{pipeline_name}/{task_name}-{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"[start.py] 模型输出目录: {output_dir}")
    return output_dir


def build_config(args, output_dir):
    """构建 LLaMAFactory 训练配置文件"""

    # 检测设备
    use_cuda = has_cuda()
    if use_cuda:
        print("[start.py] 检测到 CUDA GPU，使用 GPU 加速")
    else:
        print("[start.py] 未检测到 CUDA GPU，使用 CPU 训练（速度较慢）")

    # 基础配置
    config = {
        # 模型
        "model_name_or_path": args.model_name,

        # 方法
        "finetuning_type": "lora",
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": 0.1,

        # 数据集（使用内置数据集，不指定 dataset_dir，LLaMAFactory 自动加载自带的 data/）
        "template": "default",
        "cutoff_len": args.cutoff_len,

        # 训练
        "do_train": True,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "logging_steps": 1,
        "save_steps": 2,
        "save_total_limit": 2,

        # 精度 / 设备
        "bf16": use_cuda,
        "fp16": False,

        # 输出
        "output_dir": output_dir,
        "overwrite_output_dir": True,
        "report_to": "none",
    }

    # 可选：数据集
    if args.dataset:
        config["dataset"] = args.dataset

    # 可选：LoRA target modules
    if args.lora_target:
        config["lora_target"] = args.lora_target.split(",")

    return config


def write_config_yaml(config, path):
    """将配置写入 YAML 文件"""
    import yaml
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"[start.py] 训练配置已写入: {path}")
    with open(path, "r") as f:
        print(f.read())


def install_llamafactory():
    """安装 LLaMAFactory 框架"""
    print("[start.py] 检查/安装 LLaMAFactory...")
    sys.stdout.flush()
    # 先试 import，成功则不安装
    try:
        import llamafactory  # noqa: F401
        print("[start.py] LLaMAFactory 已安装，跳过")
        return True
    except ImportError:
        pass
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-U", "llamafactory", "pyyaml", "email-validator"],
    )
    if result.returncode != 0:
        print(f"[start.py] pip install 失败，退出码: {result.returncode}")
        return False
    print("[start.py] LLaMAFactory 安装完成")
    return True


def run_training(config_path):
    """执行 LLaMAFactory 训练"""
    cmd = ["llamafactory-cli", "train", config_path]
    print(f"[start.py] 执行命令: {' '.join(cmd)}")
    sys.stdout.flush()

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end="", flush=True)
    return process.wait()


def merge_lora(model_name_or_path, adapter_path, output_dir):
    """合并 LoRA 适配器到基础模型（使用 PEFT Python API）"""
    merged_dir = os.path.join(output_dir, "merged_model")
    os.makedirs(merged_dir, exist_ok=True)

    print(f"[start.py] 加载基础模型: {model_name_or_path}")
    sys.stdout.flush()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype="auto",
    )
    # 加载 LoRA 适配器
    print(f"[start.py] 加载 LoRA 适配器: {adapter_path}")
    sys.stdout.flush()
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # 合并
    print("[start.py] 合并 LoRA 到基础模型...")
    sys.stdout.flush()
    merged_model = model.merge_and_unload()

    # 保存
    print(f"[start.py] 保存合并后模型到: {merged_dir}")
    sys.stdout.flush()
    merged_model.save_pretrained(merged_dir)

    # 保存分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.save_pretrained(merged_dir)

    print(f"[start.py] 合并完成 ✅")
    sys.stdout.flush()
    return 0, merged_dir


def main():
    print("=" * 60)
    print("  llama_factory LoRA 微调任务 - 启动")
    print("=" * 60)

    # 1. 打印环境信息
    print(f"[start.py] 工作目录: {os.getcwd()}")
    print(f"[start.py] Python: {sys.version}")
    for key in ["KFJ_CREATOR", "KFJ_PIPELINE_NAME", "KFJ_TASK_NAME", "HF_ENDPOINT"]:
        print(f"[start.py] {key}: {os.environ.get(key, '(未设置)')}")

    # 2. 解析参数
    args = parse_args()
    print(f"[start.py] 解析参数:")
    for k, v in vars(args).items():
        print(f"    --{k}: {v}")

    # 3. 解析输出目录
    output_dir = resolve_output_dir()

    # 4. 构建训练配置
    config = build_config(args, output_dir)

    # 5. 安装 LLamaFactory
    if not install_llamafactory():
        sys.exit(1)

    # 5b. 确保 data/dataset_info.json 和 data/identity.json 存在（LLaMAFactory 需要）
    data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(data_dir, exist_ok=True)

    # dataset_info.json
    dataset_info_path = os.path.join(data_dir, "dataset_info.json")
    if not os.path.exists(dataset_info_path):
        default_dataset_info = {
            "identity": {
                "file_name": "identity.json",
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output"
                }
            }
        }
        with open(dataset_info_path, "w", encoding="utf-8") as f:
            json.dump(default_dataset_info, f, ensure_ascii=False, indent=2)
        print(f"[start.py] 已创建 {dataset_info_path}")

    # identity.json（无论 dataset_info.json 是否已存在都要保证有）
    identity_path = os.path.join(data_dir, "identity.json")
    if not os.path.exists(identity_path):
        identity_data = [
            {"instruction": "介绍一下你自己", "input": "", "output": "我是 Qwen，一个由阿里云开发的大语言模型。"},
            {"instruction": "Who are you?", "input": "", "output": "I am Qwen, a large language model developed by Alibaba Cloud."},
            {"instruction": "你好", "input": "", "output": "你好！我是 Qwen，有什么可以帮助你的吗？"},
            {"instruction": "Hello", "input": "", "output": "Hello! I am Qwen, how can I help you?"},
        ]
        with open(identity_path, "w", encoding="utf-8") as f:
            json.dump(identity_data, f, ensure_ascii=False, indent=2)
        print(f"[start.py] 已创建 {identity_path}")

    config["dataset_dir"] = data_dir

    # 写入配置（在补充好所有字段后）
    config_path = os.path.join(output_dir, "train_config.yaml")
    write_config_yaml(config, config_path)

    # 6. 执行训练
    exit_code = run_training(config_path)

    # 7. 训练完成后合并 LoRA 到基础模型
    merged_dir = None
    if exit_code == 0:
        print("=" * 60)
        print("  开始合并 LoRA 到基础模型...")
        print("=" * 60)
        
        adapter_path = output_dir  # LoRA 适配器保存在 output_dir
        merge_exit_code, merged_dir = merge_lora(args.model_name, adapter_path, output_dir)
        
        if merge_exit_code == 0:
            print("=" * 60)
            print(f"  LoRA 合并完成 ✅  合并后模型: {merged_dir}")
            print("=" * 60)
        else:
            print("=" * 60)
            print(f"  LoRA 合并失败 ⚠️  退出码: {merge_exit_code}")
            print("  仅保存了 LoRA 适配器，未生成完整模型")
            print("=" * 60)

    # 8. 结果
    print("=" * 60)
    if exit_code == 0:
        if merged_dir:
            print(f"  训练完成 ✅  完整模型保存至: {merged_dir}")
        else:
            print(f"  训练完成 ✅  LoRA适配器保存至: {output_dir}")
    else:
        print(f"  训练失败 ❌  退出码: {exit_code}")
    print("=" * 60)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
