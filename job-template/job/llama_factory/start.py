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
import ast
import json
import os
import re
import subprocess
import sys
import textwrap
from datetime import datetime

from common.training_monitor import TrainingMonitor


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
                        help="训练数据集名称，对应dataset_info.json中的数据集名，支持多个用逗号分隔")
    parser.add_argument("--dataset_dir", type=str, default="",
                        help="数据集目录路径，包含数据集JSON文件和dataset_info.json")
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

    # === 输出参数 ===
    parser.add_argument("--output_dir", type=str, default="",
                        help="模型保存路径，如 /mnt/storage/models-storage/output/lora，不传则自动生成")

    return parser.parse_args()


def resolve_output_dir(args):
    """解析模型输出目录，优先使用用户指定的路径"""
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"[start.py] 模型输出目录（用户指定）: {output_dir}")
        return output_dir

    creator = os.environ.get("KFJ_CREATOR", "admin")
    pipeline_name = os.environ.get("KFJ_PIPELINE_NAME", "default")
    task_name = os.environ.get("KFJ_TASK_NAME", "qwen3-lora")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"/mnt/{creator}/pipeline/{pipeline_name}/{task_name}-{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"[start.py] 模型输出目录（自动生成）: {output_dir}")
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


def run_training(config_path, monitor):
    """执行 LLaMAFactory 训练"""
    cmd = ["llamafactory-cli", "train", config_path]
    print(f"[start.py] 执行命令: {' '.join(cmd)}")
    sys.stdout.flush()

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end="", flush=True)
        # 解析 LLaMAFactory 输出的指标行: {'loss': ..., 'learning_rate': ..., 'epoch': ...}
        line_stripped = line.strip()
        if line_stripped.startswith("{") and line_stripped.endswith("}"):
            try:
                metrics = ast.literal_eval(line_stripped)
                if isinstance(metrics, dict):
                    # 只保留数值类型的指标
                    numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                    if numeric_metrics:
                        monitor.log(numeric_metrics)
            except Exception:
                pass
    return process.wait()


def merge_lora(model_name_or_path, adapter_path, output_dir):
    """合并 LoRA 适配器到基础模型（使用 PEFT Python API）"""
    merged_dir = os.path.join(output_dir, "merged_model")
    os.makedirs(merged_dir, exist_ok=True)

    print(f"[merge_lora] ====== 进入合并流程 ======")
    print(f"[merge_lora] model_name_or_path = {model_name_or_path}")
    print(f"[merge_lora] adapter_path        = {adapter_path}")
    print(f"[merge_lora] output_dir          = {output_dir}")
    print(f"[merge_lora] merged_dir          = {merged_dir}")
    print(f"[merge_lora] os.getcwd()         = {os.getcwd()}")
    print(f"[merge_lora] 当前进程 PID        = {os.getpid()}")
    sys.stdout.flush()

    # 1. 设置 CUDA 环境变量
    print(f"[merge_lora] 步骤1: 设置 CUDA_VISIBLE_DEVICES=0")
    print(f"[merge_lora]   设置前 CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '(未设置)')}")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(f"[merge_lora]   设置后 CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    sys.stdout.flush()

    # 2. import torch 及相关库
    print(f"[merge_lora] 步骤2: 开始 import torch ...")
    sys.stdout.flush()
    import torch
    print(f"[merge_lora]   torch.__version__ = {torch.__version__}")
    print(f"[merge_lora]   torch.cuda.is_available() = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[merge_lora]   torch.cuda.device_count() = {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"[merge_lora]   GPU[{i}] = {torch.cuda.get_device_name(i)}")
    sys.stdout.flush()

    print(f"[merge_lora]   开始 import peft ...")
    sys.stdout.flush()
    import peft
    print(f"[merge_lora]   peft.__version__ = {peft.__version__}")
    sys.stdout.flush()

    print(f"[merge_lora]   开始 import transformers ...")
    sys.stdout.flush()
    import transformers
    print(f"[merge_lora]   transformers.__version__ = {transformers.__version__}")
    sys.stdout.flush()

    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    # 3. 先加载 config 查看模型信息
    print(f"[merge_lora] 步骤3: 加载模型配置 (AutoConfig.from_pretrained)")
    sys.stdout.flush()
    try:
        model_config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        print(f"[merge_lora]   模型类型: {model_config.architectures}")
        print(f"[merge_lora]   隐藏层大小: {getattr(model_config, 'hidden_size', 'N/A')}")
        print(f"[merge_lora]   层数: {getattr(model_config, 'num_hidden_layers', 'N/A')}")
        print(f"[merge_lora]   注意力头数: {getattr(model_config, 'num_attention_heads', 'N/A')}")
        print(f"[merge_lora]   总参数量: {getattr(model_config, 'num_parameters', 'N/A')}")
    except Exception as e:
        print(f"[merge_lora]   加载配置失败: {e}")
        import traceback
        traceback.print_exc()
    sys.stdout.flush()

    # 4. 加载基础模型
    print(f"[merge_lora] 步骤4: 加载基础模型 (AutoModelForCausalLM.from_pretrained)")
    print(f"[merge_lora]   参数: device_map='auto', torch_dtype='auto', trust_remote_code=True")
    sys.stdout.flush()
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        print(f"[merge_lora]   基础模型加载成功")
        print(f"[merge_lora]   模型类型: {type(base_model).__name__}")
        print(f"[merge_lora]   模型设备: {base_model.device}")
        print(f"[merge_lora]   模型 dtype: {base_model.dtype}")
        print(f"[merge_lora]   模型参数量: {base_model.num_parameters():,}")
        # 检查模型在哪些设备上
        if hasattr(base_model, 'hf_device_map'):
            print(f"[merge_lora]   设备映射 (hf_device_map): {base_model.hf_device_map}")
    except Exception as e:
        print(f"[merge_lora]   基础模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return 1, merged_dir
    sys.stdout.flush()

    # 5. 加载 LoRA 适配器
    print(f"[merge_lora] 步骤5: 加载 LoRA 适配器 (PeftModel.from_pretrained)")
    print(f"[merge_lora]   适配器路径: {adapter_path}")
    sys.stdout.flush()
    try:
        # 先列出适配器目录内容
        if os.path.isdir(adapter_path):
            adapter_files = os.listdir(adapter_path)
            print(f"[merge_lora]   适配器目录内容: {adapter_files}")
        else:
            print(f"[merge_lora]   适配器目录不存在!")
            return 1, merged_dir
        sys.stdout.flush()

        model = PeftModel.from_pretrained(base_model, adapter_path)
        print(f"[merge_lora]   LoRA 适配器加载成功")
        print(f"[merge_lora]   模型当前设备: {model.device}")
        print(f"[merge_lora]   是否为 PeftModel: {isinstance(model, peft.PeftModel)}")
        # 打印 LoRA 配置信息
        if hasattr(model, 'peft_config'):
            print(f"[merge_lora]   peft_config: {model.peft_config}")
    except Exception as e:
        print(f"[merge_lora]   LoRA 适配器加载失败: {e}")
        import traceback
        traceback.print_exc()
        return 1, merged_dir
    sys.stdout.flush()

    # 6. merge_and_unload
    print(f"[merge_lora] 步骤6: 执行 merge_and_unload()")
    print(f"[merge_lora]   开始时间: {datetime.now().strftime('%H:%M:%S')}")
    sys.stdout.flush()
    try:
        merged_model = model.merge_and_unload()
        print(f"[merge_lora]   merge_and_unload 成功返回")
        print(f"[merge_lora]   结束时间: {datetime.now().strftime('%H:%M:%S')}")
        print(f"[merge_lora]   合并后模型类型: {type(merged_model).__name__}")
        print(f"[merge_lora]   合并后模型设备: {merged_model.device}")
        print(f"[merge_lora]   合并后模型 dtype: {merged_model.dtype}")
        print(f"[merge_lora]   合并后参数量: {merged_model.num_parameters():,}")
    except Exception as e:
        print(f"[merge_lora]   merge_and_unload 失败: {e}")
        import traceback
        traceback.print_exc()
        return 1, merged_dir
    sys.stdout.flush()

    # 7. 验证目录可写（先于保存，快速失败）
    print(f"[merge_lora] 步骤7: 验证保存目录可写")
    print(f"[merge_lora]   目标目录: {merged_dir}")
    sys.stdout.flush()
    try:
        test_file = os.path.join(merged_dir, ".write_test")
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
        print(f"[merge_lora]   目录可写验证通过")
    except Exception as e:
        print(f"[merge_lora]   目录不可写: {e}")
        import traceback
        traceback.print_exc()
        return 1, merged_dir
    sys.stdout.flush()

    # 8. 直接在 GPU 上保存模型权重（不走 .cpu()，避免 OOM）
    #    模型在单卡 cuda:0 上，没有分片问题，可以直接保存
    print(f"[merge_lora] 步骤8: 直接在 GPU 上保存模型权重 (save_pretrained)")
    print(f"[merge_lora]   当前模型设备: {merged_model.device}")
    print(f"[merge_lora]   开始时间: {datetime.now().strftime('%H:%M:%S')}")
    print(f"[merge_lora]   参数: safe_serialization=True, max_shard_size='10GB'")
    sys.stdout.flush()

    print(f"[merge_lora]   尝试方案A: safe_serialization=True")
    sys.stdout.flush()
    try:
        # 直接从 GPU 保存，不做 .cpu()，避免额外内存占用导致 OOM
        merged_model.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="10GB")
        print(f"[merge_lora]   方案A 成功!")
        print(f"[merge_lora]   结束时间: {datetime.now().strftime('%H:%M:%S')}")
        # 保存完成后清理 GPU 缓存
        torch.cuda.empty_cache()
        print(f"[merge_lora]   GPU 缓存已清理")
    except Exception as e:
        print(f"[merge_lora]   方案A 失败: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()

        # 检查 safetensors 是否安装
        try:
            import safetensors
            print(f"[merge_lora]   safetensors 已安装 (版本: {safetensors.__version__})")
        except ImportError:
            print(f"[merge_lora]   safetensors 未安装")
        sys.stdout.flush()

        # 回退到 unsafe
        print(f"[merge_lora]   尝试方案B: safe_serialization=False")
        sys.stdout.flush()
        try:
            merged_model.save_pretrained(merged_dir, safe_serialization=False, max_shard_size="10GB")
            print(f"[merge_lora]   方案B 成功!")
            print(f"[merge_lora]   结束时间: {datetime.now().strftime('%H:%M:%S')}")
            torch.cuda.empty_cache()
        except Exception as e2:
            print(f"[merge_lora]   方案B 也失败: {e2}")
            traceback.print_exc()
            return 1, merged_dir
    sys.stdout.flush()

    # 9. 立即检查保存的文件
    print(f"[merge_lora] 步骤9: 检查保存后的目录内容")
    sys.stdout.flush()
    try:
        saved_files = os.listdir(merged_dir)
        print(f"[merge_lora]   目录文件数: {len(saved_files)}")
        for f in sorted(saved_files):
            file_path = os.path.join(merged_dir, f)
            file_size = os.path.getsize(file_path)
            print(f"[merge_lora]   {f:50s} {file_size:>12,} bytes")
    except Exception as e:
        print(f"[merge_lora]   列出目录失败: {e}")
    sys.stdout.flush()

    # 10. 保存分词器
    print(f"[merge_lora] 步骤10: 保存分词器")
    sys.stdout.flush()
    try:
        print(f"[merge_lora]   加载分词器: AutoTokenizer.from_pretrained({model_name_or_path})")
        sys.stdout.flush()
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        print(f"[merge_lora]   分词器类型: {type(tokenizer).__name__}")
        print(f"[merge_lora]   vocab_size: {tokenizer.vocab_size}")
        print(f"[merge_lora]   保存分词器到: {merged_dir}")
        sys.stdout.flush()
        tokenizer.save_pretrained(merged_dir)
        print(f"[merge_lora]   分词器保存完成")
    except Exception as e:
        print(f"[merge_lora]   分词器保存失败: {e}")
        import traceback
        traceback.print_exc()
    sys.stdout.flush()

    # 11. 最终检查
    print(f"[merge_lora] 步骤11: 最终确认")
    sys.stdout.flush()
    try:
        final_files = os.listdir(merged_dir)
        print(f"[merge_lora]   合并后目录最终文件数: {len(final_files)}")
        total_size = sum(os.path.getsize(os.path.join(merged_dir, f)) for f in final_files)
        print(f"[merge_lora]   合并后目录总大小: {total_size:,} bytes ({total_size/1024/1024/1024:.2f} GB)")
        has_model = any(f.startswith("pytorch_model") or f.startswith("model-") for f in final_files)
        has_config = any("config.json" in f for f in final_files)
        print(f"[merge_lora]   包含模型权重: {has_model}")
        print(f"[merge_lora]   包含配置文件: {has_config}")
    except Exception as e:
        print(f"[merge_lora]   最终检查失败: {e}")
    sys.stdout.flush()

    print(f"[merge_lora] ====== 合并流程结束 ======")
    sys.stdout.flush()
    return 0, merged_dir


def _auto_create_dataset_info(dataset_dir):
    """自动检测 HuggingFace 数据集格式并生成 dataset_info.json

    当用户指定的数据集目录缺少 dataset_info.json 时，
    从 HuggingFace 格式的 dataset_infos.json 和数据文件第一行推断列映射。
    """
    import json
    import os
    import glob

    # 1. 优先尝试从 dataset_infos.json（HuggingFace 格式）获取特征名称
    hf_info_path = os.path.join(dataset_dir, "dataset_infos.json")
    hf_features = None
    if os.path.exists(hf_info_path):
        try:
            with open(hf_info_path, "r") as f:
                hf_info = json.load(f)
            # HuggingFace 格式：{"default": {"features": {"conversations": {...}}, ...}}
            for split_name, split_info in hf_info.items():
                features = split_info.get("features", {})
                if features:
                    hf_features = list(features.keys())
                    print(f"[start.py] 从 dataset_infos.json 检测到特征列: {hf_features}")
                    break
        except Exception as e:
            print(f"[start.py] 解析 dataset_infos.json 失败: {e}")

    # 2. 查找数据文件（jsonl 优先）
    data_files = glob.glob(os.path.join(dataset_dir, "*.jsonl"))
    data_files += glob.glob(os.path.join(dataset_dir, "*.json"))
    data_files = [f for f in data_files if not f.endswith("dataset_infos.json") and not f.endswith("dataset_info.json")]

    if not data_files:
        print(f"[start.py] 警告: 在 {dataset_dir} 中未找到 .jsonl 或 .json 数据文件")
        return

    # 3. 对每个数据文件，读取第一行推断格式并生成配置
    dataset_info = {}
    for data_file in data_files:
        file_name = os.path.basename(data_file)
        # 数据集名称：去掉扩展名
        dataset_name = file_name.rsplit(".", 1)[0]

        try:
            with open(data_file, "r") as f:
                first_line = f.readline().strip()
            if not first_line:
                continue
            sample = json.loads(first_line)
        except Exception as e:
            print(f"[start.py] 跳过 {file_name}（读取失败: {e}）")
            continue

        # 推断列映射规则：
        # - 有 conversations 列 → ShareGPT 格式
        # - 有 instruction/input/output → 标准格式
        # - 有 messages 列 → ShareGPT messages 格式
        # - 有 prompt/response → 标准格式
        sample_keys = list(sample.keys())

        if "conversations" in sample_keys:
            # ShareGPT 格式：检测对话内是 from/value 还是 role/content
            conv_sample = sample["conversations"]
            tags = {}
            if conv_sample and isinstance(conv_sample, list) and len(conv_sample) > 0:
                first_msg = conv_sample[0]
                if "from" in first_msg:
                    # 默认就是 from/gpt/human，无需额外 tag
                    pass
                elif "role" in first_msg:
                    # LLaMAFactory 默认 role_tag="from"，需覆盖为 "role"
                    tags = {
                        "role_tag": "role",
                        "content_tag": "content",
                        "user_tag": "user",
                        "assistant_tag": "assistant"
                    }
            entry = {
                "file_name": file_name,
                "formatting": "sharegpt",
                "columns": {"messages": "conversations"}
            }
            if tags:
                entry["tags"] = tags
        elif "messages" in sample_keys:
            entry = {
                "file_name": file_name,
                "formatting": "sharegpt",
                "columns": {"messages": "messages"}
            }
        elif "instruction" in sample_keys:
            entry = {
                "file_name": file_name,
                "columns": {
                    "prompt": "instruction",
                    "query": "input" if "input" in sample_keys else "",
                    "response": "output"
                }
            }
        elif "prompt" in sample_keys and "response" in sample_keys:
            entry = {
                "file_name": file_name,
                "columns": {
                    "prompt": "prompt",
                    "response": "response"
                }
            }
        elif hf_features:
            # 用 dataset_infos.json 的特征名，默认按 ShareGPT 处理
            entry = {
                "file_name": file_name,
                "formatting": "sharegpt",
                "columns": {"messages": hf_features[0] if hf_features else "conversations"}
            }
        else:
            print(f"[start.py] 跳过 {file_name}（无法推断列映射，字段: {sample_keys}）")
            continue

        dataset_info[dataset_name] = entry
        print(f"[start.py]    {dataset_name} → {json.dumps(entry, ensure_ascii=False)}")

    if dataset_info:
        output_path = os.path.join(dataset_dir, "dataset_info.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        print(f"[start.py] 已自动生成 {output_path}")
    else:
        print(f"[start.py] 警告: 未生成任何数据集配置，请检查数据文件格式")


def main():
    print("=" * 60)
    print("  llama_factory LoRA 微调任务 - 启动")
    print("=" * 60)

    # 0. 创建 TrainingMonitor（日志目录由环境变量 SWANLAB_LOGDIR 控制）
    monitor = TrainingMonitor()
    final_status = "FAILED"
    exit_code = 1

    try:
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
        output_dir = resolve_output_dir(args)

        # 4. 构建训练配置
        config = build_config(args, output_dir)

        # 5. 安装 LLamaFactory
        if not install_llamafactory():
            sys.exit(1)

        # 6. 启动 SwanLab 监控
        monitor.start()

        # 7. 确定数据集目录
        if args.dataset_dir:
            # 用户指定了数据集目录
            data_dir = args.dataset_dir
            if not os.path.isdir(data_dir):
                print(f"[start.py] 错误：数据集目录不存在: {data_dir}")
                sys.exit(1)
            print(f"[start.py] 使用用户指定的数据集目录: {data_dir}")

            # 每次运行时自动检测并生成 dataset_info.json（覆盖已有文件）
            print(f"[start.py] 正在自动检测数据集格式并生成 dataset_info.json...")
            _auto_create_dataset_info(data_dir)
        else:
            # 未指定数据集目录，创建默认的 data/ 目录
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

            print(f"[start.py] 使用默认数据集目录: {data_dir}")

        config["dataset_dir"] = data_dir

        # 写入配置（在补充好所有字段后）
        config_path = os.path.join(output_dir, "train_config.yaml")
        write_config_yaml(config, config_path)

        # 8. 执行训练（传入 monitor 以便实时上报指标）
        exit_code = run_training(config_path, monitor)

        # 9. 训练完成后合并 LoRA 到基础模型
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
                final_status = "SUCCEEDED"
            else:
                print("=" * 60)
                print(f"  LoRA 合并失败 ⚠️  退出码: {merge_exit_code}")
                print("  仅保存了 LoRA 适配器，未生成完整模型")
                print("=" * 60)
                final_status = "FAILED"
        else:
            final_status = "FAILED"

        # 10. 结果
        print("=" * 60)
        if exit_code == 0:
            if merged_dir:
                print(f"  训练完成 ✅  完整模型保存至: {merged_dir}")
            else:
                print(f"  训练完成 ✅  LoRA适配器保存至: {output_dir}")
        else:
            print(f"  训练失败 ❌  退出码: {exit_code}")
        print("=" * 60)

    except SystemExit:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[start.py] 未捕获异常: {e}")
        final_status = "FAILED"
    finally:
        try:
            monitor.finish(final_status)
        except Exception as e:
            print(f"[start.py] monitor.finish 警告: {e}", flush=True)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
