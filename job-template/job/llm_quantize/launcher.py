#!/usr/bin/env python3
"""
LLM 量化任务模板 — GPTQModel
支持: GPTQ / AWQ / bitsandbytes
参数通过环境变量或命令行传入
"""

import argparse
import json
import os
import shutil
import sys

# 指定数据集缓存路径（PVC 持久化，避免每次重复下载）
os.environ['HF_DATASETS_CACHE'] = '/mnt/storage/models-storage/datasets/quantization-dataset/'


def _copy_tokenizer(src: str, dst: str):
    """从源模型目录复制 tokenizer 文件到输出目录"""
    tokenizer_files = [
        "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
        "vocab.json", "merges.txt", "added_tokens.json",
        "special_tokens_map.json", "sentencepiece.bpe.model",
    ]
    for fname in tokenizer_files:
        src_path = os.path.join(src, fname)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, os.path.join(dst, fname))
    # 也复制配置文件
    for fname in ["config.json", "generation_config.json"]:
        src_path = os.path.join(src, fname)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, os.path.join(dst, fname))


def quantize_gptq(model_path: str, output: str, bits: int, group_size: int,
                  dataset: str = "c4", nsamples: int = 128):
    """使用 GPTQModel 进行 GPTQ 量化"""
    from gptqmodel import GPTQModel, QuantizeConfig
    from datasets import load_dataset

    print(f"[GPTQModel] 加载模型: {model_path}")
    quant_config = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=False,
    )
    model = GPTQModel.load(model_path, quant_config)

    print(f"[GPTQModel] 加载校准数据: {dataset}, {nsamples} 条")
    # 根据数据集名称自动选择配置
    if dataset == "wikitext2":
        calib = load_dataset("wikitext2", "wikitext-2-raw-v1", split="train",
                             trust_remote_code=True).select(range(nsamples))["text"]
    else:
        calib = load_dataset(dataset, "en", split="train",
                             trust_remote_code=True).select(range(nsamples))["text"]

    print(f"[GPTQModel] 开始量化 ({bits}bit, group_size={group_size})...")
    model.quantize(calib, batch_size=2)

    os.makedirs(output, exist_ok=True)
    model.save(output)
    # 复制 tokenizer 和配置文件
    _copy_tokenizer(model_path, output)
    print(f"[GPTQModel] 量化完成，保存至: {output}")
    return {"engine": "gptqmodel", "bits": bits, "group_size": group_size, "output": output}


def quantize_awq(model_path: str, output: str, bits: int):
    """使用 AutoAWQ 量化"""
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    print(f"[AWQ] 加载模型: {model_path}")
    model = AutoAWQForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"[AWQ] 开始量化 ({bits}bit)...")
    model.quantize(tokenizer, quant_config={"w_bit": bits, "version": "GEMM"})

    os.makedirs(output, exist_ok=True)
    model.save_quantized(output)
    tokenizer.save_pretrained(output)
    print(f"[AWQ] 量化完成，保存至: {output}")
    return {"engine": "awq", "bits": bits, "output": output}


def quantize_bnb(model_path: str, output: str, bits: int):
    """使用 bitsandbytes 运行时量化（仅支持 4bit 和 8bit）"""
    if bits not in (4, 8):
        raise ValueError(f"bitsandbytes 仅支持 4bit 和 8bit，当前值: {bits}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"[BNB] 加载模型 (NF{bits})...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=(bits == 4),
        load_in_8bit=(bits == 8),
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    os.makedirs(output, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    print(f"[BNB] 量化完成，保存至: {output}")
    return {"engine": "bitsandbytes", "bits": bits, "output": output}


def main():
    parser = argparse.ArgumentParser(description="LLM 量化模板")
    parser.add_argument("--method", type=str, default=os.getenv("QUANT_METHOD", "gptq"),
                        choices=["gptq", "awq", "bnb"],
                        help="量化方法: gptq / awq / bnb")
    parser.add_argument("--bits", type=int, default=int(os.getenv("QUANT_BITS", "4")),
                        help="量化位数: 2, 3, 4, 8")
    parser.add_argument("--group_size", type=int, default=int(os.getenv("QUANT_GROUP_SIZE", "128")),
                        help="GPTQ group size: 32/64/128/256")
    parser.add_argument("--model", type=str, default=os.getenv("MODEL_PATH", ""),
                        help="待量化模型路径或 HuggingFace model ID")
    parser.add_argument("--output", type=str, default=os.getenv("OUTPUT_PATH", "/mnt/admin/models/quant"),
                        help="量化后模型保存路径")
    parser.add_argument("--dataset", type=str, default=os.getenv("QUANT_DATASET", "wikitext2"),
                        help="GPTQ 校准数据集（默认 wikitext2，仅需 ~10MB）")
    parser.add_argument("--nsamples", type=int, default=int(os.getenv("QUANT_NSAMPLES", "128")),
                        help="GPTQ 校准样本数")
    args = parser.parse_args()

    if not args.model:
        print(json.dumps({"status": "failed", "error": "MODEL_PATH 未设置"}))
        sys.exit(1)

    # 如果是本地路径，检查是否存在
    if os.path.isabs(args.model) and not os.path.exists(args.model):
        print(json.dumps({"status": "failed", "error": f"模型路径不存在: {args.model}"}))
        sys.exit(1)

    try:
        if args.method == "gptq":
            meta = quantize_gptq(args.model, args.output, args.bits,
                                 args.group_size, args.dataset, args.nsamples)
        elif args.method == "awq":
            meta = quantize_awq(args.model, args.output, args.bits)
        elif args.method == "bnb":
            meta = quantize_bnb(args.model, args.output, args.bits)
        else:
            raise ValueError(f"不支持的量化方法: {args.method}")

        meta["status"] = "success"
        meta["method"] = args.method

    except Exception as e:
        meta = {"status": "failed", "error": str(e), "method": args.method}

    # 写入量化清单
    manifest_path = os.path.join(args.output, "quant_manifest.json")
    os.makedirs(args.output, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, ensure_ascii=False))
    if meta["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
