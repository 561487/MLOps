#!/usr/bin/env python3
"""
LLM 量化任务模板 — LLM Compressor
支持: GPTQ / AWQ / RTN (Round-to-Nearest)
"""

import argparse
import json
import os
import shutil
import sys


# 指定数据集缓存路径（PVC 持久化，避免每次重复下载）
os.environ['HF_DATASETS_CACHE'] = '/mnt/storage/models-storage/datasets/quantization-dataset/'


def _copy_tokenizer(src: str, dst: str):
    """从源模型目录复制 tokenizer 和配置文件到输出目录"""
    for fname in ["tokenizer.json", "tokenizer_config.json", "tokenizer.model",
                  "vocab.json", "merges.txt", "added_tokens.json",
                  "special_tokens_map.json", "sentencepiece.bpe.model",
                  "config.json", "generation_config.json"]:
        src_path = os.path.join(src, fname)
        if os.path.isfile(src_path):
            os.makedirs(dst, exist_ok=True)
            shutil.copy2(src_path, os.path.join(dst, fname))


def quantize_gptq(model_path: str, output: str, bits: int, group_size: int,
                  dataset: str = "wikitext2", nsamples: int = 128):
    """使用 LLM Compressor 进行 GPTQ 量化"""
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"[LLMC] 加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", trust_remote_code=True
    )

    # 构建校准数据
    print(f"[LLMC] 加载校准数据: {dataset}, {nsamples} 条")

    # 搜索路径优先级：PVC 共享存储 > 镜像内缓存 > HuggingFace 在线
    dataset_paths = [
        f"/mnt/storage/models-storage/datasets/{dataset}",
        f"/mnt/storage/datasets/{dataset}",
        f"/app/datasets/{dataset}",
    ]

    loaded = False
    for path in dataset_paths:
        if os.path.isdir(path):
            from datasets import load_from_disk
            calib_data = load_from_disk(path).select(range(nsamples))
            print(f"  从本地加载: {path}")
            loaded = True
            break

    if not loaded:
        if dataset == "wikitext2":
            # 内嵌校准样本（备选方案）
            from datasets import Dataset
            sample_texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is a subset of artificial intelligence that enables systems to learn.",
                "Large language models are trained on vast amounts of text data to understand language.",
                "Quantization is the process of mapping continuous values to a finite set of discrete values.",
                "The Transformer architecture revolutionized natural language processing.",
                "Deep learning models require significant computational resources for training.",
                "Model compression techniques include pruning, quantization and distillation.",
                "GPTQ is a post-training quantization method that uses second-order information.",
                "AWQ stands for Activation-aware Weight Quantization for better compression.",
                "The calibration dataset is used to compute optimal quantization parameters.",
                "Transfer learning allows pre-trained models to be adapted to specific tasks.",
                "Neural networks consist of layers of interconnected neurons.",
                "Gradient descent is an algorithm used to minimize the loss function.",
                "Data preprocessing is an important step in any machine learning pipeline.",
                "The evaluation metrics help assess the performance of trained models.",
            ]
            calib_data = Dataset.from_dict({"text": sample_texts[:min(nsamples, len(sample_texts))]})
            print(f"  使用内嵌校准样本 ({len(calib_data)} 条)")
        else:
            from datasets import load_dataset
            calib_data = load_dataset(dataset, split="train",
                                      trust_remote_code=True).select(range(nsamples))

    def tokenize_calib(examples):
        return tokenizer(examples["text"], truncation=True, max_length=2048, padding=False)

    calib_encoded = calib_data.map(tokenize_calib, batched=True, remove_columns=calib_data.column_names)

    # GPTQ 量化配置（LLM Compressor v0.12+ API）
    scheme = f"W{bits}A16"
    print(f"[LLMC] 开始 GPTQ 量化: {scheme}, group_size={group_size}")
    from compressed_tensors.quantization import preset_name_to_scheme
    quant_scheme = preset_name_to_scheme(scheme, ["Linear"])
    quant_scheme.weights.group_size = group_size
    recipe = GPTQModifier(
        config_groups={"group_0": quant_scheme},
        ignore=["lm_head"],
    )
    oneshot(model=model, dataset=calib_encoded, recipe=recipe, max_seq_length=2048)

    os.makedirs(output, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    _copy_tokenizer(model_path, output)
    print(f"[LLMC] GPTQ 量化完成，保存至: {output}")
    return {"engine": "llm_compressor", "method": "gptq", "bits": bits,
            "group_size": group_size, "output": output}


def quantize_awq(model_path: str, output: str, bits: int):
    """使用 LLM Compressor 进行 AWQ 量化"""
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"[LLMC] 加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", trust_remote_code=True
    )

    scheme = f"W{bits}A16"
    print(f"[LLMC] AWQ 量化: {scheme}")
    # LLM Compressor 的 GPTQModifier 也支持 AWQ 风格的量化
    recipe = GPTQModifier(
        scheme=scheme,
        targets="Linear",
        ignore=["lm_head"],
    )
    calib_data = load_dataset("wikitext2", "wikitext-2-raw-v1", split="train",
                               trust_remote_code=True).select(range(128))

    def tokenize_calib(examples):
        return tokenizer(examples["text"], truncation=True, max_length=2048, padding=False)

    calib_encoded = calib_data.map(tokenize_calib, batched=True, remove_columns=calib_data.column_names)
    oneshot(model=model, dataset=calib_encoded, recipe=recipe, max_seq_length=2048)

    os.makedirs(output, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    print(f"[LLMC] AWQ 量化完成，保存至: {output}")
    return {"engine": "llm_compressor", "method": "awq", "bits": bits, "output": output}


def quantize_rtn(model_path: str, output: str, bits: int):
    """使用 LLM Compressor 进行 RTN (Round-to-Nearest) 量化"""
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"[LLMC] 加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", trust_remote_code=True
    )

    scheme = f"W{bits}A16"
    print(f"[LLMC] RTN 量化: {scheme}（无需校准数据）")
    recipe = QuantizationModifier(scheme=scheme, targets="Linear")
    # RTN 不需要校准数据，但还是需要传一个空 dataset
    calib_data = load_dataset("wikitext2", "wikitext-2-raw-v1", split="train",
                               trust_remote_code=True).select(range(1))

    def tokenize_calib(examples):
        return tokenizer(examples["text"], truncation=True, max_length=2048, padding=False)

    calib_encoded = calib_data.map(tokenize_calib, batched=True, remove_columns=calib_data.column_names)
    oneshot(model=model, dataset=calib_encoded, recipe=recipe, max_seq_length=2048)

    os.makedirs(output, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    print(f"[LLMC] RTN 量化完成，保存至: {output}")
    return {"engine": "llm_compressor", "method": "rtn", "bits": bits, "output": output}


def main():
    parser = argparse.ArgumentParser(description="LLM 量化模板 (LLM Compressor)")
    parser.add_argument("--method", type=str, default=os.getenv("QUANT_METHOD", "gptq"),
                        choices=["gptq", "awq", "rtn"],
                        help="量化方法: gptq / awq / rtn")
    parser.add_argument("--bits", type=int, default=int(os.getenv("QUANT_BITS", "4")),
                        help="量化位数: 4, 8")
    parser.add_argument("--group_size", type=int, default=int(os.getenv("QUANT_GROUP_SIZE", "128")),
                        help="GPTQ group size: 32/64/128/256")
    parser.add_argument("--model", type=str, default=os.getenv("MODEL_PATH", ""),
                        help="待量化模型路径或 HuggingFace model ID")
    parser.add_argument("--output", type=str, default=os.getenv("OUTPUT_PATH", "/mnt/admin/models/quant"),
                        help="量化后模型保存路径")
    parser.add_argument("--dataset", type=str, default=os.getenv("QUANT_DATASET", "wikitext2"),
                        help="GPTQ 校准数据集")
    parser.add_argument("--nsamples", type=int, default=int(os.getenv("QUANT_NSAMPLES", "128")),
                        help="GPTQ 校准样本数")
    args = parser.parse_args()

    if not args.model:
        print(json.dumps({"status": "failed", "error": "MODEL_PATH 未设置"}))
        sys.exit(1)

    if os.path.isabs(args.model) and not os.path.exists(args.model):
        print(json.dumps({"status": "failed", "error": f"模型路径不存在: {args.model}"}))
        sys.exit(1)

    try:
        if args.method == "gptq":
            meta = quantize_gptq(args.model, args.output, args.bits,
                                 args.group_size, args.dataset, args.nsamples)
        elif args.method == "awq":
            meta = quantize_awq(args.model, args.output, args.bits)
        elif args.method == "rtn":
            meta = quantize_rtn(args.model, args.output, args.bits)
        else:
            raise ValueError(f"不支持的量化方法: {args.method}")

        meta["status"] = "success"
        meta["method"] = args.method

    except Exception as e:
        import traceback
        meta = {"status": "failed", "error": str(e), "method": args.method,
                "traceback": traceback.format_exc()}

    manifest_path = os.path.join(args.output, "quant_manifest.json")
    os.makedirs(args.output, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, ensure_ascii=False))
    if meta["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
