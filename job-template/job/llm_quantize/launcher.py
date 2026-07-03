#!/usr/bin/env python3
"""
LLM 量化任务模板 — GPTQModel
支持: GPTQ / AWQ / bitsandbytes
"""

import argparse
import json
import os
import shutil
import sys

os.environ['HF_DATASETS_CACHE'] = '/mnt/storage/models-storage/datasets/quantization-dataset/'


def _copy_tokenizer(src: str, dst: str):
    for fname in ["tokenizer.json", "tokenizer_config.json", "tokenizer.model",
                  "vocab.json", "merges.txt", "added_tokens.json",
                  "special_tokens_map.json", "sentencepiece.bpe.model",
                  "config.json", "generation_config.json"]:
        src_path = os.path.join(src, fname)
        if os.path.isfile(src_path):
            os.makedirs(dst, exist_ok=True)
            shutil.copy2(src_path, os.path.join(dst, fname))


def _load_calib_text(dataset: str, nsamples: int):
    # 如果是绝对路径，直接使用
    if os.path.isabs(dataset):
        local_path = dataset
    else:
        local_path = f"/mnt/storage/models-storage/datasets/{dataset}"

    if os.path.isdir(local_path):
        from datasets import load_from_disk
        ds = load_from_disk(local_path)
        texts = ds.select(range(min(nsamples, len(ds))))["text"]
        print(f"  从 PVC 加载校准数据: {local_path}")
        return texts

    # 如果是短名称（如 wikitext2），fallback 到内嵌样本
    if not os.path.isabs(dataset) and dataset == "wikitext2":
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Large language models are trained on vast amounts of text data.",
            "GPTQ is a post-training quantization method using second-order information.",
            "AWQ stands for Activation-aware Weight Quantization.",
            "The Transformer architecture revolutionized natural language processing.",
            "Deep learning models require significant computational resources.",
            "Model compression techniques include pruning, quantization and distillation.",
            "Transfer learning allows pre-trained models to be adapted to specific tasks.",
            "Data preprocessing is an important step in machine learning pipelines.",
            "The evaluation metrics help assess the performance of trained models.",
            "Neural networks consist of layers of interconnected neurons.",
            "Gradient descent is used to minimize the loss function during training.",
            "The calibration dataset helps compute optimal quantization parameters.",
            "Structured pruning removes entire neurons or channels from the network.",
        ]
        texts = sample_texts[:min(nsamples, len(sample_texts))]
        print(f"  使用内嵌校准样本 ({len(texts)} 条)")
        return texts

    # 最后尝试从 HuggingFace 下载
    if not os.path.isabs(dataset):
        from datasets import load_dataset
        raw = load_dataset(dataset, split="train", trust_remote_code=True)
        texts = raw.select(range(nsamples))["text"]
        print(f"  从 HuggingFace 下载校准数据: {dataset}")
        return texts

    raise FileNotFoundError(f"校准数据集路径不存在: {local_path}")


def quantize_gptq(model_path: str, output: str, bits: int, group_size: int,
                  dataset: str = "wikitext2", nsamples: int = 128):
    from gptqmodel import GPTQModel, QuantizeConfig

    print(f"[GPTQ] 加载模型: {model_path}")
    quant_config = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=False,
    )
    model = GPTQModel.load(model_path, quant_config)

    print(f"[GPTQ] 加载校准数据: {dataset}, {nsamples} 条")
    texts = _load_calib_text(dataset, nsamples)

    print(f"[GPTQ] 开始量化 ({bits}bit, group_size={group_size})...")
    model.quantize(texts, batch_size=2)

    os.makedirs(output, exist_ok=True)
    model.save(output)
    _copy_tokenizer(model_path, output)
    print(f"[GPTQ] 量化完成，保存至: {output}")
    return {"engine": "gptqmodel", "bits": bits, "group_size": group_size, "output": output}


def quantize_awq(model_path: str, output: str, bits: int):
    import os
    import awq.utils.calib_data
    import awq.quantize.quantizer
    from datasets import load_from_disk
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    # 猴子补丁：替换 AWQ 的数据集加载函数，使用本地数据集
    # 检查多个可能的路径（兼容 datastes 拼写错误）
    local_dataset_path = "/mnt/storage/models-share-volume/datasets/pile-val-backup"
    if not os.path.isdir(local_dataset_path):
        alt = "/mnt/storage/models-share-volume/datastes/pile-val-backup"
        if os.path.isdir(alt):
            local_dataset_path = alt
    original_fn = awq.utils.calib_data.get_calib_dataset

    def patched_get_calib_dataset(*args, **kwargs):
        if not os.path.isdir(local_dataset_path):
            return original_fn(*args, **kwargs)
        print(f"[AWQ] 使用本地数据集: {local_dataset_path}")
        # 尝试多种格式加载
        try:
            ds = load_from_disk(local_dataset_path)
            return ds["validation"] if "validation" in ds else ds
        except Exception:
            pass
        try:
            from datasets import Dataset
            import json
            # 尝试 MsDataset 格式（modelscope）
            for fname in ["dataset_info.json", "config.json", "data.jsonl", "data.json"]:
                fpath = os.path.join(local_dataset_path, fname)
                if os.path.isfile(fpath):
                    if fname.endswith(".jsonl"):
                        data = [json.loads(l) for l in open(fpath, encoding="utf-8")]
                        return Dataset.from_list(data)
                    elif fname.endswith(".json"):
                        with open(fpath, encoding="utf-8") as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            return Dataset.from_list(data)
                        if "rows" in data:
                            return Dataset.from_list(data["rows"])
            # 尝试 parquet 格式
            import glob
            parquet_files = glob.glob(os.path.join(local_dataset_path, "*.parquet"))
            if parquet_files:
                return Dataset.from_parquet(parquet_files[0])
            # 尝试 CSV 格式（ModelScope 常用）
            csv_files = glob.glob(os.path.join(local_dataset_path, "*.csv"))
            if csv_files:
                import pandas as pd
                df = pd.read_csv(csv_files[0])
                ds = Dataset.from_pandas(df)
                # 只取 text 列
                if "text" in ds.column_names:
                    ds = ds.select_columns("text")
                print(f"[AWQ] 从 CSV 加载: {csv_files[0]}, {len(ds)} 条")
                return ds
        except Exception as e:
            print(f"[AWQ] 本地数据集加载失败: {e}")
        return original_fn(*args, **kwargs)

    # 两个导入路径都打补丁，确保拦截成功
    awq.utils.calib_data.get_calib_dataset = patched_get_calib_dataset
    awq.quantize.quantizer.get_calib_dataset = patched_get_calib_dataset

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
        elif args.method == "bnb":
            meta = quantize_bnb(args.model, args.output, args.bits)
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
