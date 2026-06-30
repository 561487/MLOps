#!/usr/bin/env python3
"""
LLM 模型剪枝任务模板 — Torch-Pruning (DepGraph)
支持: 结构化剪枝 / LLM 剪枝 / 层级剪枝
"""

import argparse
import json
import os
import sys
import shutil


def _copy_config(src: str, dst: str):
    """复制模型配置文件和 tokenizer 到输出目录"""
    for fname in ["config.json", "generation_config.json", "tokenizer.json",
                  "tokenizer_config.json", "tokenizer.model", "special_tokens_map.json"]:
        src_path = os.path.join(src, fname)
        if os.path.isfile(src_path):
            os.makedirs(dst, exist_ok=True)
            shutil.copy2(src_path, os.path.join(dst, fname))


def prune_structural(model_path: str, output: str, ratio: float,
                     example_shape: str = "1,3,224,224"):
    """结构化剪枝 — 使用 DepGraph 自动处理层间依赖"""
    import torch
    import torch.nn as nn
    import torch_pruning as tp
    from transformers import AutoModelForImageClassification, AutoConfig

    print(f"[Prune] 加载模型: {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageClassification.from_pretrained(
        model_path, config=config, trust_remote_code=True
    )
    model.eval()

    # 构造示例输入
    shape = tuple(int(x) for x in example_shape.split(","))
    example_inputs = torch.randn(*shape)

    print(f"[Prune] 构建依赖图, 剪枝比例: {ratio}")
    # 重要性标准: L1 范数
    imp = tp.importance.GroupNormImportance(p=2)

    # 构建依赖图并进行剪枝
    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs,
        importance=imp,
        pruning_ratio=ratio,
        global_pruning=False,
        iterative_steps=1,
    )

    # 执行剪枝
    for group in pruner.step(interactive=True):
        print(f"[Prune] 剪枝层: {group}")
        group.prune()

    os.makedirs(output, exist_ok=True)
    model.save_pretrained(output)
    config.save_pretrained(output)
    _copy_config(model_path, output)
    print(f"[Prune] 结构化剪枝完成, 保存至: {output}")
    return {"engine": "torch_pruning", "method": "structural", "ratio": ratio, "output": output}


def prune_llm(model_path: str, output: str, ratio: float,
              prune_heads: bool = True, prune_layers: bool = False,
              n_layers_remove: int = 0):
    """LLM 结构化剪枝 — 支持注意力头剪枝和层级剪枝"""
    import torch
    import torch_pruning as tp
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[Prune] 加载 LLM: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # 构造示例输入
    example_inputs = tokenizer("Hello, world!", return_tensors="pt")
    if hasattr(example_inputs, "input_ids"):
        example_inputs = example_inputs["input_ids"]

    meta = {"engine": "torch_pruning", "method": "llm", "ratio": ratio,
            "prune_heads": prune_heads, "prune_layers": prune_layers,
            "n_layers_remove": n_layers_remove}

    if prune_layers and n_layers_remove > 0:
        print(f"[Prune] 移除 {n_layers_remove} 层 Transformer 层")
        # 使用 Torch-Pruning 的 LLM 层剪枝
        tp.prune_llm_layers(model, example_inputs, n_remove=n_layers_remove)
        meta["layers_removed"] = n_layers_remove

    if prune_heads and ratio > 0:
        print(f"[Prune] LLM 结构化剪枝, 比例: {ratio}")
        # 使用 Torch-Pruning 的 LLM 结构化剪枝
        tp.prune_llm(model, example_inputs, pruning_ratio=ratio)

    os.makedirs(output, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    print(f"[Prune] LLM 剪枝完成, 保存至: {output}")
    meta["output"] = output
    return meta


def prune_layer(model_path: str, output: str, layer_names: str = ""):
    """按名称移除指定层"""
    import torch
    import torch_pruning as tp
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not layer_names:
        raise ValueError("layer_names 不能为空，请指定要移除的层名，以逗号分隔")

    print(f"[Prune] 加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    example_inputs = tokenizer("Hello!", return_tensors="pt")

    # 构建依赖图
    dg = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

    layers_to_remove = [name.strip() for name in layer_names.split(",") if name.strip()]
    for layer_name in layers_to_remove:
        print(f"[Prune] 移除层: {layer_name}")
        # 按名称查找模块并剪枝
        module = dict(model.named_modules()).get(layer_name)
        if module is None:
            print(f"[Prune] 警告: 未找到层 {layer_name}, 跳过")
            continue
        group = dg.get_pruning_group(module, tp.prune_linear_out, idxs=list(range(module.out_features)))
        group.prune()

    os.makedirs(output, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    print(f"[Prune] 层级剪枝完成, 保存至: {output}")
    return {"engine": "torch_pruning", "method": "layer", "layers": layer_names, "output": output}


def main():
    parser = argparse.ArgumentParser(description="模型剪枝模板 (Torch-Pruning)")
    parser.add_argument("--model", type=str, default=os.getenv("MODEL_PATH", ""),
                        help="模型路径或 HuggingFace ID")
    parser.add_argument("--output", type=str, default=os.getenv("OUTPUT_PATH", "/mnt/admin/models/pruned"),
                        help="剪枝后模型保存路径")
    parser.add_argument("--method", type=str, default=os.getenv("PRUNE_METHOD", "structural"),
                        choices=["structural", "llm", "layer"],
                        help="剪枝方法: structural结构化 / llm大模型 / layer层级移除")
    parser.add_argument("--ratio", type=float, default=float(os.getenv("PRUNE_RATIO", "0.3")),
                        help="剪枝比例 0.0~1.0")
    parser.add_argument("--example_shape", type=str, default=os.getenv("EXAMPLE_SHAPE", "1,3,224,224"),
                        help="结构化剪枝的示例输入形状, 逗号分隔")
    parser.add_argument("--prune_heads", type=str, default=os.getenv("PRUNE_HEADS", "true"),
                        help="是否剪枝注意力头 (LLM): true/false")
    parser.add_argument("--prune_layers", type=str, default=os.getenv("PRUNE_LAYERS", "false"),
                        help="是否剪枝整个 Transformer 层 (LLM): true/false")
    parser.add_argument("--n_layers_remove", type=int, default=int(os.getenv("N_LAYERS_REMOVE", "0")),
                        help="移除的 Transformer 层数")
    parser.add_argument("--layer_names", type=str, default=os.getenv("LAYER_NAMES", ""),
                        help="按名称移除的层名, 逗号分隔")
    args = parser.parse_args()

    # 字符串转 bool（平台传入的是 "true"/"false" 字符串）
    args.prune_heads = args.prune_heads.lower() == "true"
    args.prune_layers = args.prune_layers.lower() == "true"

    if not args.model:
        print(json.dumps({"status": "failed", "error": "MODEL_PATH 未设置"}))
        sys.exit(1)

    if os.path.isabs(args.model) and not os.path.exists(args.model):
        print(json.dumps({"status": "failed", "error": f"模型路径不存在: {args.model}"}))
        sys.exit(1)

    try:
        if args.method == "structural":
            meta = prune_structural(args.model, args.output, args.ratio, args.example_shape)
        elif args.method == "llm":
            meta = prune_llm(args.model, args.output, args.ratio,
                             args.prune_heads, args.prune_layers, args.n_layers_remove)
        elif args.method == "layer":
            meta = prune_layer(args.model, args.output, args.layer_names)
        else:
            raise ValueError(f"不支持的剪枝方法: {args.method}")

        meta["status"] = "success"
        meta["method"] = args.method

    except Exception as e:
        meta = {"status": "failed", "error": str(e), "method": args.method}

    manifest_path = os.path.join(args.output, "prune_manifest.json")
    os.makedirs(args.output, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, ensure_ascii=False))
    if meta["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
