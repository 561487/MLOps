#!/usr/bin/env python3
"""Cube Studio LLM quantization job template."""

import argparse
import datetime
import importlib.metadata
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

MANIFEST_NAME = "quant_manifest.json"
DEFAULT_MODEL_CACHE = "/mnt/storage/models-storage/modelscope-cache"
DATASET_ROOTS = (
    "/mnt/storage/models-storage/datasets",
    "/mnt/storage/models-share-volume/datasets",
)
TEXT_COLUMNS = ("text", "content", "prompt", "instruction", "question", "input")
EMBEDDED_CALIBRATION_TEXTS = (
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
)


_ACTIVE_MONITOR = None
_MONITOR_STARTED_CLOCK = None
_MONITOR_EVENT_STEP = 0
_STAGE_PROGRESS = {
    "model_source": (1, 10), "model_download": (1, 10),
    "validate": (2, 20), "calibration": (3, 35),
    "load_model": (4, 50), "quantize": (5, 65),
    "save_model": (6, 85), "complete": (7, 100), "skip": (7, 100),
    "failed": (7, 100), "manifest_failed": (7, 100),
}

def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _emit(stage: str, message: str, **details):
    global _MONITOR_EVENT_STEP
    event = {
        "event": "quantization",
        "stage": stage,
        "message": message,
        "timestamp": _utc_now(),
    }
    event.update(details)
    print(json.dumps(event, ensure_ascii=False), flush=True)
    if _ACTIVE_MONITOR is None:
        return
    try:
        _MONITOR_EVENT_STEP += 1
        stage_index, progress = _STAGE_PROGRESS.get(stage, (0, 0))
        metrics = {
            "metrics/stage_index": stage_index,
            "metrics/progress_percent": progress,
            "metrics/event_count": _MONITOR_EVENT_STEP,
        }
        if _MONITOR_STARTED_CLOCK is not None:
            metrics["metrics/elapsed_seconds"] = round(
                time.monotonic() - _MONITOR_STARTED_CLOCK, 3
            )
        for name, metric_name in {
            "samples": "dataset/calibration_samples",
            "file_count": "metrics/output_files",
            "total_bytes": "metrics/output_size_bytes",
            "bits": "config/bits",
            "group_size": "config/group_size",
        }.items():
            value = details.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[metric_name] = value
        _ACTIVE_MONITOR.log(metrics, step=_MONITOR_EVENT_STEP)
    except Exception as monitor_error:
        print(f"TrainingMonitor 指标记录失败: {monitor_error}", flush=True)


def _create_monitor():
    from common.training_monitor import TrainingMonitor
    return TrainingMonitor()


def _start_monitor(args):
    global _ACTIVE_MONITOR, _MONITOR_STARTED_CLOCK, _MONITOR_EVENT_STEP
    try:
        monitor = _create_monitor()
        monitor.start()
        monitor.log({
            "config/bits": int(args.bits),
            "config/group_size": int(args.group_size),
            "config/nsamples": int(args.nsamples),
            "config/method_gptq": int(args.method == "gptq"),
            "config/method_awq": int(args.method == "awq"),
            "config/method_bnb": int(args.method == "bnb"),
            "config/source_pvc": int(args.model_source == "pvc"),
            "config/source_modelscope": int(args.model_source == "modelscope"),
        }, step=0)
        _ACTIVE_MONITOR = monitor
        _MONITOR_STARTED_CLOCK = time.monotonic()
        _MONITOR_EVENT_STEP = 0
        return monitor
    except Exception as monitor_error:
        _ACTIVE_MONITOR = None
        print(f"TrainingMonitor 初始化失败: {monitor_error}", flush=True)
        return None


def _finish_monitor(monitor, status, manifest=None):
    global _ACTIVE_MONITOR, _MONITOR_STARTED_CLOCK, _MONITOR_EVENT_STEP
    if monitor is None:
        return
    try:
        if manifest:
            artifacts = manifest.get("artifacts") or {}
            monitor.log({
                "metrics/duration_seconds": float(manifest.get("duration_seconds", 0)),
                "metrics/output_files": int(artifacts.get("file_count", 0)),
                "metrics/output_size_bytes": int(artifacts.get("total_bytes", 0)),
            })
        monitor.finish(status)
    except Exception as monitor_error:
        print(f"TrainingMonitor 结束失败: {monitor_error}", flush=True)
    finally:
        _ACTIVE_MONITOR = None
        _MONITOR_STARTED_CLOCK = None
        _MONITOR_EVENT_STEP = 0


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"无法识别的布尔值: {value}")


def build_parser():
    parser = argparse.ArgumentParser(description="LLM 模型量化")
    parser.add_argument("--method", default=os.getenv("QUANT_METHOD", "gptq"),
                        choices=("gptq", "awq", "bnb"), help="量化方法")
    parser.add_argument("--bits", type=int, default=int(os.getenv("QUANT_BITS", "4")),
                        help="GPTQ: 2/3/4/8，AWQ: 4，BnB: 4/8")
    parser.add_argument("--group_size", type=int,
                        default=int(os.getenv("QUANT_GROUP_SIZE", "128")),
                        help="GPTQ group size: 32/64/128/256")
    parser.add_argument("--model_source", default=os.getenv("MODEL_SOURCE", "pvc"),
                        choices=("pvc", "modelscope"),
                        help="模型来源：PVC 路径或 ModelScope 模型 ID")
    parser.add_argument("--model", default=os.getenv("MODEL_PATH", ""),
                        help="待量化模型的 PVC 绝对路径或 ModelScope 模型 ID")
    parser.add_argument("--model_revision", default=os.getenv("MODEL_REVISION", "master"),
                        help="ModelScope 模型版本，PVC 来源时忽略")
    parser.add_argument("--model_cache",
                        default=os.getenv("MODELSCOPE_CACHE", DEFAULT_MODEL_CACHE),
                        help="ModelScope 模型下载到 PVC 的缓存目录")
    parser.add_argument("--output", default=os.getenv("OUTPUT_PATH", "/mnt/admin/models/quant"),
                        help="量化后模型保存路径")
    parser.add_argument("--dataset", default=os.getenv("QUANT_DATASET", "wikitext2"),
                        help="GPTQ/AWQ 校准数据集的 PVC 路径、文件或 wikitext2 离线样本")
    parser.add_argument("--nsamples", type=int,
                        default=int(os.getenv("QUANT_NSAMPLES", "128")),
                        help="GPTQ/AWQ 校准样本数")
    parser.add_argument("--force", type=_parse_bool, nargs="?", const=True,
                        default=_parse_bool(os.getenv("QUANT_FORCE", "false")),
                        help="允许复用非空输出目录")
    return parser


def validate_args(args):
    if not str(args.model).strip():
        raise ValueError("MODEL_PATH/--model 未设置")
    if not str(args.output).strip():
        raise ValueError("OUTPUT_PATH/--output 未设置")
    if args.model_source == "pvc":
        if not os.path.isabs(args.model):
            raise ValueError("PVC 模型必须使用绝对路径")
        if not os.path.isdir(args.model):
            raise FileNotFoundError(f"PVC 模型目录不存在: {args.model}")
        if os.path.realpath(args.model) == os.path.realpath(args.output):
            raise ValueError("输出目录不能与输入模型目录相同")
    elif args.model_source == "modelscope":
        if not str(args.model_revision).strip():
            raise ValueError("ModelScope model_revision 不能为空")
        if not os.path.isabs(args.model_cache):
            raise ValueError("ModelScope model_cache 必须是 PVC 绝对路径")
    if not 1 <= args.nsamples <= 10000:
        raise ValueError("nsamples 必须在 1 到 10000 之间")

    if args.method == "gptq":
        if args.bits not in (2, 3, 4, 8):
            raise ValueError("GPTQ 仅支持 2、3、4、8 bit")
        if args.group_size not in (32, 64, 128, 256):
            raise ValueError("GPTQ group_size 仅支持 32、64、128、256")
        if not str(args.dataset).strip():
            raise ValueError("GPTQ 必须提供校准数据集")
    elif args.method == "awq":
        if args.bits != 4:
            raise ValueError("当前 AutoAWQ 节点仅支持 4 bit")
        if not str(args.dataset).strip():
            raise ValueError("AWQ 必须提供校准数据集")
    elif args.method == "bnb" and args.bits not in (4, 8):
        raise ValueError("bitsandbytes 仅支持 4、8 bit")


def _resolve_model(args):
    if args.model_source == "pvc":
        resolved = os.path.realpath(args.model)
        _emit("model_source", "使用 PVC 模型目录", source="pvc", path=resolved)
        return resolved

    from modelscope import snapshot_download

    os.makedirs(args.model_cache, exist_ok=True)
    _emit("model_download", "从 ModelScope 下载模型",
          source="modelscope", model_id=args.model,
          revision=args.model_revision, cache_dir=args.model_cache)
    resolved = snapshot_download(
        args.model,
        revision=args.model_revision,
        cache_dir=args.model_cache,
    )
    if not resolved or not os.path.isdir(resolved):
        raise FileNotFoundError(f"ModelScope 下载结果不是有效模型目录: {resolved}")
    resolved = os.path.realpath(resolved)
    _emit("model_download", "ModelScope 模型已缓存到 PVC",
          source="modelscope", model_id=args.model, path=resolved)
    return resolved


def _copy_tokenizer(src: str, dst: str):
    for filename in (
        "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
        "vocab.json", "merges.txt", "added_tokens.json",
        "special_tokens_map.json", "sentencepiece.bpe.model",
        "config.json", "generation_config.json",
    ):
        source = os.path.join(src, filename)
        if os.path.isfile(source):
            os.makedirs(dst, exist_ok=True)
            shutil.copy2(source, os.path.join(dst, filename))


def _dataset_candidates(dataset: str):
    if os.path.isabs(dataset):
        return (dataset,)
    return tuple(os.path.join(root, dataset) for root in DATASET_ROOTS)


def _select_dataset_split(dataset):
    if hasattr(dataset, "select") and hasattr(dataset, "column_names"):
        return dataset
    if hasattr(dataset, "keys"):
        keys = list(dataset.keys())
        if not keys:
            raise ValueError("校准数据集不包含任何 split")
        return dataset["train"] if "train" in keys else dataset[keys[0]]
    raise TypeError(f"不支持的数据集对象: {type(dataset).__name__}")


def _extract_texts(dataset, nsamples: int):
    dataset = _select_dataset_split(dataset)
    columns = list(dataset.column_names)
    text_column = next((name for name in TEXT_COLUMNS if name in columns), None)
    if text_column is None:
        raise ValueError(
            f"校准数据集缺少文本列；支持: {', '.join(TEXT_COLUMNS)}；实际: {', '.join(columns)}"
        )
    values = dataset.select(range(min(nsamples, len(dataset))))[text_column]
    texts = [str(value).strip() for value in values
             if value is not None and str(value).strip()]
    if not texts:
        raise ValueError(f"校准数据集文本列 {text_column!r} 没有有效内容")
    return texts


def _load_local_dataset(path: str):
    from datasets import load_dataset, load_from_disk

    if os.path.isdir(path):
        try:
            return load_from_disk(path)
        except (FileNotFoundError, ValueError):
            files = sorted(
                item for item in Path(path).iterdir()
                if item.suffix.lower() in {".csv", ".json", ".jsonl", ".txt"}
            )
            if not files:
                raise ValueError(f"目录不是 datasets save_to_disk 格式且没有支持的数据文件: {path}")
            path = str(files[0])
    loader = {".csv": "csv", ".json": "json", ".jsonl": "json", ".txt": "text"}.get(
        Path(path).suffix.lower()
    )
    if not loader:
        raise ValueError(f"不支持的校准数据文件格式: {Path(path).suffix or '(无扩展名)'}")
    return load_dataset(loader, data_files=path, split="train")


def _load_calib_text(dataset: str, nsamples: int):
    for candidate in _dataset_candidates(dataset):
        if os.path.exists(candidate):
            texts = _extract_texts(_load_local_dataset(candidate), nsamples)
            _emit("calibration", "已从本地/PVC加载校准数据",
                  source=candidate, samples=len(texts))
            return texts

    if os.path.basename(dataset.rstrip("/")).lower() in {"wikitext2", "wikitext-2"}:
        texts = list(EMBEDDED_CALIBRATION_TEXTS[:nsamples])
        _emit("calibration", "校准数据路径不存在，使用内嵌离线样本",
              source="embedded:wikitext2", samples=len(texts))
        return texts
    candidates = ", ".join(_dataset_candidates(dataset))
    raise FileNotFoundError(f"校准数据集不存在，仅支持 PVC/本地路径: {candidates}")


def quantize_gptq(model_path, output, bits, group_size, dataset, nsamples):
    from gptqmodel import GPTQModel, QuantizeConfig

    _emit("load_model", "加载 GPTQ 模型", method="gptq", model=model_path)
    model = GPTQModel.load(
        model_path,
        QuantizeConfig(bits=bits, group_size=group_size, desc_act=False),
    )
    texts = _load_calib_text(dataset, nsamples)
    _emit("quantize", "开始 GPTQ 量化", method="gptq", bits=bits,
          group_size=group_size, samples=len(texts))
    model.quantize(texts, batch_size=1)
    _emit("save_model", "保存 GPTQ 量化模型", method="gptq", output=output)
    os.makedirs(output, exist_ok=True)
    model.save(output)
    _copy_tokenizer(model_path, output)
    return {"engine": "gptqmodel", "bits": bits, "group_size": group_size,
            "calibration_samples": len(texts), "output": output}


def quantize_awq(model_path, output, bits, dataset, nsamples):
    import awq.quantize.quantizer
    import awq.utils.calib_data
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    texts = _load_calib_text(dataset, nsamples)
    original_loader = awq.utils.calib_data.get_calib_dataset

    def patched_loader(*args, **kwargs):
        tokenizer = kwargs.get("tokenizer")
        if tokenizer is None:
            tokenizer = next(
                (arg for arg in args if callable(arg) and hasattr(arg, "encode")),
                None,
            )
        if tokenizer is None:
            return original_loader(*args, **kwargs)
        max_length = kwargs.get(
            "seqlen",
            kwargs.get("block_size", kwargs.get("max_calib_seq_len", 2048)),
        )
        tokenized = tokenizer(texts, truncation=True, padding=True,
                              max_length=max_length, return_tensors="pt")
        return [tokenized["input_ids"][index:index + 1] for index in range(len(texts))]

    awq.utils.calib_data.get_calib_dataset = patched_loader
    awq.quantize.quantizer.get_calib_dataset = patched_loader
    _emit("load_model", "加载 AWQ 模型", method="awq", model=model_path)
    model = AutoAWQForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _emit("quantize", "开始 AWQ 量化", method="awq", bits=bits, samples=len(texts))
    model.quantize(tokenizer, quant_config={"w_bit": bits, "version": "GEMM"})
    _emit("save_model", "保存 AWQ 量化模型", method="awq", output=output)
    os.makedirs(output, exist_ok=True)
    model.save_quantized(output)
    tokenizer.save_pretrained(output)
    return {"engine": "autoawq", "bits": bits,
            "calibration_samples": len(texts), "output": output}


def quantize_bnb(model_path, output, bits):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    _emit("load_model", "加载 bitsandbytes 模型", method="bnb", model=model_path)
    config = BitsAndBytesConfig(
        load_in_4bit=(bits == 4), load_in_8bit=(bits == 8),
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=(bits == 4),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=config, device_map="auto", trust_remote_code=True
    )
    _emit("save_model", "保存 bitsandbytes 量化模型", method="bnb", bits=bits)
    os.makedirs(output, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    return {"engine": "bitsandbytes", "bits": bits, "output": output}


def _output_summary(output: str):
    files = [item for item in Path(output).rglob("*") if item.is_file()]
    return {"file_count": len(files), "total_bytes": sum(item.stat().st_size for item in files)}


def _package_versions():
    versions = {}
    for package in ("gptqmodel", "autoawq", "bitsandbytes", "torch", "transformers", "datasets", "modelscope"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def _successful_manifest(output: str):
    path = os.path.join(output, MANIFEST_NAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        return manifest if manifest.get("status") == "success" else None
    except (OSError, ValueError, TypeError):
        return None


def _ensure_output_ready(output: str, force: bool):
    successful = _successful_manifest(output)
    if successful and not force:
        return successful
    if os.path.isdir(output) and any(Path(output).iterdir()) and not force:
        raise FileExistsError(f"输出目录非空且没有成功清单: {output}；确认可覆盖后传入 --force true")
    return None


def _write_manifest(output: str, manifest):
    os.makedirs(output, exist_ok=True)
    path = os.path.join(output, MANIFEST_NAME)
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    os.replace(temporary_path, path)
    return path


def run(args):
    started_clock = time.monotonic()
    monitor = _start_monitor(args)
    manifest = {
        "status": "running", "method": args.method, "model": args.model,
        "model_source": args.model_source, "output": args.output,
        "parameters": {"bits": args.bits, "group_size": args.group_size,
                       "dataset": args.dataset, "nsamples": args.nsamples,
                       "model_revision": args.model_revision,
                       "model_cache": args.model_cache},
        "started_at": _utc_now(), "runtime_versions": _package_versions(),
    }
    try:
        validate_args(args)
        previous = _ensure_output_ready(args.output, args.force)
        if previous:
            _emit("skip", "检测到成功清单，跳过重复量化", output=args.output)
            skipped = {"status": "skipped", "reason": "already_succeeded",
                       "previous": previous}
            _finish_monitor(monitor, "SUCCEEDED", skipped)
            return skipped
        model_path = _resolve_model(args)
        if os.path.realpath(model_path) == os.path.realpath(args.output):
            raise ValueError("输出目录不能与解析后的输入模型目录相同")
        manifest["resolved_model"] = model_path
        _emit("validate", "参数校验通过", method=args.method,
              model_source=args.model_source)
        if args.method == "gptq":
            result = quantize_gptq(model_path, args.output, args.bits, args.group_size,
                                   args.dataset, args.nsamples)
        elif args.method == "awq":
            result = quantize_awq(model_path, args.output, args.bits,
                                  args.dataset, args.nsamples)
        else:
            result = quantize_bnb(model_path, args.output, args.bits)
        manifest.update(result)
        manifest["status"] = "success"
        manifest["artifacts"] = _output_summary(args.output)
        _emit("complete", "量化完成", output=args.output, **manifest["artifacts"])
    except Exception as exc:
        manifest.update({"status": "failed", "error": str(exc),
                         "error_type": type(exc).__name__, "traceback": traceback.format_exc()})
        _emit("failed", "量化失败", error=str(exc), error_type=type(exc).__name__)
    manifest["finished_at"] = _utc_now()
    manifest["duration_seconds"] = round(time.monotonic() - started_clock, 3)
    try:
        manifest["manifest_path"] = _write_manifest(args.output, manifest)
    except Exception as exc:
        manifest["manifest_error"] = f"{type(exc).__name__}: {exc}"
        _emit("manifest_failed", "量化清单写入失败", error=str(exc))
    _finish_monitor(
        monitor,
        "SUCCEEDED" if manifest["status"] == "success" else "FAILED",
        manifest,
    )
    return manifest


def main(argv=None):
    manifest = run(build_parser().parse_args(argv))
    print(json.dumps(manifest, ensure_ascii=False))
    return 0 if manifest["status"] in {"success", "skipped"} else 1


if __name__ == "__main__":
    sys.exit(main())
