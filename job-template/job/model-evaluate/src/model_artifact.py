#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolve a model_path into a full HuggingFace model or a LoRA checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ADAPTER_WEIGHT_FILES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
)
FULL_WEIGHT_PREFIXES = (
    "model.safetensors",
    "pytorch_model.bin",
)


@dataclass
class ModelArtifact:
    artifact_type: str
    model_path: str
    load_path: str
    base_model_path: str = ""
    adapter_path: str = ""
    fingerprints: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取模型配置 {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"模型配置必须是 JSON 对象: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint_selected_files(directory: Path, names: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in names:
        path = directory / name
        if path.is_file():
            result[name] = _sha256(path)
    return result


def _find_base_model(adapter_dir: Path, adapter_config: dict) -> str:
    candidates: list[str] = []
    configured = adapter_config.get("base_model_name_or_path")
    if configured:
        candidates.append(str(configured))

    args_path = adapter_dir / "args.json"
    if args_path.is_file():
        args = _read_json(args_path)
        for key in ("model", "model_id_or_path", "model_name_or_path"):
            if args.get(key):
                candidates.append(str(args[key]))

    for candidate in candidates:
        expanded = os.path.expandvars(os.path.expanduser(candidate))
        if os.path.isabs(expanded):
            if os.path.exists(expanded):
                return os.path.realpath(expanded)
            continue

        relative = (adapter_dir / expanded).resolve()
        if relative.exists():
            return str(relative)

        # HuggingFace/ModelScope model IDs are allowed when they are not local paths.
        if "/" in candidate and not candidate.startswith((".", os.sep)):
            return candidate

    shown = ", ".join(candidates) if candidates else "未记录"
    raise FileNotFoundError(
        "检测到 LoRA checkpoint，但无法解析可访问的基础模型。"
        f" adapter_dir={adapter_dir}, candidates={shown}"
    )


def resolve_model_artifact(model_path: str) -> ModelArtifact:
    """Resolve one user-facing model_path.

    Remote model IDs are treated as full models. Local directories are validated.
    """
    raw_path = model_path.strip()
    if not raw_path:
        raise ValueError("model_path 不能为空")

    expanded = os.path.expandvars(os.path.expanduser(raw_path))
    looks_local = os.path.isabs(expanded) or expanded.startswith(".")
    if not looks_local and not os.path.exists(expanded):
        return ModelArtifact(
            artifact_type="remote_model",
            model_path=raw_path,
            load_path=raw_path,
            fingerprints={},
        )

    directory = Path(expanded).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"模型目录不存在: {directory}")

    adapter_config_path = directory / "adapter_config.json"
    adapter_weight = next(
        (directory / name for name in ADAPTER_WEIGHT_FILES
         if (directory / name).is_file()),
        None,
    )
    if adapter_config_path.is_file() or adapter_weight is not None:
        if not adapter_config_path.is_file():
            raise FileNotFoundError(f"LoRA 目录缺少 adapter_config.json: {directory}")
        if adapter_weight is None:
            raise FileNotFoundError(f"LoRA 目录缺少 Adapter 权重: {directory}")

        adapter_config = _read_json(adapter_config_path)
        base_model_path = _find_base_model(directory, adapter_config)
        fingerprints = _fingerprint_selected_files(
            directory,
            ["adapter_config.json", adapter_weight.name, "args.json"],
        )
        return ModelArtifact(
            artifact_type="lora_adapter",
            model_path=str(directory),
            load_path=base_model_path,
            base_model_path=base_model_path,
            adapter_path=str(directory),
            fingerprints=fingerprints,
        )

    config_path = directory / "config.json"
    has_full_weight = any((directory / name).is_file() for name in FULL_WEIGHT_PREFIXES)
    shard_patterns = (
        "model-*.safetensors", "model.safetensors-*-of-*.safetensors",
        "pytorch_model-*.bin", "pytorch_model.bin-*-of-*.bin",
    )
    has_full_weight = has_full_weight or any(
        path.is_file() for pattern in shard_patterns for path in directory.glob(pattern)
    )
    # Transformers discovers shards through the standard index filename;
    # filenames in weight_map need not follow a particular shard naming style.
    for index_name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
        index_path = directory / index_name
        if not index_path.is_file():
            continue
        mapping = _read_json(index_path).get('weight_map')
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"模型索引缺少有效 weight_map: {index_path}")
        for filename in mapping.values():
            if not isinstance(filename, str) or not filename:
                raise ValueError(f"模型索引中的权重文件名无效: {index_path}")
            shard = (directory / filename).resolve()
            if directory not in shard.parents:
                raise ValueError(f"模型索引权重路径超出模型目录: {filename}")
            if not shard.is_file():
                raise FileNotFoundError(f"模型索引引用的权重文件不存在: {shard}")
        has_full_weight = True
    if not config_path.is_file():
        raise FileNotFoundError(f"完整模型目录缺少 config.json: {directory}")
    if not has_full_weight:
        raise FileNotFoundError(f"完整模型目录缺少模型权重文件: {directory}")

    fingerprints = _fingerprint_selected_files(directory, ["config.json"])
    index_files = sorted(directory.glob("*.index.json"))
    for index_file in index_files:
        fingerprints[index_file.name] = _sha256(index_file)

    return ModelArtifact(
        artifact_type="full_model",
        model_path=str(directory),
        load_path=str(directory),
        fingerprints=fingerprints,
    )


def configure_adapter_environment(artifact: ModelArtifact) -> None:
    """Tell sitecustomize to attach the adapter in OpenCompass subprocesses."""
    if artifact.artifact_type == "lora_adapter":
        os.environ["MLOPS_LORA_ADAPTER_PATH"] = artifact.adapter_path
    else:
        os.environ.pop("MLOPS_LORA_ADAPTER_PATH", None)


def select_model_loader(model_path):
    """Preserve the configured multimodal wrapper, even for text-only evaluation."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
    from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    expected = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.get(config.model_type)
    candidates = (expected,) if isinstance(expected, str) else (expected or ())
    if any(name in candidates for name in (config.architectures or [])):
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


def load_model_and_tokenizer(artifact: ModelArtifact, model_kwargs=None, require_chat=True):
    """Load one effective model for native custom-dataset evaluation."""
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        artifact.load_path,
        trust_remote_code=True,
    )
    if require_chat and not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            f"模型 tokenizer 未提供 chat_template: {artifact.load_path}"
        )

    kwargs = dict(trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
    kwargs.update(model_kwargs or {})
    loader = select_model_loader(artifact.load_path)
    print(f'[INFO] model loader={loader.__name__} base={artifact.load_path}', flush=True)
    model = loader.from_pretrained(artifact.load_path, **kwargs)
    if artifact.artifact_type == "lora_adapter":
        import re
        adapter_config = _read_json(Path(artifact.adapter_path) / 'adapter_config.json')
        targets = adapter_config.get('target_modules')
        if isinstance(targets, str) and targets != 'all-linear':
            matched = [name for name, _ in model.named_modules() if re.fullmatch(targets, name)]
            if not matched:
                raise ValueError(f'LoRA 目标层与 {type(model).__name__} 不匹配: {targets}')
            print(f'[INFO] LoRA matched target modules={len(matched)}', flush=True)
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("业务镜像缺少 peft，无法加载 LoRA Adapter") from exc
        model = PeftModel.from_pretrained(
            model,
            artifact.adapter_path,
            is_trainable=False,
        )
        print(f'[INFO] LoRA adapter loaded: {artifact.adapter_path}; active={model.active_adapters}', flush=True)
    model.eval()
    return model, tokenizer
