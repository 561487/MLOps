"""Qwen3.5-compatible text distillation trainer.

This module intentionally uses the Transformers Trainer directly.  It keeps the
model-distillation pipeline node independent and avoids coupling its runtime to
the parallel ms-swift node.
"""
import argparse
import hashlib
import inspect
import json
import logging
import math
import os
from typing import Dict, Iterable, List

import jsonlines
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

SYSTEM_PROMPT = "You are a helpful assistant."
LOGGER = logging.getLogger(__name__)


def _detect_format(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return "parquet" if ext == ".parquet" else "csv" if ext == ".csv" else "json"


def _vocab_size(config) -> int:
    value = getattr(config, "vocab_size", None)
    if value is None and getattr(config, "text_config", None) is not None:
        value = getattr(config.text_config, "vocab_size", None)
    if value is None:
        raise ValueError("模型配置缺少 vocab_size/text_config.vocab_size")
    return int(value)


def _tokenizer_fingerprint(tokenizer) -> str:
    vocab = tokenizer.get_vocab()
    digest = hashlib.sha256()
    for token, token_id in sorted(vocab.items(), key=lambda item: item[1]):
        digest.update(str(token_id).encode("ascii"))
        digest.update(b"\0")
        digest.update(token.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _select_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _load_student(model_path: str):
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=_select_dtype(),
            low_cpu_mem_usage=True,
        )
    except (KeyError, ValueError) as exc:
        raise RuntimeError(
            "无法加载学生模型。Qwen3.5 需要 Transformers 5.9+，"
            "并要求其 CausalLM 权重映射可用。"
        ) from exc


def _messages(instruction, output=None):
    result = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(instruction or "")},
    ]
    if output is not None:
        result.append({"role": "assistant", "content": str(output or "")})
    return result


def _prepare_dataset(dataset, tokenizer, renderer, inst_col: str, out_col: str, max_length: int):
    source_columns = dataset.column_names

    def tokenize(example, index):
        full_text = renderer.apply_chat_template(
            _messages(example.get(inst_col), example.get(out_col)),
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = renderer.apply_chat_template(
            _messages(example.get(inst_col)),
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
        )
        prompt_ids = tokenizer(
            prompt_text,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
        )["input_ids"]
        labels = list(encoded["input_ids"])
        prompt_length = min(len(prompt_ids), len(labels))
        labels[:prompt_length] = [-100] * prompt_length
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
            "sample_id": int(index),
        }

    return dataset.map(
        tokenize,
        with_indices=True,
        remove_columns=source_columns,
        desc="Tokenizing distillation dataset",
    )


class DistillCollator:
    def __init__(self, tokenizer):
        self.base = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )

    def __call__(self, features):
        sample_ids = [int(feature["sample_id"]) for feature in features]
        model_features = [
            {key: value for key, value in feature.items() if key != "sample_id"}
            for feature in features
        ]
        batch = self.base(model_features)
        batch["sample_id"] = torch.tensor(sample_ids, dtype=torch.long)
        return batch


def _legacy_sparse_row(row):
    indices, values = [], []
    for position in row:
        ids = [int(key) for key in position.keys()]
        probs = [max(float(position[str(key)]), 1e-30) for key in ids]
        indices.append(ids)
        values.append([math.log(prob) for prob in probs])
    return {"topk_indices": indices, "topk_logits": values}


def _load_sparse_logits(path: str) -> Dict[int, dict]:
    rows = {}
    with jsonlines.open(path) as reader:
        for fallback_id, row in enumerate(reader):
            sample_id = int(row.get("sample_id", fallback_id)) if isinstance(row, dict) else fallback_id
            rows[sample_id] = row if isinstance(row, dict) else _legacy_sparse_row(row)
    if not rows:
        raise ValueError(f"Teacher logits 文件为空: {path}")
    return rows


class SparseDistillTrainer(Trainer):
    def __init__(
        self,
        *args,
        logits_path: str,
        kd_ratio: float,
        temperature: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_rows = _load_sparse_logits(logits_path)
        self.kd_ratio = float(kd_ratio)
        self.temperature = float(temperature)
        if not 0.0 <= self.kd_ratio <= 1.0:
            raise ValueError("kd_ratio 必须在 0~1")
        if self.temperature <= 0:
            raise ValueError("temperature 必须大于 0")

    def _sparse_forward_kl(self, student_logits, labels, sample_ids):
        losses = []
        student_vocab = student_logits.size(-1)
        temperature = self.temperature

        for batch_index, sample_id in enumerate(sample_ids.tolist()):
            if sample_id not in self.teacher_rows:
                raise KeyError(f"缺少 sample_id={sample_id} 的 Teacher logits")
            row = self.teacher_rows[sample_id]
            all_indices = row["topk_indices"]
            all_logits = row["topk_logits"]
            limit = min(student_logits.size(1) - 1, labels.size(1) - 1, len(all_indices) - 1)
            if limit <= 0:
                continue

            active_positions = torch.nonzero(
                labels[batch_index, 1 : limit + 1].ne(-100), as_tuple=False
            ).flatten()
            for position in active_positions.tolist():
                ids = torch.tensor(
                    all_indices[position], dtype=torch.long, device=student_logits.device
                )
                if ids.numel() == 0:
                    continue
                if int(ids.max()) >= student_vocab or int(ids.min()) < 0:
                    raise ValueError(
                        f"Teacher/Student 词表不兼容: token id 超出 Student vocab={student_vocab}"
                    )
                teacher_values = torch.tensor(
                    all_logits[position],
                    dtype=torch.float32,
                    device=student_logits.device,
                )
                teacher_log_probs = F.log_softmax(teacher_values / temperature, dim=-1)
                student_values = student_logits[batch_index, position].float() / temperature
                student_log_probs = (
                    student_values.gather(0, ids) - torch.logsumexp(student_values, dim=-1)
                )
                losses.append(
                    torch.sum(teacher_log_probs.exp() * (teacher_log_probs - student_log_probs))
                )

        if not losses:
            return student_logits.new_zeros((), dtype=torch.float32)
        return torch.stack(losses).mean() * (temperature ** 2)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        sample_ids = inputs.pop("sample_id")
        outputs = model(**inputs)
        lm_loss = outputs.loss
        kd_loss = self._sparse_forward_kl(outputs.logits, inputs["labels"], sample_ids)
        total_loss = (1.0 - self.kd_ratio) * lm_loss + self.kd_ratio * kd_loss
        return (total_loss, outputs) if return_outputs else total_loss


def _training_arguments(config: dict, dtype):
    values = dict(config)
    values.pop("max_length", None)
    values["remove_unused_columns"] = False
    values["bf16"] = dtype == torch.bfloat16
    values["fp16"] = dtype == torch.float16
    signature = inspect.signature(TrainingArguments.__init__)
    supported = {key: value for key, value in values.items() if key in signature.parameters}
    ignored = sorted(set(values) - set(supported))
    if ignored:
        LOGGER.warning("忽略当前 Transformers 不支持的训练参数: %s", ignored)
    return TrainingArguments(**supported)


def train(config):
    dataset_config = config["dataset"]
    training_config = config["training"]
    data_path = dataset_config["labeled_path"]
    dataset = load_dataset(
        _detect_format(data_path),
        data_files=data_path,
        split="train",
    )
    max_samples = int(dataset_config.get("max_samples", 0) or 0)
    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["student"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    renderer = tokenizer
    if "kd_white_box" in config["job_type"]:
        renderer = AutoTokenizer.from_pretrained(
            config["models"]["teacher"],
            trust_remote_code=True,
        )
    if renderer.chat_template is None:
        with open(dataset_config["template"], encoding="utf-8") as handle:
            renderer.chat_template = handle.read()
    if renderer.pad_token is None:
        renderer.pad_token = renderer.eos_token

    max_length = int(training_config.get("max_length", 512))
    tokenized = _prepare_dataset(
        dataset,
        tokenizer,
        renderer,
        dataset_config.get("inst_col", "instruction"),
        dataset_config.get("out_col", "output"),
        max_length,
    )
    model = _load_student(config["models"]["student"])
    dtype = _select_dtype()
    args = _training_arguments(training_config, dtype)
    common = {
        "model": model,
        "args": args,
        "train_dataset": tokenized,
        "data_collator": DistillCollator(tokenizer),
    }

    if "kd_white_box" in config["job_type"]:
        meta_path = dataset_config["logits_path"] + ".meta.json"
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as handle:
                metadata = json.load(handle)
            student_vocab = _vocab_size(model.config)
            if int(metadata["vocab_size"]) != student_vocab:
                raise ValueError(
                    f"Teacher vocab={metadata['vocab_size']} 与 Student vocab={student_vocab} 不一致"
                )
            fingerprint = _tokenizer_fingerprint(tokenizer)
            if metadata.get("tokenizer_fingerprint") != fingerprint:
                raise ValueError("Teacher 与 Student tokenizer 的 token-id 映射不一致")

        trainer = SparseDistillTrainer(
            **common,
            logits_path=dataset_config["logits_path"],
            kd_ratio=config["distillation"]["kd_ratio"],
            temperature=config["distillation"].get("temperature", 1.0),
        )
    elif "kd_black_box" in config["job_type"]:
        trainer = Trainer(**common)
    else:
        raise ValueError(f"不支持的 job_type: {config['job_type']}")

    trainer.train(resume_from_checkpoint=training_config.get("resume_from_checkpoint"))
    trainer.save_model(training_config["output_dir"])
    tokenizer.save_pretrained(training_config["output_dir"])


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        train(json.load(handle))


if __name__ == "__main__":
    main()
