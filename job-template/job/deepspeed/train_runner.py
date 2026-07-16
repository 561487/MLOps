#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def split_csv(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_precision_options(precision):
    return {
        "fp16": precision == "fp16",
        "bf16": precision == "bf16",
    }


def build_training_options(args):
    lora = None
    if str2bool(args.use_lora):
        lora = {
            "rank": int(args.lora_rank),
            "alpha": int(args.lora_alpha),
            "dropout": float(args.lora_dropout),
            "target_modules": split_csv(args.lora_target),
        }
    return {
        "mode": args.mode,
        "model_name_or_path": args.model_name_or_path,
        "dataset_path": args.dataset_path,
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "model_config_path": args.model_config_path,
        "output_dir": args.output_dir,
        "num_train_epochs": float(args.num_train_epochs),
        "max_steps": int(args.max_steps),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "per_device_train_batch_size": int(args.per_device_train_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "max_seq_length": int(args.max_seq_length),
        "logging_steps": int(args.logging_steps),
        "save_steps": int(args.save_steps),
        "precision": args.precision,
        "deepspeed_config": args.deepspeed_config,
        "lora": lora,
    }


def build_custom_command(args):
    command = args.command.strip()
    if not command:
        raise ValueError("--command is required in custom mode")
    if str2bool(args.append_config_to_command):
        config_arg_name = args.config_arg_name
        if not config_arg_name.startswith("--"):
            config_arg_name = "--" + config_arg_name
        command = "%s %s %s" % (
            command,
            shlex.quote(config_arg_name),
            shlex.quote(args.deepspeed_config),
        )
    return command


def save_training_artifacts(trainer, tokenizer, output_dir):
    trainer.save_model(output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output_dir)


def _load_dataset(dataset_path):
    from datasets import load_dataset

    if os.path.exists(dataset_path):
        if os.path.isdir(dataset_path):
            return load_dataset(dataset_path, split="train")
        ext = os.path.splitext(dataset_path)[1].lower()
        if ext in {".json", ".jsonl"}:
            return load_dataset("json", data_files=dataset_path, split="train")
        if ext == ".csv":
            return load_dataset("csv", data_files=dataset_path, split="train")
        return load_dataset("text", data_files=dataset_path, split="train")
    return load_dataset(dataset_path, split="train")


def _row_to_text(row):
    for key in ("text", "content", "prompt", "instruction"):
        if key in row and row[key]:
            return str(row[key])
    return "\n".join(str(value) for value in row.values() if value is not None)


def _tokenize_dataset(dataset, tokenizer, max_seq_length):
    column_names = list(dataset.column_names)

    def tokenize(batch):
        rows = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]
        texts = [_row_to_text(row) for row in rows]
        return tokenizer(texts, truncation=True, max_length=max_seq_length)

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=column_names,
        desc="Tokenizing dataset",
    )

    def add_labels(batch):
        batch["labels"] = [ids[:] for ids in batch["input_ids"]]
        return batch

    return tokenized.map(add_labels, batched=True, desc="Adding labels")


def run_hf_causal_lm(args):
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    tokenizer_path = args.tokenizer_name_or_path or args.model_name_or_path
    if not tokenizer_path:
        raise ValueError("--tokenizer_name_or_path or --model_name_or_path is required")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.mode == "pretrain" and args.model_config_path:
        config = AutoConfig.from_pretrained(args.model_config_path)
        model = AutoModelForCausalLM.from_config(config)
    else:
        if not args.model_name_or_path:
            raise ValueError("--model_name_or_path is required for finetune")
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    if str2bool(args.use_lora):
        from peft import LoraConfig, get_peft_model

        lora_kwargs = {
            "r": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        targets = split_csv(args.lora_target)
        if targets:
            lora_kwargs["target_modules"] = targets
        model = get_peft_model(model, LoraConfig(**lora_kwargs))
        model.print_trainable_parameters()

    dataset = _load_dataset(args.dataset_path)
    train_dataset = _tokenize_dataset(dataset, tokenizer, args.max_seq_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps if int(args.max_steps) > 0 else -1,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        deepspeed=args.deepspeed_config,
        report_to=[],
        **build_precision_options(args.precision),
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    save_training_artifacts(trainer, tokenizer, args.output_dir)


def run_custom(args):
    if args.working_dir:
        os.chdir(args.working_dir)
    command = build_custom_command(args)
    print("[train_runner] execute custom command: %s" % command, flush=True)
    return subprocess.call(command, shell=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser("DeepSpeed train runner")
    parser.add_argument("--mode", choices=["finetune", "pretrain", "custom"], default="finetune")
    parser.add_argument("--model_name_or_path", default="")
    parser.add_argument("--dataset_path", default="")
    parser.add_argument("--tokenizer_name_or_path", default="")
    parser.add_argument("--model_config_path", default="")
    parser.add_argument("--output_dir", default="/tmp/deepspeed-output")
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument(
        "--precision", choices=["fp16", "bf16", "fp32"], default="fp16"
    )
    parser.add_argument("--use_lora", type=str2bool, default=False)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target", default="")
    parser.add_argument("--deepspeed_config", "--deepspeed", default="/tmp/ds_config.json")
    parser.add_argument("--working_dir", default="")
    parser.add_argument("--command", default="")
    parser.add_argument("--append_config_to_command", type=str2bool, default=True)
    parser.add_argument(
        "--config_arg_name",
        default="--deepspeed",
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    args, _ = parser.parse_known_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)
    print("[train_runner] options: %s" % build_training_options(args), flush=True)
    if args.mode == "custom":
        return run_custom(args)
    if not args.dataset_path:
        raise ValueError("--dataset_path is required for %s mode" % args.mode)
    run_hf_causal_lm(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
