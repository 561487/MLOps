#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PVC QA dataset loading, validation, deterministic inference and scoring."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from model_artifact import ModelArtifact, load_model_and_tokenizer


SUPPORTED_EXTENSIONS = {".json", ".jsonl", ".csv"}
INPUT_FIELDS = ("input", "question", "query", "prompt")
TARGET_FIELDS = ("target", "answer")
VALID_TYPES = {"choice", "multi_choice", "short_answer", "numeric", "contains"}
MAX_INPUT_TOKENS = 2048
MAX_NEW_TOKENS = {
    "choice": 64,
    "multi_choice": 128,
    "numeric": 128,
    "short_answer": 512,
    "contains": 512,
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def discover_dataset_files(dataset_path: str) -> list[Path]:
    path = Path(os.path.expandvars(os.path.expanduser(dataset_path))).resolve()
    if not path.exists():
        raise FileNotFoundError(f"自定义数据集路径不存在: {path}")
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的数据格式: {path.suffix}")
        return [path]
    for name in ('dataset_manifest.json', 'manifest.json'):
        manifest = path / name
        if manifest.is_file():
            info = json.loads(manifest.read_text(encoding='utf-8-sig'))
            filename = info.get('output_file') or info.get('data_file')
            if not isinstance(filename, str) or not filename:
                raise ValueError('manifest 缺少 output_file/data_file')
            data = (path / filename).resolve()
            if path not in data.parents or not data.is_file() or data.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError('manifest 数据文件无效或不在输入目录内')
            return [data]
    files = sorted(
        item for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS
        and item.name not in {'rejected.jsonl', 'conversion_preview.json', 'conflicts.jsonl'}
        and not item.name.endswith('_report.json')
    )
    if not files:
        raise FileNotFoundError(f"目录下没有 JSON、JSONL 或 CSV 文件: {path}")
    return files


def _read_file(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return [dict(row) for row in csv.DictReader(stream)]

    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        try:
            value = json.loads(text)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                for key in ("data", "items", "records"):
                    if isinstance(value.get(key), list):
                        return value[key]
                return [value]
        except json.JSONDecodeError:
            pass

    values = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path}:{line_number} 不是合法 JSON: {exc}"
            ) from exc
    return values


def _choose_field(record: dict, candidates: tuple[str, ...], label: str) -> str:
    found = [name for name in candidates if name in record]
    if len(found) > 1:
        raise ValueError(
            f"{label}字段不明确，同时存在 {found}；请先通过 dataset-convert 统一为 eval_qa"
        )
    if not found:
        raise ValueError(
            f"缺少{label}字段，支持字段: {list(candidates)}"
        )
    return found[0]


def _parse_json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _normalize_targets(value: Any) -> list[str]:
    value = _parse_json_value(value)
    if isinstance(value, list):
        result = [str(item).strip() for item in value if str(item).strip()]
    else:
        result = [str(value).strip()] if value is not None else []
    return result


def _normalize_choices(record: dict) -> list[dict]:
    raw = _parse_json_value(record.get("choices"))
    choices: list[dict] = []
    if isinstance(raw, dict):
        for key, value in raw.items():
            choices.append({"label": str(key).upper(), "text": str(value)})
    elif isinstance(raw, list):
        for index, value in enumerate(raw):
            if isinstance(value, dict):
                label = str(value.get("label", chr(65 + index))).upper()
                text = str(value.get("text", value.get("value", "")))
            else:
                label = chr(65 + index)
                text = str(value)
            choices.append({"label": label, "text": text})

    if not choices:
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            if letter in record and str(record[letter]).strip():
                choices.append({"label": letter, "text": str(record[letter])})
            elif choices:
                break
    return choices


def load_and_validate(dataset_path: str) -> tuple[list[dict], list[dict], dict]:
    valid: list[dict] = []
    invalid: list[dict] = []
    seen_ids: set[str] = set()
    mappings: dict[str, dict] = {}

    for file_path in discover_dataset_files(dataset_path):
        try:
            records = _read_file(file_path)
        except Exception as exc:
            invalid.append({
                "source_file": str(file_path),
                "error_type": "file_parse_error",
                "reason": str(exc),
            })
            continue

        if not records:
            invalid.append({
                "source_file": str(file_path),
                "error_type": "empty_file",
                "reason": "文件中没有样本",
            })
            continue
        if not isinstance(records[0], dict):
            invalid.append({
                "source_file": str(file_path),
                "error_type": "invalid_record",
                "reason": "样本必须是 JSON 对象",
            })
            continue

        try:
            input_field = _choose_field(records[0], INPUT_FIELDS, "输入")
            target_field = _choose_field(records[0], TARGET_FIELDS, "参考答案")
            mappings[str(file_path)] = {
                "input": input_field,
                "target": target_field,
            }
        except ValueError as exc:
            invalid.append({
                "source_file": str(file_path),
                "error_type": "schema_error",
                "reason": str(exc),
            })
            continue

        for index, raw in enumerate(records, 1):
            source = {"source_file": str(file_path), "source_line": index}
            if not isinstance(raw, dict):
                invalid.append({
                    **source,
                    "error_type": "invalid_record",
                    "reason": "样本必须是对象",
                })
                continue
            if "messages" in raw and input_field not in raw:
                invalid.append({
                    **source,
                    "error_type": "messages_not_supported",
                    "reason": "messages 必须先通过 dataset-convert 转为 eval_qa",
                })
                continue

            question = str(raw.get(input_field, "") or "").strip()
            targets = _normalize_targets(raw.get(target_field))
            sample_id = str(raw.get("id", "") or "").strip()
            if not sample_id:
                sample_id = f"{file_path.name}:{index}"

            if sample_id in seen_ids:
                invalid.append({
                    **source,
                    "sample_id": sample_id,
                    "error_type": "duplicate_id",
                    "reason": "样本 ID 重复",
                })
                continue
            if not question or not targets:
                invalid.append({
                    **source,
                    "sample_id": sample_id,
                    "error_type": "empty_required_field",
                    "reason": "输入或参考答案为空",
                })
                continue

            choices = _normalize_choices(raw)
            keywords = _parse_json_value(raw.get("keywords", []))
            if isinstance(keywords, str):
                keywords = [item.strip() for item in keywords.split(",") if item.strip()]
            if not isinstance(keywords, list):
                keywords = []

            sample_type = str(raw.get("type", "") or "").strip().lower()
            if not sample_type:
                sample_type = "choice" if choices else (
                    "contains" if keywords else "short_answer"
                )
            if sample_type not in VALID_TYPES:
                invalid.append({
                    **source,
                    "sample_id": sample_id,
                    "error_type": "unsupported_type",
                    "reason": f"不支持的 type: {sample_type}",
                })
                continue
            if sample_type in ("choice", "multi_choice") and len(choices) < 2:
                invalid.append({
                    **source,
                    "sample_id": sample_id,
                    "error_type": "invalid_choices",
                    "reason": "选择题至少需要两个选项",
                })
                continue
            if sample_type == "multi_choice":
                labels = [item['label'] for item in choices]
                if (not isinstance(raw.get(target_field), list)
                        or len(set(labels)) != len(labels)
                        or any(not re.fullmatch('[A-Z]', item) for item in labels)
                        or any(item not in labels for item in targets)
                        or len(set(targets)) != len(targets)):
                    invalid.append({**source, 'sample_id': sample_id,
                                    'error_type': 'invalid_multi_choice_target',
                                    'reason': '多选答案须为有效且不重复的标签数组'})
                    continue
            if sample_type == "numeric":
                try:
                    _extract_number(targets[0])
                except ValueError as exc:
                    invalid.append({
                        **source,
                        "sample_id": sample_id,
                        "error_type": "invalid_numeric_target",
                        "reason": str(exc),
                    })
                    continue

            seen_ids.add(sample_id)
            if not isinstance(raw.get('system', ''), str):
                invalid.append({**source, 'error_type': 'invalid_system', 'reason': 'system 必须是字符串'})
                continue
            valid.append({
                **source,
                "sample_id": sample_id,
                "input": question,
                "system": raw.get('system', ''),
                "targets": targets,
                "type": sample_type,
                "choices": choices,
                "category": str(raw.get("category", "") or ""),
                "difficulty": str(raw.get("difficulty", "") or ""),
                "keywords": [str(item) for item in keywords],
            })

    metadata = {
        "files": [str(path) for path in discover_dataset_files(dataset_path)],
        "field_mappings": mappings,
        "valid_count": len(valid),
        "invalid_count": len(invalid),
    }
    return valid, invalid, metadata


def clean_answer_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").lower().strip()
    value = re.sub(r"^(answer|答案|最终答案)\s*[:：]\s*", "", value)
    return value


def normalize_text(value: str) -> str:
    value = clean_answer_text(value)
    value = re.sub(r"[\s\W_]+", "", value, flags=re.UNICODE)
    return value


def _tokens(value: str) -> list[str]:
    value = clean_answer_text(value)
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", value)


def token_f1(prediction: str, reference: str) -> tuple[float, float, float]:
    pred_tokens = _tokens(prediction)
    ref_tokens = _tokens(reference)
    if not pred_tokens or not ref_tokens:
        equal = float(pred_tokens == ref_tokens)
        return equal, equal, equal
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    return 2 * precision * recall / (precision + recall), precision, recall


def _extract_number(value: str) -> float:
    matches = re.findall(
        r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?",
        value or "",
    )
    if not matches:
        raise ValueError(f"无法提取数值: {value!r}")
    return float(matches[-1].replace(",", ""))


def _extract_choice(prediction: str, choices: list[dict]) -> str:
    answer, _, _ = parse_choice_answer(prediction, choices)
    return answer[0] if answer else ""


SCORING_VERSION = 'choice-strict-v2'


def parse_choice_answer(prediction, choices, multiple=False):
    """Parse explicit answers, never isolated option mentions in reasoning."""
    text = unicodedata.normalize('NFKC', prediction or '')
    if '<think>' in text and '</think>' not in text:
        return [], 'unclosed_reasoning', 'answer_parse_error'
    text = text.split('</think>')[-1].strip()
    text = text.replace('**', '').replace('`', '')
    labels = {str(c['label']).upper(): str(c['label']) for c in choices}

    def payload(value):
        value = value.strip().strip('[]()').strip().rstrip('.。').strip()
        # Entire option text only; duplicated option text is ambiguous.
        matches = [str(c['label']) for c in choices
                   if normalize_text(c['text']) == normalize_text(value)]
        if len(matches) > 1:
            return [], 'answer_ambiguous'
        if len(matches) == 1:
            return matches, ''
        tokens = re.split(r'\s*(?:[,，、;；]|\band\b|\s+)\s*', value, flags=re.I)
        if tokens and all(t.upper() in labels for t in tokens):
            result = sorted({labels[t.upper()] for t in tokens})
            if multiple or len(result) == 1:
                return result, ''
            return [], 'answer_ambiguous'
        return [], 'answer_parse_error'

    marker = re.compile(r'(?:\b(?:final\s+answer|correct\s+answer|answer)\b|最终答案|正确答案|答案)\s*(?:is\b|为|是|[:：])\s*[:：]?\s*', re.I)
    markers = list(marker.finditer(text))
    if markers:
        answers = []
        for i, match in enumerate(markers):
            end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
            clause = text[match.end():end].strip().split('\n')[0]
            answer, error = payload(clause)
            # Permit a clearly delimited answer followed by an explanation.
            if error == 'answer_parse_error':
                prefix = re.split(r'[.。:：]|\s+[（(]|\s*[-—]\s*', clause, maxsplit=1)[0]
                if prefix != clause:
                    answer, error = payload(prefix)
            if error:
                return [], 'explicit', error
            answers.append(answer)
        if any(a != answers[0] for a in answers[1:]):
            return [], 'explicit', 'answer_ambiguous'
        return answers[0], 'explicit', ''
    answer, error = payload(text)
    return answer, 'whole_response', error


def _reference_choice(targets: list[str], choices: list[dict]) -> str:
    labels = [choice["label"] for choice in choices]
    for target in targets:
        candidate = target.strip().upper()
        if candidate in labels:
            return candidate
        normalized = normalize_text(target)
        for choice in choices:
            if normalize_text(choice["text"]) == normalized:
                return choice["label"]
    return ""


def score_prediction(sample: dict, prediction: str) -> dict:
    clean_prediction = prediction.split("</think>")[-1].strip()
    targets = sample["targets"]
    result = {
        "metric_name": "",
        "score": 0.0,
        "passed": False,
    }

    if sample["type"] in ("choice", "multi_choice"):
        multiple = sample['type'] == 'multi_choice'
        predicted, method, error = parse_choice_answer(prediction, sample['choices'], multiple)
        reference = []
        for target in targets:
            labels, _, ref_error = parse_choice_answer(target, sample['choices'], multiple)
            if ref_error:
                raise ValueError('Invalid choice reference: ' + repr(target))
            reference.extend(labels)
        reference = sorted(set(reference))
        if not multiple and len(reference) != 1:
            raise ValueError('Single-choice reference must identify exactly one option')
        passed = not error and predicted == reference
        result.update(metric_name='multi_choice_exact_match' if multiple else 'choice_accuracy',
                      score=float(passed), passed=passed,
                      parsed_answer=predicted if multiple else (predicted[0] if predicted else ''),
                      reference_parsed=reference if multiple else reference[0],
                      scoring_version=SCORING_VERSION, answer_parse_method=method)
        if error:
            result['error_type'] = error
        return result

    if sample["type"] == "numeric":
        try:
            predicted_number = _extract_number(clean_prediction)
            reference_number = _extract_number(targets[0])
            tolerance = max(1e-6, abs(reference_number) * 1e-4)
            error = abs(predicted_number - reference_number)
            passed = error <= tolerance
            result.update({
                "metric_name": "numeric_accuracy",
                "score": float(passed),
                "passed": passed,
                "parsed_answer": predicted_number,
                "reference_parsed": reference_number,
                "absolute_error": error,
            })
        except ValueError:
            result.update({
                "metric_name": "numeric_accuracy",
                "error_type": "answer_parse_error",
            })
        return result

    if sample["type"] == "contains":
        keywords = sample["keywords"] or targets
        covered = [
            keyword for keyword in keywords
            if normalize_text(keyword) in normalize_text(clean_prediction)
        ]
        coverage = len(covered) / len(keywords) if keywords else 0.0
        result.update({
            "metric_name": "keypoint_coverage",
            "score": coverage,
            "passed": bool(keywords) and len(covered) == len(keywords),
            "covered_keywords": covered,
            "missing_keywords": [k for k in keywords if k not in covered],
        })
        return result

    exact = any(
        normalize_text(clean_prediction) == normalize_text(target)
        for target in targets
    )
    f1_values = [token_f1(clean_prediction, target) for target in targets]
    best_f1, best_precision, best_recall = max(
        f1_values,
        key=lambda item: item[0],
    )
    result.update({
        "metric_name": "exact_match",
        "score": float(exact),
        "passed": exact,
        "exact_match": float(exact),
        "token_f1": best_f1,
        "token_precision": best_precision,
        "token_recall": best_recall,
    })
    return result


PROMPT_VERSION = 'typed-label-v1'


def _choice_instruction(sample):
    labels = ', '.join(str(c['label']) for c in sample['choices'])
    chinese = bool(re.search(r'[\u4e00-\u9fff]', sample['input']))
    if sample['type'] == 'choice':
        return (f'本题为单选题。只能从以下标签中选择一个：{labels}。只输出该标签，不要解释、选项内容或答案前缀。'
                if chinese else
                f'This is a single-choice question. Choose exactly one label from: {labels}. '
                'Output only that label. Do not include explanations, option text, or an answer prefix.')
    return (f'本题为多选题。只能使用以下标签：{labels}。按选项顺序输出所选标签，以英文逗号分隔。不要解释、选项内容或答案前缀。'
            if chinese else
            f'This is a multiple-choice question. Select one or more labels from: {labels}. '
            'Output only selected labels, separated by commas, in the order the options appear. '
            'Do not include explanations, option text, or an answer prefix.')


def _check_system_format(sample):
    """Detect common explicit conflicts without rewriting business instructions."""
    system = sample.get('system', '')
    patterns = [r'输出.{0,8}答案\s*[:：]',
                r'(?:必须|务必)(?:提供|给出|输出).{0,4}(?:解释|推理过程)',
                r'(?:must|always)\s+(?:explain|provide an explanation)']
    if sample['type'] == 'multi_choice':
        patterns += [r'(?:choose|select) exactly one', r'只能.{0,4}一个']
    if any(re.search(pattern, system, re.I) for pattern in patterns):
        raise ValueError('system 中的作答要求与选择题模板冲突，请检查数据中的 system；组件不会自动删除业务指令')


def _build_prompt(sample: dict) -> str:
    if sample["type"] not in ("choice", "multi_choice"):
        return sample["input"]
    _check_system_format(sample)
    option_lines = [
        f'{choice["label"]}. {choice["text"]}'
        for choice in sample["choices"]
    ]
    return (
        sample["input"]
        + "\n\n"
        + "\n".join(option_lines)
        + "\n\n" + _choice_instruction(sample)
    )


def _build_messages(sample):
    messages = []
    if sample.get('system'):
        messages.append({'role': 'system', 'content': sample['system']})
    messages.append({'role': 'user', 'content': _build_prompt(sample)})
    return messages


def _model_device(model):
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return getattr(model, "device", "cuda")


def _generate_batch(model, tokenizer, samples, max_seq_len=MAX_INPUT_TOKENS,
              max_out_len=None, model_type='hf_chat') -> list[tuple[str, int, int, float]]:
    import torch

    prompts = []
    for sample in samples:
        prompt = _build_prompt(sample)
        if model_type == 'hf_chat':
            prompt = tokenizer.apply_chat_template(
                _build_messages(sample), tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
        elif sample.get('system'):
            prompt = sample['system'] + '\n\n' + prompt
        prompts.append(prompt)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError('批量推理需要 pad_token 或 eos_token')
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(prompts, padding=True, truncation=False, return_tensors='pt',
                        add_special_tokens=model_type != 'hf_chat')
    counts = encoded['attention_mask'].sum(dim=1).tolist()
    if max(counts) > max_seq_len:
        raise ValueError(f'输入长度 {max(counts)} 超过限制 {max_seq_len}')
    input_ids = encoded['input_ids'].to(_model_device(model))
    attention_mask = encoded['attention_mask'].to(_model_device(model))
    width = input_ids.shape[1]

    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_out_len or max(MAX_NEW_TOKENS[s['type']] for s in samples),
            pad_token_id=tokenizer.pad_token_id,
        )
    latency = time.perf_counter() - started
    if generated.shape[0] != len(samples):
        raise ValueError('生成结果数量与输入批次不一致')
    results = []
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    for index, row in enumerate(generated[:, width:].tolist()):
        end = next((i + 1 for i, token in enumerate(row) if token in eos), len(row))
        row = row[:end]
        results.append((tokenizer.decode(row, skip_special_tokens=True),
                        int(counts[index]), len(row), latency / len(samples)))
    return results


def _generate(model, tokenizer, sample, **kwargs):
    return _generate_batch(model, tokenizer, [sample], **kwargs)[0]


def _aggregate(details: list[dict], invalid: list[dict]) -> dict:
    successful = [item for item in details if not item.get("inference_error")]
    passed_count = sum(1 for item in successful if item.get("passed"))
    type_groups: dict[str, list[dict]] = defaultdict(list)
    category_groups: dict[str, list[dict]] = defaultdict(list)
    for item in details:
        type_groups[item["type"]].append(item)
        if item.get("category"):
            category_groups[item["category"]].append(item)

    def group_summary(items: list[dict]) -> dict:
        count = len(items)
        return {
            "count": count,
            "pass_rate": (
                sum(1 for item in items if item.get("passed")) / count
                if count else 0.0
            ),
            "average_score": (
                sum(float(item.get("score", 0.0)) for item in items) / count
                if count else 0.0
            ),
        }

    short_items = [
        item for item in successful if item["type"] == "short_answer"
    ]
    latencies = sorted(
        float(item["latency_seconds"])
        for item in successful if item.get("latency_seconds") is not None
    )

    def percentile(values: list[float], ratio: float) -> float:
        if not values:
            return 0.0
        index = min(len(values) - 1, math.ceil(len(values) * ratio) - 1)
        return values[index]

    return {
        "scoring_version": SCORING_VERSION,
        "total_count": len(details),
        "evaluation_complete": not any(x.get('inference_error') for x in details),
        "answer_parse_error_count": sum(x.get('error_type') == 'answer_parse_error' for x in successful),
        "answer_ambiguous_count": sum(x.get('error_type') == 'answer_ambiguous' for x in successful),
        "answer_parse_failure_rate": (
            sum(x.get('error_type') in ('answer_parse_error', 'answer_ambiguous') for x in successful)
            / len(successful) if successful else 0.0
        ),
        "output_limit_reached_count": sum(bool(x.get('output_limit_reached')) for x in successful),
        "output_limit_known_count": sum('output_limit_reached' in x for x in successful),
        "evaluated_count": len(successful),
        "invalid_count": len(invalid),
        "inference_error_count": sum(
            1 for item in details if item.get("inference_error")
        ),
        "passed_count": passed_count,
        "failed_count": len(successful) - passed_count,
        "pass_rate": passed_count / len(details) if details else 0.0,
        "exact_match": (
            sum(item.get("exact_match", 0.0) for item in short_items)
            / len(short_items) if short_items else None
        ),
        "token_f1": (
            sum(item.get("token_f1", 0.0) for item in short_items)
            / len(short_items) if short_items else None
        ),
        "by_type": {
            key: group_summary(value) for key, value in sorted(type_groups.items())
        },
        "by_category": {
            key: group_summary(value)
            for key, value in sorted(category_groups.items())
        },
        "performance": {
            "latency_average_seconds": (
                sum(latencies) / len(latencies) if latencies else 0.0
            ),
            "latency_p50_seconds": percentile(latencies, 0.50),
            "latency_p95_seconds": percentile(latencies, 0.95),
            "average_input_tokens": (
                sum(item.get("input_tokens", 0) for item in successful)
                / len(successful) if successful else 0.0
            ),
            "average_output_tokens": (
                sum(item.get("output_tokens", 0) for item in successful)
                / len(successful) if successful else 0.0
            ),
        },
    }


def evaluate_custom_dataset(
    dataset_path: str,
    output_path: str,
    artifact: ModelArtifact,
    max_samples=0, max_seq_len=MAX_INPUT_TOKENS, max_out_len=None,
    model_type='hf_chat', model_kwargs=None, batch_size=4,
) -> dict:
    output_dir = Path(output_path)
    valid, invalid, dataset_metadata = load_and_validate(dataset_path)
    if max_samples:
        valid = valid[:max_samples]
    dataset_metadata['selected_count'] = len(valid)
    _write_jsonl(output_dir / "custom_invalid.jsonl", invalid)
    if not valid:
        raise ValueError("自定义数据集没有可评测样本")

    print(
        f"[INFO] 自定义数据校验完成: valid={len(valid)}, invalid={len(invalid)}"
    )
    for sample in valid:
        _build_prompt(sample)  # Fail on explicit format conflicts before allocating the model.
    print(f'[INFO] evaluation prompt_version={PROMPT_VERSION}; choice labels are read from each sample', flush=True)
    model, tokenizer = load_model_and_tokenizer(
        artifact, model_kwargs=model_kwargs, require_chat=model_type == 'hf_chat')
    details: list[dict] = []

    def make_record(sample, result=None, error=None):
        record = {
            key: value for key, value in sample.items()
            if key not in ("source_file", "source_line")
        }
        record['prompt_version'] = PROMPT_VERSION
        record['inference_messages'] = _build_messages(sample) if model_type == 'hf_chat' else None
        if error is None:
            prediction, input_tokens, output_tokens, latency = result
            record.update({
                "model_output": prediction,
                "reference_answer": sample["targets"],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "output_limit_reached": output_tokens >= (max_out_len or MAX_NEW_TOKENS[sample['type']]),
                "latency_seconds": latency,
                "latency_kind": "amortized_batch_time",
            })
            record.update(score_prediction(sample, prediction))
        else:
            record.update({
                "model_output": "",
                "reference_answer": sample["targets"],
                "score": 0.0,
                "passed": False,
                "inference_error": error,
                "error_type": "inference_error",
            })
        return record

    if batch_size < 1:
        raise ValueError('batch_size 必须大于 0')
    import gc
    import torch
    effective_batch = batch_size
    cursor = 0
    started = time.perf_counter()
    # Persist completed batches, including when the job is later interrupted.
    with (output_dir / 'custom_details.jsonl').open('w', encoding='utf-8') as stream:
        while cursor < len(valid):
            batch = valid[cursor:cursor + effective_batch]
            failure = None
            try:
                results = _generate_batch(model, tokenizer, batch,
                    max_seq_len=max_seq_len, max_out_len=max_out_len, model_type=model_type)
            except Exception as exc:
                failure = str(exc)
            if failure is not None:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if len(batch) > 1:
                    effective_batch = max(1, len(batch) // 2)
                    print(f'[WARN] 批次失败，缩小至 {effective_batch} 重试: {failure}', flush=True)
                    continue
                records = [make_record(batch[0], error=failure)]
            else:
                records = [make_record(sample, result=result) for sample, result in zip(batch, results)]
            for record in records:
                details.append(record)
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            cursor += len(batch)
            elapsed = time.perf_counter() - started
            speed = cursor / max(elapsed, 1e-9)
            print(f'[INFO] 自定义评测 {cursor}/{len(valid)} batch_size={effective_batch} '
                  f'samples/s={speed:.3f} ETA={(len(valid)-cursor)/speed:.0f}s', flush=True)

    summary = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "dataset_metadata": dataset_metadata,
        **_aggregate(details, invalid),
    }
    _write_json(output_dir / "custom_summary.json", summary)
    _write_jsonl(output_dir / "custom_details.jsonl", details)
    if summary['inference_error_count']:
        raise RuntimeError('部分或全部样本推理失败，详见 custom_details.jsonl；不将不完整评测标记为成功')
    return summary
