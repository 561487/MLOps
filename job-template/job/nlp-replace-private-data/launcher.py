"""Rule- and LLM-assisted privacy redaction for Markdown and text files."""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from llm_client import OpenAIChatClient
from prompts import SYSTEM_PROMPT, build_user_prompt


TEXT_SUFFIXES = {".md", ".txt"}
BOUNDARIES = ("\n\n", "\n", "。", "！", "？", ". ", "! ", "? ")
CATEGORY_LABELS = {
    "email": "邮箱",
    "cn_id": "身份证",
    "phone": "手机号",
    "bank_card": "银行卡",
    "person_name": "姓名",
    "detailed_address": "详细地址",
    "account_id": "账号",
    "vehicle_id": "车辆标识",
    "device_id": "设备标识",
    "other_private": "其他隐私",
}
SEMANTIC_CATEGORIES = {
    "person_name",
    "detailed_address",
    "account_id",
    "vehicle_id",
    "device_id",
    "other_private",
}
PLACEHOLDER_PATTERN = re.compile(
    r"\[(?:邮箱|身份证|手机号|银行卡|姓名|详细地址|账号|车辆标识|设备标识|其他隐私)_\d+\]"
)
PATTERNS = (
    (
        "email",
        re.compile(
            r"(?<![A-Za-z0-9_.-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z0-9_.-])"
        ),
    ),
    (
        "cn_id",
        re.compile(r"(?<![0-9A-Za-z])\d{17}[0-9Xx](?![0-9A-Za-z])"),
    ),
    (
        "phone",
        re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)"),
    ),
    (
        "bank_card",
        re.compile(r"(?<!\d)(?:\d[ -]?){15,18}\d(?!\d)"),
    ),
)


def find_text_files(input_dir):
    root = Path(input_dir).resolve()
    files = []
    for current, directories, filenames in os.walk(str(root), followlinks=False):
        directories[:] = sorted(
            name
            for name in directories
            if not (Path(current) / name).is_symlink()
        )
        for filename in sorted(filenames):
            path = Path(current) / filename
            if (
                not path.is_symlink()
                and path.suffix.lower() in TEXT_SUFFIXES
            ):
                files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def split_text(text, max_chars):
    chunks = []
    start = 0
    while start < len(text):
        limit = min(start + max_chars, len(text))
        end = limit
        if limit < len(text):
            best = -1
            window = text[start:limit]
            for boundary in BOUNDARIES:
                position = window.rfind(boundary)
                if position >= 0 and position + len(boundary) > best:
                    best = position + len(boundary)
            if best > 0:
                end = start + best
        chunks.append(text[start:end])
        start = end
    return chunks


def overlapping_chunks(text, max_chars, overlap=200):
    if not text:
        return []
    overlap = min(overlap, max_chars // 2)
    step = max_chars - overlap
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + max_chars])
        if start + max_chars >= len(text):
            break
        start += step
    return chunks


def validate_paths(input_dir, output_dir):
    source = Path(input_dir).resolve()
    target = Path(output_dir).resolve()
    if not source.is_dir():
        raise ValueError("输入目录不存在或不是目录")
    if target == source or source in target.parents:
        raise ValueError("输出目录不能位于输入目录内部")


def _luhn_valid(value):
    digits = [int(char) for char in value if char.isdigit()]
    if not 16 <= len(digits) <= 19:
        return False
    total = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


class PlaceholderRegistry:
    def __init__(self):
        self.mapping = {}
        self.next_index = defaultdict(int)

    def placeholder(self, category, original):
        key = (category, original)
        if key not in self.mapping:
            self.next_index[category] += 1
            self.mapping[key] = "[{}_{}]".format(
                CATEGORY_LABELS[category],
                self.next_index[category],
            )
        return self.mapping[key]


def redact_rules(text, registry):
    counts = Counter()
    result = text
    for category, pattern in PATTERNS:

        def replace(match):
            original = match.group(0)
            if category == "bank_card" and not _luhn_valid(original):
                return original
            counts[category] += 1
            return registry.placeholder(category, original)

        result = pattern.sub(replace, result)
    return result, dict(counts)


def _validate_entities_payload(payload):
    entities = payload.get("entities")
    if not isinstance(entities, list):
        raise ValueError("entities 必须是数组")
    return entities


def _identify_semantic_entities(chunk, client):
    entities = client.complete_json(
        SYSTEM_PROMPT,
        build_user_prompt(chunk),
        _validate_entities_payload,
    )
    accepted = []
    for item in entities:
        if not isinstance(item, dict):
            continue
        original = item.get("text")
        category = item.get("category")
        if (
            category in SEMANTIC_CATEGORIES
            and isinstance(original, str)
            and 0 < len(original) <= 200
            and original in chunk
            and not (original.startswith("[") and original.endswith("]"))
        ):
            accepted.append((original, category))
    return accepted


def apply_entity_replacements(text, entities, registry):
    candidates = []
    protected_spans = [
        (match.start(), match.end())
        for match in PLACEHOLDER_PATTERN.finditer(text)
    ]
    for original, category in set(entities):
        for match in re.finditer(re.escape(original), text):
            if any(
                match.start() < protected_end
                and match.end() > protected_start
                for protected_start, protected_end in protected_spans
            ):
                continue
            candidates.append(
                (match.start(), match.end(), original, category)
            )

    selected = []
    for candidate in sorted(
        candidates,
        key=lambda item: (-(item[1] - item[0]), item[0], item[2], item[3]),
    ):
        start, end, _, _ = candidate
        if any(start < chosen[1] and end > chosen[0] for chosen in selected):
            continue
        selected.append(candidate)

    result = text
    counts = Counter()
    placeholders = {
        candidate: registry.placeholder(candidate[3], candidate[2])
        for candidate in sorted(selected, key=lambda item: item[0])
    }
    for start, end, original, category in sorted(
        selected,
        key=lambda item: item[0],
        reverse=True,
    ):
        result = (
            result[:start]
            + placeholders[(start, end, original, category)]
            + result[end:]
        )
        counts[category] += 1
    return result, dict(counts)


def redact_semantic_chunk(chunk, registry, client):
    entities = _identify_semantic_entities(chunk, client)
    return apply_entity_replacements(chunk, entities, registry)


def atomic_write_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _failure(source_file, error, chunk_index=None):
    failure = {
        "source_file": source_file,
        "error_type": type(error).__name__,
    }
    if chunk_index is not None:
        failure["chunk_index"] = chunk_index
    return failure


def process_directory(input_dir, output_dir, client, chunk_size):
    validate_paths(input_dir, output_dir)
    source = Path(input_dir).resolve()
    target = Path(output_dir).resolve()
    files = find_text_files(source)
    target.mkdir(parents=True, exist_ok=True)
    replacements = Counter()
    report = {
        "input_files": len(files),
        "processed_files": 0,
        "failed_files": 0,
        "chunks": 0,
        "failed_chunks": 0,
        "replacements": {},
        "failures": [],
    }

    for input_path in files:
        relative_path = input_path.relative_to(source).as_posix()
        output_path = target / input_path.relative_to(source)
        try:
            text = input_path.read_text(encoding="utf-8-sig")
            registry = PlaceholderRegistry()
            rule_redacted, rule_counts = redact_rules(text, registry)
            replacements.update(rule_counts)
            semantic_entities = []
            for chunk_index, chunk in enumerate(
                overlapping_chunks(rule_redacted, chunk_size)
            ):
                report["chunks"] += 1
                if not chunk.strip():
                    continue
                try:
                    semantic_entities.extend(
                        _identify_semantic_entities(
                            chunk,
                            client,
                        )
                    )
                except Exception as error:
                    report["failed_chunks"] += 1
                    report["failures"].append(
                        _failure(relative_path, error, chunk_index)
                    )

            redacted, semantic_counts = apply_entity_replacements(
                rule_redacted,
                semantic_entities,
                registry,
            )
            replacements.update(semantic_counts)
            atomic_write_text(output_path, redacted)
            report["processed_files"] += 1
        except Exception as error:
            report["failed_files"] += 1
            report["failures"].append(_failure(relative_path, error))

    report["replacements"] = dict(sorted(replacements.items()))
    (target / "processing_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def _build_parser():
    parser = argparse.ArgumentParser("nlp-replace-private-data")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--api_key", default="")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--chunk_size", type=int, default=6000)
    return parser


def _validate_runtime_args(args):
    if not 500 <= args.chunk_size <= 30000:
        raise ValueError("chunk_size 必须在 500 到 30000 之间")
    if args.timeout <= 0:
        raise ValueError("timeout 必须大于 0")
    if args.max_retries < 0:
        raise ValueError("max_retries 不能小于 0")


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_runtime_args(args)
        validate_paths(args.input_dir, args.output_dir)
        if not find_text_files(args.input_dir):
            raise ValueError("输入目录中没有 .md 或 .txt 文件")
        client = OpenAIChatClient(
            args.base_url,
            args.api_key or os.getenv("API_SECRET_KEY", ""),
            args.model_name,
            args.temperature,
            args.timeout,
            args.max_retries,
        )
        report = process_directory(
            args.input_dir,
            args.output_dir,
            client,
            args.chunk_size,
        )
        print(
            "处理完成: files={}, failed_files={}, failed_chunks={}, replacements={}".format(
                report["processed_files"],
                report["failed_files"],
                report["failed_chunks"],
                sum(report["replacements"].values()),
            )
        )
        return 0
    except (OSError, ValueError) as error:
        print("ERROR: {}".format(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
