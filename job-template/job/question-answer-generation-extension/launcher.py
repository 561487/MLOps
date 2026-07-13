"""Generate and generalize evidence-grounded QA pairs from Markdown/text."""

import argparse
import json
import os
import re
from pathlib import Path

from llm_client import OpenAIChatClient
from prompts import SYSTEM_PROMPT, build_user_prompt


TEXT_SUFFIXES = {".md", ".txt"}
BOUNDARIES = ("\n\n", "\n", "。", "！", "？", ". ", "! ", "? ")


class BatchProcessingError(RuntimeError):
    def __init__(self, report):
        super().__init__("所有可处理文本块均失败，未覆盖已有问答数据")
        self.report = report


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
            window = text[start:limit]
            best = max(
                (
                    window.rfind(mark) + len(mark)
                    for mark in BOUNDARIES
                    if window.rfind(mark) >= 0
                ),
                default=-1,
            )
            if best > 0:
                end = start + best
        chunks.append(text[start:end])
        start = end
    return chunks


def validate_paths(input_dir, output_dir):
    source = Path(input_dir).resolve()
    target = Path(output_dir).resolve()
    if not source.is_dir():
        raise ValueError("输入目录不存在或不是目录")
    if target == source or source in target.parents:
        raise ValueError("输出目录不能位于输入目录内部")


def normalize_question(question):
    return re.sub(r"[\s？?。！!，,；;：:]", "", question).lower()


def validate_qa_payload(payload, source, limit, extension_limit):
    if not isinstance(payload, dict):
        raise ValueError("模型结果必须是 JSON 对象")
    pairs = payload.get("qa_pairs")
    if not isinstance(pairs, list):
        raise ValueError("qa_pairs 必须是数组")

    accepted = []
    seen = set()
    for item in pairs[:limit]:
        if not isinstance(item, dict):
            continue
        question = item.get("question")
        answer = item.get("answer")
        evidence = item.get("evidence")
        extensions = item.get("generalized_questions")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (question, answer, evidence)
        ):
            continue
        if (
            evidence not in source
            or answer.strip() not in evidence
            or not isinstance(extensions, list)
        ):
            continue

        key = normalize_question(question)
        if not key or key in seen:
            continue
        seen.add(key)
        clean_extensions = []
        extension_keys = {key}
        if extension_limit > 0:
            for extension in extensions:
                if not isinstance(extension, str) or not extension.strip():
                    continue
                extension_key = normalize_question(extension)
                if not extension_key or extension_key in extension_keys:
                    continue
                extension_keys.add(extension_key)
                clean_extensions.append(extension.strip())
                if len(clean_extensions) == extension_limit:
                    break

        accepted.append(
            {
                "question": question.strip(),
                "answer": answer.strip(),
                "generalized_questions": clean_extensions,
                "evidence": evidence,
            }
        )
    return accepted


def _failure(source_file, error, chunk_index=None):
    failure = {
        "source_file": source_file,
        "error_type": type(error).__name__,
    }
    if chunk_index is not None:
        failure["chunk_index"] = chunk_index
    return failure


def _atomic_write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_jsonl(path, records):
    content = "".join(
        json.dumps(record, ensure_ascii=False) + "\n"
        for record in records
    )
    _atomic_write_text(Path(path), content)


def _atomic_write_report(path, report):
    _atomic_write_text(
        Path(path),
        json.dumps(report, ensure_ascii=False, indent=2),
    )


def process_directory(
    input_dir,
    output_dir,
    client,
    chunk_size,
    min_chunk_chars,
    questions_per_chunk,
    extensions_per_question,
):
    validate_paths(input_dir, output_dir)
    source = Path(input_dir).resolve()
    target = Path(output_dir).resolve()
    files = find_text_files(source)
    target.mkdir(parents=True, exist_ok=True)
    records = []
    seen = set()
    report = {
        "input_files": len(files),
        "chunks": 0,
        "skipped_chunks": 0,
        "failed_chunks": 0,
        "successful_chunks": 0,
        "qa_pairs": 0,
        "failures": [],
    }

    for input_path in files:
        relative_path = input_path.relative_to(source).as_posix()
        try:
            text = input_path.read_text(encoding="utf-8-sig")
        except Exception as error:
            report["failed_chunks"] += 1
            report["failures"].append(_failure(relative_path, error))
            continue

        if not text.strip():
            report["skipped_chunks"] += 1
            continue

        for chunk_index, chunk in enumerate(split_text(text, chunk_size)):
            report["chunks"] += 1
            if len(chunk.strip()) < min_chunk_chars:
                report["skipped_chunks"] += 1
                continue
            try:
                pairs = client.complete_json(
                    SYSTEM_PROMPT,
                    build_user_prompt(
                        chunk,
                        questions_per_chunk,
                        extensions_per_question,
                    ),
                    lambda payload: validate_qa_payload(
                        payload,
                        chunk,
                        questions_per_chunk,
                        extensions_per_question,
                    ),
                )
                report["successful_chunks"] += 1
                for pair_item in pairs:
                    dedupe_key = (
                        relative_path,
                        normalize_question(pair_item["question"]),
                    )
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    records.append(
                        {
                            "source_file": relative_path,
                            "chunk_index": chunk_index,
                            "question": pair_item["question"],
                            "answer": pair_item["answer"],
                            "generalized_questions": pair_item[
                                "generalized_questions"
                            ],
                            "evidence": pair_item["evidence"],
                        }
                    )
            except Exception as error:
                report["failed_chunks"] += 1
                report["failures"].append(
                    _failure(relative_path, error, chunk_index)
                )

    report["qa_pairs"] = len(records)
    _atomic_write_report(target / "processing_report.json", report)
    partial_path = target / "qa_pairs.partial.jsonl"
    if report["failed_chunks"]:
        atomic_write_jsonl(partial_path, records)
        raise BatchProcessingError(report)
    if partial_path.exists():
        partial_path.unlink()
    atomic_write_jsonl(target / "qa_pairs.jsonl", records)
    return report


def _build_parser():
    parser = argparse.ArgumentParser("question-answer-generation-extension")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--api_key", default="")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--chunk_size", type=int, default=6000)
    parser.add_argument("--min_chunk_chars", type=int, default=80)
    parser.add_argument("--questions_per_chunk", type=int, default=5)
    parser.add_argument("--extensions_per_question", type=int, default=3)
    return parser


def _validate_runtime_args(args):
    if not 500 <= args.chunk_size <= 30000:
        raise ValueError("chunk_size 必须在 500 到 30000 之间")
    if not 1 <= args.min_chunk_chars <= args.chunk_size:
        raise ValueError("min_chunk_chars 必须在 1 到 chunk_size 之间")
    if not 1 <= args.questions_per_chunk <= 20:
        raise ValueError("questions_per_chunk 必须在 1 到 20 之间")
    if not 0 <= args.extensions_per_question <= 10:
        raise ValueError("extensions_per_question 必须在 0 到 10 之间")
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
            args.min_chunk_chars,
            args.questions_per_chunk,
            args.extensions_per_question,
        )
        print(
            "处理完成: files={}, chunks={}, failed_chunks={}, qa_pairs={}".format(
                report["input_files"],
                report["chunks"],
                report["failed_chunks"],
                report["qa_pairs"],
            )
        )
        return 0
    except (BatchProcessingError, OSError, ValueError) as error:
        print("ERROR: {}".format(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
