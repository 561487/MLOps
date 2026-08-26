import argparse
import csv
import fnmatch
import json
import math
import random
import re
import shutil
import sys
import tempfile
from pathlib import Path


SUPPORTED_FORMATS = {"json", "jsonl", "csv", "tsv", "parquet", "txt"}
EXTENSION_FORMATS = {".json": "json", ".jsonl": "jsonl", ".csv": "csv", ".tsv": "tsv", ".parquet": "parquet", ".txt": "txt"}
ROLE_ALIASES = {"human": "user", "user": "user", "question": "user", "gpt": "assistant", "bot": "assistant", "assistant": "assistant", "answer": "assistant", "system": "system"}
CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
TEMPLATE_FIELD = re.compile(r"{{\s*([^{}]+?)\s*}}")


class ProcessingError(Exception):
    pass


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false, got %r" % value)


def parse_fields(value):
    if not value:
        return []
    text = str(value).strip()
    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ProcessingError("field configuration must be a JSON array or comma-separated text")
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in text.split(",") if item.strip()]


def clean_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    elif isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)
    return CONTROL_CHARACTERS.sub("", text).strip()


def get_value(record, field, default=None):
    if isinstance(record, dict) and field in record:
        return record[field]
    current = record
    for part in field.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


def render_template(template, record):
    return clean_text(TEMPLATE_FIELD.sub(lambda match: clean_text(get_value(record, match.group(1).strip(), "")), template or ""))


def join_fields(record, fields):
    parts = []
    for field in fields:
        value = clean_text(get_value(record, field, ""))
        if value:
            parts.append("%s：%s" % (field, value))
    return "\n".join(parts)


def select_record_fields(record, selected_fields, dropped_fields):
    selected = ({field: get_value(record, field) for field in selected_fields if get_value(record, field) is not None} if selected_fields else dict(record))
    for field in dropped_fields:
        selected.pop(field, None)
    return selected


def parse_literal(value):
    text = value.strip()
    lowered = text.lower()
    if lowered in {"null", "none"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return text


def compare_values(actual, operator, expected):
    normalized_actual = parse_literal(actual) if isinstance(actual, str) else actual
    if operator == "==":
        return normalized_actual == expected or clean_text(actual).lower() == clean_text(expected).lower()
    if operator == "!=":
        return not compare_values(actual, "==", expected)
    if operator == "contains":
        return expected in actual if isinstance(actual, (list, tuple, set, dict)) else clean_text(expected) in clean_text(actual)
    if operator == "not_contains":
        return not compare_values(actual, "contains", expected)
    try:
        left, right = float(actual), float(expected)
    except (TypeError, ValueError):
        left, right = clean_text(actual), clean_text(expected)
    return {">": left > right, ">=": left >= right, "<": left < right, "<=": left <= right}[operator]


def evaluate_filter(record, expression):
    if not expression or not expression.strip():
        return True
    clauses = re.split(r"\s*(?:&&|\band\b)\s*", expression.strip(), flags=re.IGNORECASE)
    for clause in clauses:
        is_match = re.fullmatch(r"(.+?)\s+is\s+(not\s+)?null", clause, flags=re.IGNORECASE)
        if is_match:
            actual = get_value(record, is_match.group(1).strip())
            result = actual is not None and clean_text(actual) != "" if is_match.group(2) else actual is None or clean_text(actual) == ""
            if not result:
                return False
            continue
        match = re.fullmatch(r"(.+?)\s*(not_contains|contains|==|!=|>=|<=|>|<)\s*(.+)", clause)
        if not match:
            raise ProcessingError("invalid filter clause: %s" % clause)
        field, operator, raw_expected = match.groups()
        if not compare_values(get_value(record, field.strip()), operator, parse_literal(raw_expected)):
            return False
    return True


def detect_format(path, requested_format):
    if requested_format != "auto":
        return requested_format
    detected = EXTENSION_FORMATS.get(path.suffix.lower())
    if not detected:
        raise ProcessingError("cannot detect input format for %s" % path)
    return detected


def discover_files(input_path, requested_format, file_pattern, recursive, output_dir=None):
    source = Path(input_path).expanduser().resolve()
    if not source.exists():
        raise ProcessingError("input path does not exist: %s" % source)
    if source.is_file():
        return [source]
    excluded = Path(output_dir).expanduser().resolve() if output_dir else None
    iterator = source.rglob("*") if recursive else source.glob("*")
    files = []
    for path in iterator:
        if not path.is_file() or not fnmatch.fnmatch(path.name, file_pattern or "*"):
            continue
        resolved = path.resolve()
        if excluded and (resolved == excluded or excluded in resolved.parents):
            continue
        if requested_format == "auto" and path.suffix.lower() not in EXTENSION_FORMATS:
            continue
        files.append(resolved)
    if not files:
        raise ProcessingError("no supported data files found under %s" % source)
    return sorted(files)


def iter_records(path, data_format, encoding="utf-8", batch_size=5000):
    if data_format == "jsonl":
        with path.open("r", encoding=encoding) as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    yield value if isinstance(value, dict) else {"text": value}, {"file": str(path), "line": line_number}, None
                except Exception as error:
                    yield None, {"file": str(path), "line": line_number, "raw": line[:1000]}, str(error)
        return
    if data_format == "json":
        try:
            with path.open("r", encoding=encoding) as stream:
                value = json.load(stream)
        except Exception as error:
            yield None, {"file": str(path)}, str(error)
            return
        if isinstance(value, dict) and isinstance(value.get("data"), list):
            value = value["data"]
        for index, item in enumerate(value if isinstance(value, list) else [value], 1):
            yield item if isinstance(item, dict) else {"text": item}, {"file": str(path), "index": index}, None
        return
    if data_format in {"csv", "tsv"}:
        with path.open("r", encoding=encoding, newline="") as stream:
            try:
                for line_number, item in enumerate(csv.DictReader(stream, delimiter="\t" if data_format == "tsv" else ","), 2):
                    yield dict(item), {"file": str(path), "line": line_number}, None
            except Exception as error:
                yield None, {"file": str(path)}, str(error)
        return
    if data_format == "parquet":
        try:
            import pyarrow.parquet as parquet
            index = 0
            for batch in parquet.ParquetFile(str(path)).iter_batches(batch_size=batch_size):
                for item in batch.to_pylist():
                    index += 1
                    yield item, {"file": str(path), "index": index}, None
        except Exception as error:
            yield None, {"file": str(path)}, str(error)
        return
    if data_format == "txt":
        with path.open("r", encoding=encoding) as stream:
            for line_number, line in enumerate(stream, 1):
                if line.strip():
                    yield {"text": line.rstrip("\r\n")}, {"file": str(path), "line": line_number}, None
        return
    raise ProcessingError("unsupported input format: %s" % data_format)


def first_present(record, fields):
    for field in fields:
        value = get_value(record, field)
        if value is not None and clean_text(value):
            return clean_text(value)
    return ""


def normalize_messages(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return []
    if not isinstance(value, list):
        return []
    messages = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = ROLE_ALIASES.get(clean_text(item.get("role") or item.get("from")).lower())
        content = clean_text(item.get("content") if "content" in item else item.get("value", item.get("text")))
        if role and content:
            messages.append({"role": role, "content": content})
    return messages


def make_messages(record, config):
    system_prompt = clean_text(config.system_prompt)
    user_fields, assistant_fields = parse_fields(config.user_fields), parse_fields(config.assistant_fields)
    if config.template_type == "conversation":
        raw = get_value(record, "messages", get_value(record, "conversations", get_value(record, "conversation")))
        messages = normalize_messages(raw)
        if system_prompt and not any(item["role"] == "system" for item in messages):
            messages.insert(0, {"role": "system", "content": system_prompt})
        if not any(item["role"] == "user" for item in messages) or not any(item["role"] == "assistant" for item in messages):
            raise ProcessingError("conversation record requires at least one user and assistant message")
        return messages
    if config.user_template:
        user_text = render_template(config.user_template, record)
    elif config.template_type == "qa":
        user_text = join_fields(record, user_fields) if user_fields else first_present(record, ["question", "query", "prompt", "instruction"])
    elif config.template_type == "instruction":
        user_text = join_fields(record, user_fields) if user_fields else "\n".join(part for part in [first_present(record, ["instruction", "question", "query", "prompt"]), first_present(record, ["input", "context"])] if part)
    else:
        if not user_fields:
            raise ProcessingError("custom template requires --user_fields or --user_template")
        user_text = join_fields(record, user_fields)
    if config.assistant_template:
        assistant_text = render_template(config.assistant_template, record)
    elif assistant_fields:
        assistant_text = join_fields(record, assistant_fields)
    else:
        assistant_text = first_present(record, ["answer", "response", "output", "completion"])
    if not user_text or not assistant_text:
        raise ProcessingError("SFT record requires non-empty user and assistant content")
    messages = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
    messages.extend([{"role": "user", "content": clean_text(user_text)}, {"role": "assistant", "content": clean_text(assistant_text)}])
    return messages


def write_json_line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def validate_paths(input_path, output_dir):
    source, destination = Path(input_path).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    if source == destination:
        raise ProcessingError("output directory cannot equal input path")
    if source.is_file() and destination == source.parent:
        raise ProcessingError("output directory cannot be the input file parent directory")
    if str(destination) in {"/", ""} or len(destination.parts) < 3:
        raise ProcessingError("unsafe output directory: %s" % destination)
    return source, destination


def handle_invalid(policy, rejected_stream, metadata, reason, record, counters):
    counters["rejected"] += 1
    if policy == "fail":
        raise ProcessingError("%s: %s" % (metadata, reason))
    if policy == "reject":
        write_json_line(rejected_stream, {"source": metadata, "reason": reason, "record": record})


def process_dataset(config):
    if not 0 < config.sample_ratio <= 1:
        raise ProcessingError("--sample_ratio must be in (0, 1]")
    if config.max_samples < 0:
        raise ProcessingError("--max_samples must be >= 0")
    if not 0 <= config.train_ratio <= 1 or not 0 <= config.val_ratio <= 1 or not math.isclose(config.train_ratio + config.val_ratio, 1.0, abs_tol=1e-8):
        raise ProcessingError("--train_ratio + --val_ratio must equal 1")
    source, destination = validate_paths(config.input_path, config.output_dir)
    files = discover_files(source, config.input_format, config.file_pattern, config.recursive, destination)
    selected_fields, dropped_fields, required_fields = parse_fields(config.select_fields), parse_fields(config.drop_fields), parse_fields(config.required_fields)
    if destination.exists() and not config.overwrite:
        raise ProcessingError("output directory already exists; enable --overwrite to replace it: %s" % destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".%s-processing-" % destination.name, dir=str(destination.parent)))
    staging_path, rejected_path = temporary / ".valid-records.jsonl", temporary / "rejected.jsonl"
    counters = {key: 0 for key in ["input_records", "read_errors", "filtered", "empty", "duplicate", "length_filtered", "rejected", "valid_before_sampling", "sampled_out", "output_records", "train_records", "val_records"]}
    seen = set()
    try:
        with staging_path.open("w", encoding="utf-8") as staging, rejected_path.open("w", encoding="utf-8") as rejected:
            for path in files:
                for record, metadata, read_error in iter_records(path, detect_format(path, config.input_format), config.encoding, config.batch_size):
                    counters["input_records"] += 1
                    if read_error:
                        counters["read_errors"] += 1
                        handle_invalid(config.invalid_policy, rejected, metadata, "read error: %s" % read_error, record, counters)
                        continue
                    try:
                        if not evaluate_filter(record, config.filter_expression):
                            counters["filtered"] += 1
                            continue
                        if required_fields and any(not clean_text(get_value(record, field)) for field in required_fields):
                            counters["empty"] += 1
                            handle_invalid(config.invalid_policy, rejected, metadata, "required field is empty", record, counters)
                            continue
                        messages = make_messages(select_record_fields(record, selected_fields, dropped_fields), config)
                        content_length = sum(len(item["content"]) for item in messages if item["role"] in {"user", "assistant"})
                        if config.drop_empty and content_length == 0:
                            counters["empty"] += 1
                            continue
                        if content_length < config.min_text_length or (config.max_text_length > 0 and content_length > config.max_text_length):
                            counters["length_filtered"] += 1
                            continue
                        output_record = {"messages": messages}
                        dedup_key = json.dumps(output_record, ensure_ascii=False, sort_keys=True)
                        if config.deduplicate and dedup_key in seen:
                            counters["duplicate"] += 1
                            continue
                        seen.add(dedup_key)
                        write_json_line(staging, output_record)
                        counters["valid_before_sampling"] += 1
                    except Exception as error:
                        if config.invalid_policy == "fail":
                            raise
                        handle_invalid(config.invalid_policy, rejected, metadata, str(error), record, counters)
        valid_count = counters["valid_before_sampling"]
        requested_count = max(1, int(math.floor(valid_count * config.sample_ratio))) if valid_count else 0
        if config.max_samples > 0:
            requested_count = min(requested_count, config.max_samples)
        requested_count = min(requested_count, valid_count)
        counters["sampled_out"] = valid_count - requested_count
        selected_indexes = set(range(requested_count)) if config.sampling_method == "head" else (set(random.Random(config.seed).sample(range(valid_count), requested_count)) if requested_count else set())
        selected_count = len(selected_indexes)
        val_count = min(int(round(selected_count * config.val_ratio)), selected_count)
        validation_positions = (set(random.Random(config.seed + 1).sample(range(selected_count), val_count)) if config.shuffle and val_count else set(range(selected_count - val_count, selected_count)))
        with staging_path.open("r", encoding="utf-8") as staging, (temporary / "train.jsonl").open("w", encoding="utf-8") as train, (temporary / "val.jsonl").open("w", encoding="utf-8") as validation:
            selected_position = 0
            for index, line in enumerate(staging):
                if index not in selected_indexes:
                    continue
                target = validation if selected_position in validation_positions else train
                target.write(line)
                counter = "val_records" if target is validation else "train_records"
                counters[counter] += 1
                counters["output_records"] += 1
                selected_position += 1
        staging_path.unlink()
        report = {"status": "success", "input_path": str(source), "output_dir": str(destination), "input_files": [str(path) for path in files], "template_type": config.template_type, "counters": counters, "parameters": {"input_format": config.input_format, "file_pattern": config.file_pattern, "select_fields": selected_fields, "drop_fields": dropped_fields, "required_fields": required_fields, "filter_expression": config.filter_expression, "sample_ratio": config.sample_ratio, "max_samples": config.max_samples, "sampling_method": config.sampling_method, "deduplicate": config.deduplicate, "train_ratio": config.train_ratio, "val_ratio": config.val_ratio, "seed": config.seed}}
        with (temporary / "processing_report.json").open("w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        if destination.exists():
            shutil.rmtree(str(destination))
        temporary.rename(destination)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report
    except Exception:
        shutil.rmtree(str(temporary), ignore_errors=True)
        raise


def build_parser():
    parser = argparse.ArgumentParser(description="Convert PVC datasets to ms-swift messages JSONL")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--input_format", default="auto", choices=["auto"] + sorted(SUPPORTED_FORMATS))
    parser.add_argument("--file_pattern", default="*")
    parser.add_argument("--recursive", type=parse_bool, default=True)
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--select_fields", default="")
    parser.add_argument("--filter_expression", default="")
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--sampling_method", choices=["random", "head"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop_empty", type=parse_bool, default=True)
    parser.add_argument("--required_fields", default="")
    parser.add_argument("--deduplicate", type=parse_bool, default=True)
    parser.add_argument("--min_text_length", type=int, default=1)
    parser.add_argument("--max_text_length", type=int, default=20000)
    parser.add_argument("--drop_fields", default="")
    parser.add_argument("--invalid_policy", choices=["reject", "skip", "fail"], default="reject")
    parser.add_argument("--template_type", choices=["qa", "instruction", "conversation", "custom"], default="custom")
    parser.add_argument("--system_prompt", default="")
    parser.add_argument("--user_fields", default="")
    parser.add_argument("--assistant_fields", default="")
    parser.add_argument("--user_template", default="")
    parser.add_argument("--assistant_template", default="")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--shuffle", type=parse_bool, default=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--overwrite", type=parse_bool, default=False)
    parser.add_argument("--batch_size", type=int, default=5000)
    return parser


def main(argv=None):
    try:
        process_dataset(build_parser().parse_args(argv))
        return 0
    except Exception as error:
        print("DatasetProcess failed: %s" % error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
