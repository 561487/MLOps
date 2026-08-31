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
CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


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
            raise ProcessingError("fields must be a JSON array or comma-separated text")
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_rename_fields(value):
    if not value or not str(value).strip():
        return {}
    text = str(value).strip()
    if text.startswith("{"):
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ProcessingError("rename_fields must be a JSON object or old:new pairs")
        result = {str(source).strip(): str(target).strip() for source, target in parsed.items()}
    else:
        result = {}
        for item in text.split(","):
            if not item.strip():
                continue
            if ":" not in item:
                raise ProcessingError("invalid rename pair: %s" % item)
            source, target = item.split(":", 1)
            result[source.strip()] = target.strip()
    if any(not source or not target for source, target in result.items()):
        raise ProcessingError("rename_fields cannot contain an empty source or target")
    if len(set(result.values())) != len(result):
        raise ProcessingError("rename_fields target names must be unique")
    return result


def text_value(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def clean_value(value, trim_whitespace=True):
    if isinstance(value, str):
        value = CONTROL_CHARACTERS.sub("", value)
        return value.strip() if trim_whitespace else value
    if isinstance(value, list):
        return [clean_value(item, trim_whitespace) for item in value]
    if isinstance(value, dict):
        return {key: clean_value(item, trim_whitespace) for key, item in value.items()}
    return value


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


def select_record_fields(record, selected_fields, dropped_fields, rename_fields):
    if selected_fields:
        selected = {field: get_value(record, field) for field in selected_fields if get_value(record, field) is not None}
    else:
        selected = dict(record)
    for field in dropped_fields:
        selected.pop(field, None)
    for source, target in rename_fields.items():
        if source in selected:
            selected[target] = selected.pop(source)
        elif not selected_fields:
            value = get_value(record, source)
            if value is not None:
                selected[target] = value
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
        return normalized_actual == expected or text_value(actual).lower() == text_value(expected).lower()
    if operator == "!=":
        return not compare_values(actual, "==", expected)
    if operator == "contains":
        return expected in actual if isinstance(actual, (list, tuple, set, dict)) else text_value(expected) in text_value(actual)
    if operator == "not_contains":
        return not compare_values(actual, "contains", expected)
    try:
        left, right = float(actual), float(expected)
    except (TypeError, ValueError):
        left, right = text_value(actual), text_value(expected)
    return {">": left > right, ">=": left >= right, "<": left < right, "<=": left <= right}[operator]


def evaluate_filter(record, expression):
    if not expression or not expression.strip():
        return True
    clauses = re.split(r"\s*(?:&&|\band\b)\s*", expression.strip(), flags=re.IGNORECASE)
    for clause in clauses:
        is_match = re.fullmatch(r"(.+?)\s+is\s+(not\s+)?null", clause, flags=re.IGNORECASE)
        if is_match:
            actual = get_value(record, is_match.group(1).strip())
            is_empty = actual is None or text_value(actual) == ""
            if (not is_empty if is_match.group(2) else is_empty) is False:
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
                reader = csv.DictReader(stream, delimiter="\t" if data_format == "tsv" else ",")
                for line_number, item in enumerate(reader, 2):
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


def collect_text(record, text_fields):
    if text_fields:
        return "\n".join(text_value(get_value(record, field)) for field in text_fields if text_value(get_value(record, field)))
    values = []

    def walk(value):
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(record)
    return "\n".join(values)


def has_content(record, text_fields):
    if text_fields:
        return bool(collect_text(record, text_fields))

    def present(value):
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, dict):
            return any(present(item) for item in value.values())
        if isinstance(value, list):
            return any(present(item) for item in value)
        return True

    return present(record)


def write_json_line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")


def validate_paths(input_path, output_dir):
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
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
    if config.min_text_length < 0 or config.max_text_length < 0:
        raise ProcessingError("text length limits must be >= 0")
    if config.max_text_length and config.max_text_length < config.min_text_length:
        raise ProcessingError("--max_text_length must be 0 or >= --min_text_length")
    source, destination = validate_paths(config.input_path, config.output_dir)
    files = discover_files(source, config.input_format, config.file_pattern, config.recursive, destination)
    selected_fields, dropped_fields = parse_fields(config.select_fields), parse_fields(config.drop_fields)
    required_fields, text_fields = parse_fields(config.required_fields), parse_fields(config.text_fields)
    dedup_fields, rename_fields = parse_fields(config.dedup_fields), parse_rename_fields(config.rename_fields)
    effective_text_fields = [rename_fields.get(field, field) for field in text_fields]
    if destination.exists() and not config.overwrite:
        raise ProcessingError("output directory already exists; enable --overwrite to replace it: %s" % destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".%s-processing-" % destination.name, dir=str(destination.parent)))
    staging_path, rejected_path = temporary / ".valid-records.jsonl", temporary / "rejected.jsonl"
    counters = {key: 0 for key in ["input_records", "read_errors", "filtered", "empty", "duplicate", "length_filtered", "rejected", "valid_before_sampling", "sampled_out", "output_records"]}
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
                        if required_fields and any(not text_value(get_value(record, field)) for field in required_fields):
                            counters["empty"] += 1
                            handle_invalid(config.invalid_policy, rejected, metadata, "required field is empty", record, counters)
                            continue
                        output_record = select_record_fields(record, selected_fields, dropped_fields, rename_fields)
                        if config.clean_text:
                            output_record = clean_value(output_record, config.trim_whitespace)
                        record_text = collect_text(output_record, effective_text_fields)
                        if config.drop_empty and not has_content(output_record, effective_text_fields):
                            counters["empty"] += 1
                            continue
                        length = len(record_text)
                        if length < config.min_text_length or (config.max_text_length and length > config.max_text_length):
                            counters["length_filtered"] += 1
                            continue
                        if config.deduplicate:
                            value = {field: get_value(output_record, field) for field in dedup_fields} if dedup_fields else output_record
                            key = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
                            if key in seen:
                                counters["duplicate"] += 1
                                continue
                            seen.add(key)
                        write_json_line(staging, output_record)
                        counters["valid_before_sampling"] += 1
                    except Exception as error:
                        if config.invalid_policy == "fail":
                            raise
                        handle_invalid(config.invalid_policy, rejected, metadata, str(error), record, counters)

        valid_count = counters["valid_before_sampling"]
        requested_count = max(1, int(math.floor(valid_count * config.sample_ratio))) if valid_count else 0
        if config.max_samples:
            requested_count = min(requested_count, config.max_samples)
        requested_count = min(requested_count, valid_count)
        counters["sampled_out"] = valid_count - requested_count
        selected_indexes = set(range(requested_count)) if config.sampling_method == "head" else (set(random.Random(config.seed).sample(range(valid_count), requested_count)) if requested_count else set())
        with staging_path.open("r", encoding="utf-8") as staging, (temporary / "dataset.jsonl").open("w", encoding="utf-8") as output:
            for index, line in enumerate(staging):
                if index in selected_indexes:
                    output.write(line)
                    counters["output_records"] += 1
        staging_path.unlink()
        report = {
            "status": "success", "component": "DatasetPreprocess", "input_path": str(source),
            "output_dir": str(destination), "output_file": str(destination / "dataset.jsonl"),
            "input_files": [str(path) for path in files], "counters": counters,
            "parameters": {"input_format": config.input_format, "file_pattern": config.file_pattern,
                "select_fields": selected_fields, "drop_fields": dropped_fields, "rename_fields": rename_fields,
                "required_fields": required_fields, "text_fields": text_fields, "filter_expression": config.filter_expression,
                "sample_ratio": config.sample_ratio, "max_samples": config.max_samples, "sampling_method": config.sampling_method,
                "clean_text": config.clean_text, "trim_whitespace": config.trim_whitespace,
                "deduplicate": config.deduplicate, "dedup_fields": dedup_fields, "seed": config.seed},
        }
        with (temporary / "preprocessing_report.json").open("w", encoding="utf-8") as stream:
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
    parser = argparse.ArgumentParser(description="Clean and normalize PVC datasets without SFT conversion or splitting")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--input_format", default="auto", choices=["auto"] + sorted(SUPPORTED_FORMATS))
    parser.add_argument("--file_pattern", default="*")
    parser.add_argument("--recursive", type=parse_bool, default=True)
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--select_fields", default="")
    parser.add_argument("--drop_fields", default="")
    parser.add_argument("--rename_fields", default="")
    parser.add_argument("--filter_expression", default="")
    parser.add_argument("--required_fields", default="")
    parser.add_argument("--text_fields", default="")
    parser.add_argument("--drop_empty", type=parse_bool, default=True)
    parser.add_argument("--clean_text", type=parse_bool, default=True)
    parser.add_argument("--trim_whitespace", type=parse_bool, default=True)
    parser.add_argument("--min_text_length", type=int, default=0)
    parser.add_argument("--max_text_length", type=int, default=0)
    parser.add_argument("--deduplicate", type=parse_bool, default=True)
    parser.add_argument("--dedup_fields", default="")
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--sampling_method", choices=["random", "head"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--invalid_policy", choices=["reject", "skip", "fail"], default="reject")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--overwrite", type=parse_bool, default=False)
    parser.add_argument("--batch_size", type=int, default=5000)
    return parser


def main(argv=None):
    try:
        process_dataset(build_parser().parse_args(argv))
        return 0
    except Exception as error:
        print("DatasetPreprocess failed: %s" % error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
