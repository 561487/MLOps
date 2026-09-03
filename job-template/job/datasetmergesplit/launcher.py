import argparse
import hashlib
import json
import math
import random
import re
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path


SUPPORTED_FORMATS = {"messages", "qa"}
ROLE_NAMES = {"system", "user", "assistant"}
WHITESPACE = re.compile(r"\s+")
PROGRESS_INTERVAL = 1000


class MergeSplitError(Exception):
    pass


def emit_progress(stage, **values):
    payload = {
        "status": "running",
        "component": "DatasetMergeSplit",
        "stage": stage,
    }
    payload.update(values)
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str), flush=True)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false, got %r" % value)


def parse_dataset_config(value):
    text = str(value or "").strip()
    if not text:
        raise MergeSplitError("--dataset_config cannot be empty")
    candidate = Path(text).expanduser()
    if not text.startswith("[") and candidate.is_file():
        with candidate.open("r", encoding="utf-8") as stream:
            config = json.load(stream)
    else:
        try:
            config = json.loads(text)
        except ValueError as error:
            raise MergeSplitError("invalid dataset_config JSON: %s" % error)
    if not isinstance(config, list) or not config:
        raise MergeSplitError("dataset_config must be a non-empty JSON array")
    result = []
    names = set()
    for index, item in enumerate(config, 1):
        if not isinstance(item, dict):
            raise MergeSplitError("dataset_config item %d must be an object" % index)
        path = str(item.get("path") or "").strip()
        name = str(item.get("name") or "").strip()
        if not path:
            raise MergeSplitError("dataset_config item %d requires path" % index)
        if not name:
            name = Path(path.rstrip("/\\")).stem
        if not name:
            raise MergeSplitError("dataset_config item %d requires name" % index)
        if name in names:
            raise MergeSplitError("duplicate dataset name: %s" % name)
        names.add(name)
        result.append({"name": name, "path": path})
    return result


def build_dataset_items(config):
    dataset_paths = str(config.dataset_paths or "").strip()
    legacy_paths = [str(config.dataset1 or "").strip(), str(config.dataset2 or "").strip()]
    legacy_paths = [path for path in legacy_paths if path]
    legacy_config = str(config.dataset_config or "").strip()
    configured_inputs = sum(bool(value) for value in (dataset_paths, legacy_paths, legacy_config))
    if configured_inputs > 1:
        raise MergeSplitError("use --dataset_paths or legacy dataset inputs, not both")
    if dataset_paths:
        paths = [path.strip() for path in dataset_paths.split(";") if path.strip()]
        if not paths:
            raise MergeSplitError("--dataset_paths must contain at least one dataset directory")
    else:
        paths = legacy_paths
    if legacy_config:
        return parse_dataset_config(legacy_config)
    if not paths:
        raise MergeSplitError("--dataset_paths must contain at least one dataset directory separated by ';'")
    result = []
    used_names = set()
    used_paths = set()
    for index, path in enumerate(paths, 1):
        normalized_path = str(Path(path).expanduser().resolve())
        if normalized_path in used_paths:
            raise MergeSplitError("duplicate dataset path at position %d: %s" % (index, path))
        used_paths.add(normalized_path)
        name = Path(path.rstrip("/\\")).stem or "dataset%d" % index
        if name in used_names:
            name = "%s-%d" % (name, index)
        used_names.add(name)
        result.append({"name": name, "path": path})
    return result


def ensure_within(path, parent, description):
    resolved = Path(path).expanduser().resolve()
    base = Path(parent).expanduser().resolve()
    if resolved != base and base not in resolved.parents:
        raise MergeSplitError("%s escapes dataset directory: %s" % (description, resolved))
    return resolved


def infer_format(data_file):
    with data_file.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError as error:
                raise MergeSplitError("cannot infer format from %s line %d: %s" % (data_file, line_number, error))
            if isinstance(value, dict) and isinstance(value.get("messages"), list):
                return "messages"
            if isinstance(value, dict) and "question" in value and "answer" in value:
                return "qa"
            raise MergeSplitError("cannot infer format from %s line %d" % (data_file, line_number))
    raise MergeSplitError("input data file is empty: %s" % data_file)


def resolve_dataset(item, require_manifest, requested_format, schema_version):
    configured = Path(item["path"]).expanduser().resolve()
    if not configured.exists():
        raise MergeSplitError("dataset path does not exist: %s" % configured)
    manifest = None
    manifest_path = None
    if configured.is_dir():
        base_dir = configured
        for manifest_name in ("dataset_manifest.json", "manifest.json"):
            candidate = base_dir / manifest_name
            if candidate.is_file():
                manifest_path = candidate
                break
    elif configured.name in {"dataset_manifest.json", "manifest.json"}:
        base_dir = configured.parent
        manifest_path = configured
    elif configured.suffix.lower() == ".jsonl" and not require_manifest:
        base_dir = configured.parent
    else:
        raise MergeSplitError("dataset path must be an output directory%s: %s" % (" or JSONL file" if not require_manifest else "", configured))

    if manifest_path:
        try:
            with manifest_path.open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)
        except Exception as error:
            raise MergeSplitError("cannot read manifest %s: %s" % (manifest_path, error))
        if not isinstance(manifest, dict):
            raise MergeSplitError("manifest must be a JSON object: %s" % manifest_path)
        data_name = str(manifest.get("output_file") or manifest.get("data_file") or "").strip()
        if not data_name:
            raise MergeSplitError("manifest requires output_file or data_file: %s" % manifest_path)
        data_file = ensure_within(base_dir / data_name, base_dir, "data_file")
        manifest_format = str(manifest.get("target_schema") or manifest.get("format_type") or "").strip().lower()
        manifest_schema = str(manifest.get("version") or manifest.get("schema_version") or "").strip()
        if not manifest_format:
            raise MergeSplitError("manifest requires target_schema or format_type: %s" % manifest_path)
        if not manifest_schema:
            raise MergeSplitError("manifest requires version or schema_version: %s" % manifest_path)
        if schema_version != "auto" and manifest_schema != schema_version:
            raise MergeSplitError("schema_version mismatch for %s: expected %s, got %s" % (item["name"], schema_version, manifest_schema))
        if requested_format != "auto" and manifest_format != requested_format:
            raise MergeSplitError("format_type mismatch for %s: expected %s, got %s" % (item["name"], requested_format, manifest_format))
        data_format = manifest_format
    else:
        if require_manifest:
            raise MergeSplitError("dataset_manifest.json or manifest.json not found under %s" % configured)
        manifest = {}
        data_file = configured if configured.is_file() else configured / "dataset.jsonl"
        data_format = requested_format if requested_format != "auto" else infer_format(data_file)
        manifest_schema = schema_version if schema_version != "auto" else "1.0"

    if data_format not in SUPPORTED_FORMATS:
        raise MergeSplitError("unsupported format_type for %s: %s" % (item["name"], data_format))
    if not data_file.is_file():
        raise MergeSplitError("data file does not exist: %s" % data_file)
    declared_source = str(manifest.get("source_dataset") or "").strip()
    declared_sample_count = manifest.get("valid_samples")
    if not isinstance(declared_sample_count, int):
        declared_sample_count = manifest.get("sample_count")
    return {
        "name": item["name"],
        "source_dataset": item["name"] or declared_source,
        "base_dir": str(base_dir),
        "manifest_path": str(manifest_path) if manifest_path else None,
        "manifest": manifest,
        "data_file": str(data_file),
        "format_type": data_format,
        "schema_version": manifest_schema,
        "declared_sample_count": declared_sample_count,
    }


def normalized_text(value, normalize_whitespace):
    text = str(value)
    return WHITESPACE.sub(" ", text).strip() if normalize_whitespace else text


def stable_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_record(record, data_format, normalize_whitespace):
    if not isinstance(record, dict):
        raise MergeSplitError("record must be a JSON object")
    if data_format == "qa":
        question = record.get("question")
        answer = record.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise MergeSplitError("QA record requires non-empty string question")
        if not isinstance(answer, str) or not answer.strip():
            raise MergeSplitError("QA record requires non-empty string answer")
        prompt_core = normalized_text(question, normalize_whitespace)
        answer_core = normalized_text(answer, normalize_whitespace)
        sample_core = {"question": prompt_core, "answer": answer_core}
    else:
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            raise MergeSplitError("messages record requires a non-empty messages array")
        normalized_messages = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise MergeSplitError("messages[%d] must be an object" % index)
            role = message.get("role")
            content = message.get("content")
            if role not in ROLE_NAMES:
                raise MergeSplitError("messages[%d] has invalid role: %r" % (index, role))
            if not isinstance(content, str) or not content.strip():
                raise MergeSplitError("messages[%d] requires non-empty string content" % index)
            normalized_messages.append({"role": role, "content": normalized_text(content, normalize_whitespace)})
        if not any(item["role"] == "user" for item in normalized_messages):
            raise MergeSplitError("messages record requires at least one user message")
        if normalized_messages[-1]["role"] != "assistant":
            raise MergeSplitError("messages record must end with assistant")
        prompt_core = normalized_messages[:-1]
        answer_core = normalized_messages[-1]["content"]
        sample_core = normalized_messages
    prompt_hash = digest(stable_json(prompt_core))
    answer_hash = digest(stable_json(answer_core))
    sample_hash = digest(stable_json(sample_core))
    return prompt_hash, answer_hash, sample_hash


def prepare_output_record(record, source, preserve_metadata, source_field):
    output = dict(record)
    if preserve_metadata:
        metadata = output.get("metadata")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise MergeSplitError("metadata must be an object when preserve_metadata=true")
        metadata = dict(metadata)
        metadata.setdefault(source_field, source)
        output["metadata"] = metadata
    else:
        output.pop("metadata", None)
    return output


def write_jsonl(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")


def handle_invalid(policy, rejected, source, line_number, reason, raw, counters, source_stats):
    counters["invalid_records"] += 1
    source_stats["invalid_records"] += 1
    if policy == "fail":
        raise MergeSplitError("%s line %d: %s" % (source, line_number, reason))
    if policy == "reject":
        counters["rejected_records"] += 1
        write_jsonl(rejected, {"source_dataset": source, "line": line_number, "reason": reason, "raw": raw[:2000]})


def create_database(path):
    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("""
        CREATE TABLE records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_hash TEXT NOT NULL,
            prompt_hash TEXT NOT NULL,
            answer_hash TEXT NOT NULL,
            random_key TEXT NOT NULL,
            source TEXT NOT NULL,
            record_json TEXT NOT NULL,
            conflict INTEGER NOT NULL DEFAULT 0,
            split TEXT
        )
    """)
    connection.execute("CREATE INDEX idx_records_sample ON records(sample_hash)")
    connection.execute("CREATE INDEX idx_records_prompt ON records(prompt_hash)")
    return connection


def validate_ratios(train_ratio, validation_ratio):
    if not 0 < train_ratio <= 1:
        raise MergeSplitError("--train_ratio must be in (0, 1]")
    if not 0 <= validation_ratio < 1:
        raise MergeSplitError("--validation_ratio must be in [0, 1)")
    if not math.isclose(train_ratio + validation_ratio, 1.0, abs_tol=1e-8):
        raise MergeSplitError("--train_ratio + --validation_ratio must equal 1")


def assign_splits(connection, total, validation_ratio, shuffle, seed):
    if total <= 0:
        raise MergeSplitError("no valid records remain after validation, deduplication and conflict handling")
    if validation_ratio == 0:
        connection.execute("UPDATE records SET split='train' WHERE conflict=0")
        connection.commit()
        return total, 0
    if total < 2:
        raise MergeSplitError("at least 2 valid records are required when validation_ratio > 0")
    target = int(total * validation_ratio + 0.5)
    target = max(1, min(total - 1, target))
    order = "MIN(random_key), MIN(id)" if shuffle else "MIN(id)"
    groups = connection.execute(
        "SELECT prompt_hash, COUNT(*) FROM records WHERE conflict=0 GROUP BY prompt_hash ORDER BY %s" % order
    ).fetchall()
    validation_count = 0
    assignments = []
    group_total = len(groups)
    for group_index, (prompt_hash, group_count) in enumerate(groups, 1):
        if validation_count < target and total - validation_count - group_count >= 1:
            split = "validation"
            validation_count += group_count
        else:
            split = "train"
        assignments.append((prompt_hash, split))
        if group_index % 5000 == 0 or group_index == group_total:
            emit_progress(
                "split_progress",
                processed_prompt_groups=group_index,
                total_prompt_groups=group_total,
                validation_records=validation_count,
            )
    if validation_count == 0:
        raise MergeSplitError("cannot create validation set without splitting identical prompts; add more independent samples or set validation_ratio=0")
    connection.execute("CREATE TEMP TABLE split_assignments (prompt_hash TEXT PRIMARY KEY, split TEXT NOT NULL)")
    connection.executemany("INSERT INTO split_assignments(prompt_hash, split) VALUES (?, ?)", assignments)
    connection.execute("""
        UPDATE records
        SET split=(SELECT split FROM split_assignments WHERE split_assignments.prompt_hash=records.prompt_hash)
        WHERE conflict=0
    """)
    connection.execute("CREATE INDEX idx_records_split ON records(split)")
    connection.commit()
    return total - validation_count, validation_count


def write_split(connection, output_path, split, shuffle):
    order = "random_key, id" if shuffle else "id"
    counts = {}
    with output_path.open("w", encoding="utf-8") as stream:
        for source, record_json in connection.execute(
            "SELECT source, record_json FROM records WHERE conflict=0 AND split=? ORDER BY %s" % order,
            (split,),
        ):
            stream.write(record_json + "\n")
            counts[source] = counts.get(source, 0) + 1
    return counts


def validate_output_path(output_dir, datasets):
    destination = Path(output_dir).expanduser().resolve()
    if str(destination) in {"/", ""} or len(destination.parts) < 3:
        raise MergeSplitError("unsafe output directory: %s" % destination)
    for dataset in datasets:
        base = Path(dataset["base_dir"]).resolve()
        if destination == base:
            raise MergeSplitError("output directory cannot equal an input dataset directory: %s" % destination)
    return destination


def process(config):
    validate_ratios(config.train_ratio, config.validation_ratio)
    dataset_items = build_dataset_items(config)
    datasets = [resolve_dataset(item, config.require_manifest, config.format_type, config.schema_version) for item in dataset_items]
    formats = {item["format_type"] for item in datasets}
    if len(formats) != 1:
        raise MergeSplitError("all input datasets must have the same format_type, got: %s" % sorted(formats))
    data_format = next(iter(formats))
    schema_versions = {item["schema_version"] for item in datasets}
    if len(schema_versions) != 1:
        raise MergeSplitError("all input datasets must have the same schema_version, got: %s" % sorted(schema_versions))
    output_schema_version = next(iter(schema_versions))
    emit_progress(
        "inputs_resolved",
        input_datasets=len(datasets),
        format_type=data_format,
        schema_version=output_schema_version,
        datasets=[item["source_dataset"] for item in datasets],
    )
    destination = validate_output_path(config.output_dir, datasets)
    if destination.exists() and not config.overwrite:
        raise MergeSplitError("output directory already exists; enable --overwrite to replace it: %s" % destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".%s-merge-split-" % destination.name, dir=str(destination.parent)))
    rejected_path = temporary / "rejected.jsonl"
    conflicts_path = temporary / "conflicts.jsonl"
    # Keep SQLite's random I/O on container-local storage. Only final datasets
    # and reports are written to the PVC.
    database_directory = Path(tempfile.mkdtemp(prefix="dataset-merge-split-db-"))
    database_path = database_directory / "merge-split.sqlite"
    counters = {key: 0 for key in [
        "input_datasets", "input_records", "invalid_records", "rejected_records",
        "duplicate_records", "conflict_events", "conflict_records", "valid_records",
        "train_records", "validation_records", "train_validation_overlap",
    ]}
    counters["input_datasets"] = len(datasets)
    source_statistics = {}
    warnings = []
    connection = None
    try:
        connection = create_database(database_path)
        with rejected_path.open("w", encoding="utf-8") as rejected, conflicts_path.open("w", encoding="utf-8") as conflicts:
            for dataset in datasets:
                source = dataset["source_dataset"]
                stats = {key: 0 for key in ["input_records", "valid_input_records", "invalid_records", "duplicate_records", "conflict_events", "train_records", "validation_records"]}
                source_statistics[source] = stats
                emit_progress("dataset_started", source_dataset=source, data_file=dataset["data_file"])
                with Path(dataset["data_file"]).open("r", encoding="utf-8") as stream:
                    for line_number, line in enumerate(stream, 1):
                        if not line.strip():
                            continue
                        counters["input_records"] += 1
                        stats["input_records"] += 1
                        try:
                            record = json.loads(line)
                            prompt_hash, answer_hash, sample_hash = validate_record(record, data_format, config.normalize_whitespace)
                            output_record = prepare_output_record(record, source, config.preserve_metadata, config.source_field)
                        except Exception as error:
                            handle_invalid(config.invalid_policy, rejected, source, line_number, str(error), line, counters, stats)
                            continue
                        stats["valid_input_records"] += 1
                        if config.cross_dataset_dedup and connection.execute("SELECT 1 FROM records WHERE sample_hash=? LIMIT 1", (sample_hash,)).fetchone():
                            counters["duplicate_records"] += 1
                            stats["duplicate_records"] += 1
                            continue
                        different = None
                        if config.conflict_detection:
                            different = connection.execute(
                                "SELECT id, source, answer_hash, record_json FROM records WHERE prompt_hash=? AND answer_hash<>? LIMIT 1",
                                (prompt_hash, answer_hash),
                            ).fetchone()
                        conflict = 0
                        if different:
                            counters["conflict_events"] += 1
                            stats["conflict_events"] += 1
                            write_jsonl(conflicts, {
                                "reason": "same prompt with different answers",
                                "prompt_hash": prompt_hash,
                                "sources": [different[1], source],
                                "records": [json.loads(different[3]), output_record],
                            })
                            if config.conflict_policy == "fail":
                                raise MergeSplitError("answer conflict detected for %s line %d" % (source, line_number))
                            if config.conflict_policy == "reject":
                                connection.execute("UPDATE records SET conflict=1 WHERE prompt_hash=?", (prompt_hash,))
                                conflict = 1
                        random_key = digest("%d\0%s" % (config.seed, prompt_hash))
                        connection.execute(
                            "INSERT INTO records(sample_hash,prompt_hash,answer_hash,random_key,source,record_json,conflict) VALUES(?,?,?,?,?,?,?)",
                            (sample_hash, prompt_hash, answer_hash, random_key, source, stable_json(output_record), conflict),
                        )
                        if stats["input_records"] % 5000 == 0:
                            connection.commit()
                        if stats["input_records"] % PROGRESS_INTERVAL == 0:
                            emit_progress(
                                "dataset_progress",
                                source_dataset=source,
                                dataset_records=stats["input_records"],
                                total_records=counters["input_records"],
                                valid_input_records=stats["valid_input_records"],
                                invalid_records=stats["invalid_records"],
                                duplicate_records=counters["duplicate_records"],
                                conflict_events=counters["conflict_events"],
                            )
                connection.commit()
                emit_progress(
                    "dataset_finished",
                    source_dataset=source,
                    input_records=stats["input_records"],
                    valid_input_records=stats["valid_input_records"],
                    invalid_records=stats["invalid_records"],
                    duplicate_records=stats["duplicate_records"],
                    conflict_events=stats["conflict_events"],
                )
                if stats["valid_input_records"] == 0:
                    message = "dataset %s has no valid input records" % source
                    if config.empty_dataset_policy == "fail":
                        raise MergeSplitError(message)
                    warnings.append(message)
                declared = dataset.get("declared_sample_count")
                if isinstance(declared, int) and declared != stats["input_records"]:
                    warnings.append("dataset %s manifest sample_count=%d but actual=%d" % (source, declared, stats["input_records"]))

        counters["conflict_records"] = connection.execute("SELECT COUNT(*) FROM records WHERE conflict=1").fetchone()[0]
        total = connection.execute("SELECT COUNT(*) FROM records WHERE conflict=0").fetchone()[0]
        counters["valid_records"] = total
        emit_progress(
            "index_finished",
            input_records=counters["input_records"],
            valid_records=total,
            duplicate_records=counters["duplicate_records"],
            conflict_events=counters["conflict_events"],
            conflict_records=counters["conflict_records"],
        )
        train_count, validation_count = assign_splits(connection, total, config.validation_ratio, config.shuffle_before_split, config.seed)
        emit_progress("split_assigned", train_records=train_count, validation_records=validation_count)
        emit_progress("output_writing", output_dir=str(destination))
        train_by_source = write_split(connection, temporary / "train.jsonl", "train", config.shuffle_before_split)
        validation_by_source = {}
        validation_file = None
        if config.validation_ratio > 0:
            validation_file = "validation.jsonl"
            validation_by_source = write_split(connection, temporary / validation_file, "validation", config.shuffle_before_split)
        counters["train_records"] = train_count
        counters["validation_records"] = validation_count
        overlap = connection.execute("""
            SELECT COUNT(*) FROM (
                SELECT prompt_hash FROM records WHERE conflict=0 GROUP BY prompt_hash HAVING COUNT(DISTINCT split) > 1
            )
        """).fetchone()[0]
        counters["train_validation_overlap"] = overlap
        if overlap:
            raise MergeSplitError("internal split validation failed: train/validation prompt overlap=%d" % overlap)
        for source, stats in source_statistics.items():
            stats["train_records"] = train_by_source.get(source, 0)
            stats["validation_records"] = validation_by_source.get(source, 0)

        output_manifest = {
            "schema_version": output_schema_version,
            "format_type": data_format,
            "dataset_type": "sft_train_validation",
            "train_file": "train.jsonl",
            "validation_file": validation_file,
            "total_samples": train_count + validation_count,
            "train_samples": train_count,
            "validation_samples": validation_count,
            "train_ratio": config.train_ratio,
            "validation_ratio": config.validation_ratio,
            "source_datasets": [item["source_dataset"] for item in datasets],
            "seed": config.seed,
            "created_by": "DatasetMergeSplit",
        }
        report = {
            "status": "success",
            "component": "DatasetMergeSplit",
            "format_type": data_format,
            "output_dir": str(destination),
            "inputs": datasets,
            "counters": counters,
            "source_statistics": source_statistics,
            "warnings": warnings,
            "parameters": {
                "schema_version": output_schema_version,
                "require_manifest": config.require_manifest,
                "cross_dataset_dedup": config.cross_dataset_dedup,
                "conflict_detection": config.conflict_detection,
                "conflict_policy": config.conflict_policy,
                "preserve_metadata": config.preserve_metadata,
                "source_field": config.source_field,
                "train_ratio": config.train_ratio,
                "validation_ratio": config.validation_ratio,
                "shuffle_before_split": config.shuffle_before_split,
                "seed": config.seed,
            },
        }
        with (temporary / "dataset_manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(output_manifest, stream, ensure_ascii=False, indent=2)
        with (temporary / "merge_split_report.json").open("w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        connection.close()
        connection = None
        database_path.unlink()
        database_directory.rmdir()
        if destination.exists():
            shutil.rmtree(str(destination))
        temporary.rename(destination)
        destination.chmod(0o755)
        for output_file in destination.iterdir():
            if output_file.is_file():
                output_file.chmod(0o644)
        emit_progress(
            "completed",
            output_dir=str(destination),
            train_records=train_count,
            validation_records=validation_count,
            train_validation_overlap=overlap,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return report
    except Exception:
        if connection is not None:
            connection.close()
        shutil.rmtree(str(database_directory), ignore_errors=True)
        shutil.rmtree(str(temporary), ignore_errors=True)
        raise


def build_parser():
    parser = argparse.ArgumentParser(description="Merge same-format SFT datasets and split train/validation by ratio")
    parser.add_argument("--dataset_paths", default="", help="DatasetConvert output directories separated by ';'")
    parser.add_argument("--dataset1", default="", help=argparse.SUPPRESS)
    parser.add_argument("--dataset2", default="", help=argparse.SUPPRESS)
    parser.add_argument("--dataset_config", default="", help=argparse.SUPPRESS)
    parser.add_argument("--format_type", choices=["auto", "messages", "qa"], default="auto")
    parser.add_argument("--schema_version", default="auto", help=argparse.SUPPRESS)
    parser.add_argument("--require_manifest", type=parse_bool, default=True)
    parser.add_argument("--invalid_policy", choices=["reject", "skip", "fail"], default="reject")
    parser.add_argument("--empty_dataset_policy", choices=["skip", "fail"], default="fail")
    parser.add_argument("--cross_dataset_dedup", type=parse_bool, default=True)
    parser.add_argument("--conflict_detection", type=parse_bool, default=True)
    parser.add_argument("--conflict_policy", choices=["reject", "keep_all", "fail"], default="reject")
    parser.add_argument("--normalize_whitespace", type=parse_bool, default=True)
    parser.add_argument("--preserve_metadata", type=parse_bool, default=True)
    parser.add_argument("--source_field", default="source_dataset")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--validation_ratio", type=float, default=0.1)
    parser.add_argument("--shuffle_before_split", type=parse_bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--overwrite", type=parse_bool, default=False)
    return parser


def main(argv=None):
    try:
        process(build_parser().parse_args(argv))
        return 0
    except Exception as error:
        print("DatasetMergeSplit failed: %s" % error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
