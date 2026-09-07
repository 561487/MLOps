# -*- coding: utf-8 -*-
"""dataset-convert IO 层：文件发现、格式识别、记录读取、JSONL 写出。

复用自旧组件 job-template/job/datasetprocess/launcher.py 的保留能力：
discover_files / detect_format / validate_paths / iter_records / write_json_line / clean_text。
旧组件中的清洗、抽样、划分、过滤等职责不进入本组件。
"""

import csv
import fnmatch
import json
from pathlib import Path

SUPPORTED_FORMATS = {"json", "jsonl", "csv", "tsv", "parquet", "txt"}
EXTENSION_FORMATS = {
    ".json": "json",
    ".jsonl": "jsonl",
    ".csv": "csv",
    ".tsv": "tsv",
    ".parquet": "parquet",
    ".txt": "txt",
}
PARQUET_BATCH_SIZE = 5000


class DatasetIOError(Exception):
    """IO 层配置/路径级错误（非单条记录错误）。"""


def validate_paths(input_path, output_dir):
    """校验输入输出路径安全且不冲突。"""
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if source == destination:
        raise DatasetIOError("output directory cannot equal input path")
    if source.is_file() and destination == source.parent:
        raise DatasetIOError("output directory cannot be the input file parent directory")
    if str(destination) in {"/", ""} or len(destination.parts) < 3:
        raise DatasetIOError("unsafe output directory: %s" % destination)
    return source, destination


def detect_format(path, requested_format):
    """按文件扩展名识别输入格式；requested_format 非 auto 时直接返回。"""
    if requested_format != "auto":
        return requested_format
    detected = EXTENSION_FORMATS.get(path.suffix.lower())
    if not detected:
        raise DatasetIOError("cannot detect input format for %s (支持: %s)" % (path, ",".join(sorted(EXTENSION_FORMATS))))
    return detected


def discover_files(input_path, requested_format, file_pattern, recursive, output_dir=None):
    """发现输入文件：单文件直接返回；目录按 file_pattern/递归扫描。

    requested_format=auto 时仅保留已知扩展名的文件；输出目录自动排除。
    """
    source = Path(input_path).expanduser().resolve()
    if not source.exists():
        raise DatasetIOError("input path does not exist: %s" % source)
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
        raise DatasetIOError("no supported data files found under %s" % source)
    return sorted(files)


def iter_records(path, data_format, encoding="utf-8"):
    """逐条产出 (record, metadata, read_error) 三元组。

    - JSONL：逐行读取；
    - CSV/TSV：DictReader 流式迭代；
    - Parquet：iter_batches 分批迭代；
    - TXT：每个非空行作为一条 {"text": 行内容} 记录；
    - JSON：整体 json.load。
    """
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
        # 大文件风险：json.load 会一次性载入内存。超大 JSON 建议转成 JSONL 再处理；
        # 保留整体读取行为，与旧组件一致，后续可增加流式 JSON 解析。
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
            for batch in parquet.ParquetFile(str(path)).iter_batches(batch_size=PARQUET_BATCH_SIZE):
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
    raise DatasetIOError("unsupported input format: %s" % data_format)


def write_json_line(stream, value):
    """JSONL 写出；ensure_ascii=False 保证中文原样输出。"""
    stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
