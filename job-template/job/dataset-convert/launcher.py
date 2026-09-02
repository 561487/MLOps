# -*- coding: utf-8 -*-
"""dataset-convert 启动器：参数解析、流程编排、错误码、manifest 输出。

流程：
    Job Template → launcher → Reader(iter_records) → Schema Detection
    → Converter → Validator → Writer(JSONL) → Manifest

输出契约（output_dir/）：
    converted.jsonl        有效数据（目标 schema 标准 JSONL，UTF-8，ensure_ascii=False）
    dataset_manifest.json  处理清单（含自动识别出的 input_format / source_schema）
    rejected.jsonl         仅当存在被拒绝记录时创建（行为见 README）

错误码：
    0 成功；1 通用错误；2 输入路径/文件问题；3 源结构自动识别失败；
    4 不支持的转换组合；5 field_mapping 非法；6 输出目录冲突；
    7 有效样本数为 0（无论 invalid_policy 均非 0 退出）；8 invalid_policy=fail 触发。
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from converter import check_supported, convert, validate_field_mapping
from dataset_io import (
    DatasetIOError,
    detect_format,
    discover_files,
    iter_records,
    validate_paths,
    write_json_line,
)
from schema import DETECT_SAMPLE_LIMIT, SchemaError, check_multimodal_record, detect_dataset_schema

EXIT_OK = 0
EXIT_GENERIC = 1
EXIT_INPUT_PATH = 2
EXIT_SCHEMA_DETECT = 3
EXIT_UNSUPPORTED = 4
EXIT_MAPPING = 5
EXIT_OUTPUT_CONFLICT = 6
EXIT_ZERO_VALID = 7
EXIT_FAIL_POLICY = 8

TOOL = "dataset-convert"
VERSION = "v1"


class ConvertError(Exception):
    exit_code = EXIT_GENERIC


class InputPathError(ConvertError):
    exit_code = EXIT_INPUT_PATH


class SchemaDetectError(ConvertError):
    exit_code = EXIT_SCHEMA_DETECT


class UnsupportedConversionError(ConvertError):
    exit_code = EXIT_UNSUPPORTED


class MappingError(ConvertError):
    exit_code = EXIT_MAPPING


class OutputError(ConvertError):
    exit_code = EXIT_OUTPUT_CONFLICT


class ZeroValidError(ConvertError):
    exit_code = EXIT_ZERO_VALID


class PolicyFailError(ConvertError):
    exit_code = EXIT_FAIL_POLICY


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false, got %r" % value)


def build_parser():
    parser = argparse.ArgumentParser(description="Dataset Convert：识别并转换为平台标准 JSONL（messages / text / eval_qa）")
    # 基础参数
    parser.add_argument("--input_path", required=True, help="输入数据集文件或目录（PVC 路径）")
    parser.add_argument("--output_dir", required=True, help="输出目录（converted.jsonl / dataset_manifest.json / rejected.jsonl）")
    parser.add_argument("--input_format", default="auto", choices=["auto", "json", "jsonl", "csv", "tsv", "parquet", "txt"], help="输入文件格式，auto 按扩展名识别")
    parser.add_argument("--source_schema", default="auto", choices=["auto"] + ["text", "alpaca", "messages", "sharegpt", "qa", "prompt_response", "custom"], help="源数据结构，auto 自动识别（见 DETECTION_ORDER）")
    parser.add_argument("--target_schema", default="messages", choices=["messages", "text", "eval_qa"], help="目标数据结构（不自动猜测，由用户显式选择）")
    # 高级参数
    parser.add_argument("--field_mapping", default="", help="source_schema=custom 必填：JSON 对象，如 {\"query_text\":\"user\",\"reply_text\":\"assistant\"}")
    parser.add_argument("--system_prompt", default="", help="仅 target_schema=messages 生效；已存在 system 轮次时不重复注入")
    parser.add_argument("--keep_metadata", type=parse_bool, default=False, help="true 时将多余字段收进 metadata 对象输出")
    parser.add_argument("--invalid_policy", choices=["reject", "skip", "fail"], default="reject", help="reject 写 rejected.jsonl；skip 仅计数；fail 遇首条无效记录即失败")
    parser.add_argument("--recursive", type=parse_bool, default=True, help="目录输入时递归扫描")
    parser.add_argument("--file_pattern", default="*", help="目录输入时按文件名匹配，如 *.csv")
    parser.add_argument("--encoding", default="utf-8", help="文本类文件编码")
    parser.add_argument("--overwrite", type=parse_bool, default=False, help="允许覆盖已有输出目录")
    return parser


def handle_invalid(policy, rejected_stream, metadata, reason, message, record, counters, invalid_reasons):
    """按 invalid_policy 处理单条无效记录；fail 策略直接抛错终止。"""
    counters["invalid_samples"] += 1
    invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
    detail = "file=%s line/index=%s reason=%s %s" % (metadata.get("file", "?"), metadata.get("line", metadata.get("index", "?")), reason, message)
    if policy == "fail":
        raise PolicyFailError("[invalid_policy=fail] 首条无效记录: %s" % detail)
    if policy == "reject":
        write_json_line(rejected_stream, {"source_file": metadata.get("file"), "line": metadata.get("line", metadata.get("index")), "reason": reason, "record": record})
    # skip：不写 rejected，仅计入统计


def run_convert(config):
    # ---- 1. 配置级校验（转换矩阵 / field_mapping），失败即退出 ----
    if config.source_schema != "auto":
        try:
            check_supported(config.source_schema, config.target_schema)
        except SchemaError as error:
            raise UnsupportedConversionError(error.message)
    try:
        field_mapping = validate_field_mapping(config.field_mapping, config.source_schema, config.target_schema)
    except SchemaError as error:
        raise MappingError(error.message)

    # ---- 2. 路径校验与文件发现 ----
    source, destination = validate_paths(config.input_path, config.output_dir)
    if destination.exists() and not config.overwrite:
        raise OutputError("输出目录已存在，启用 --overwrite 可覆盖: %s" % destination)
    try:
        files = discover_files(source, config.input_format, config.file_pattern, config.recursive, destination)
    except DatasetIOError as error:
        raise InputPathError(str(error))

    # ---- 3. 输入格式识别（auto → 实际格式，写入 manifest 前先记录） ----
    try:
        detected_formats = sorted({detect_format(path, config.input_format) for path in files})
    except DatasetIOError as error:
        raise InputPathError(str(error))
    input_format = ",".join(detected_formats) if len(detected_formats) > 1 else detected_formats[0]
    print("[dataset-convert] 输入文件数=%d 输入格式=%s" % (len(files), input_format))

    # ---- 4. 源结构识别（auto → 实际 schema） ----
    source_schema = config.source_schema
    if source_schema == "auto":
        try:
            source_schema, warnings = detect_dataset_schema(files, config.input_format, config.encoding)
        except DatasetIOError as error:
            raise InputPathError(str(error))
        if not source_schema:
            raise SchemaDetectError("无法识别源数据结构（采样前 %d 条记录均未命中 messages/conversations/instruction+output/prompt+response/question+answer/query+response/text）" % DETECT_SAMPLE_LIMIT)
        for warning in warnings:
            print("[dataset-convert][WARN] %s" % warning)
        print("[dataset-convert] 自动识别源结构=%s（采样判定）" % source_schema)
        try:
            check_supported(source_schema, config.target_schema)
        except SchemaError as error:
            raise UnsupportedConversionError(error.message)
    print("[dataset-convert] source_schema=%s target_schema=%s invalid_policy=%s" % (source_schema, config.target_schema, config.invalid_policy))

    # ---- 5. 流式转换（staging 临时目录，成功后原子替换） ----
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".%s-staging-" % destination.name, dir=str(destination.parent)))
    counters = {"total_samples": 0, "valid_samples": 0, "invalid_samples": 0}
    invalid_reasons = {}
    try:
        with (temporary / "converted.jsonl").open("w", encoding="utf-8") as converted, (temporary / "rejected.jsonl").open("w", encoding="utf-8") as rejected:
            for path in files:
                data_format = detect_format(path, config.input_format)
                for record, metadata, read_error in iter_records(path, data_format, config.encoding):
                    counters["total_samples"] += 1
                    if read_error:
                        handle_invalid(config.invalid_policy, rejected, metadata, "parse_error", "读取失败: %s" % read_error, record, counters, invalid_reasons)
                        continue
                    try:
                        multimodal_reason = check_multimodal_record(record)
                        if multimodal_reason:
                            raise SchemaError(multimodal_reason, "检测到多模态字段（image/images/video/audio 等），V1 仅支持文本数据转换")
                        output, consumed = convert(record, source_schema, config.target_schema, field_mapping)
                        if config.target_schema == "messages" and config.system_prompt:
                            inject_system_prompt(output, config.system_prompt)
                        if config.keep_metadata:
                            extras = {key: value for key, value in record.items() if key not in consumed}
                            if extras:
                                output["metadata"] = extras
                        write_json_line(converted, output)
                        counters["valid_samples"] += 1
                    except SchemaError as error:
                        handle_invalid(config.invalid_policy, rejected, metadata, error.reason, error.message, record, counters, invalid_reasons)
        if counters["valid_samples"] == 0:
            raise ZeroValidError("有效样本数为 0（共 %d 条输入）。任务必须失败，不能输出空结果。请检查输入数据与 schema 配置" % counters["total_samples"])

        # ---- 6. 收尾：写 manifest、原子替换输出目录 ----
        report = {
            "tool": TOOL,
            "version": VERSION,
            "input_path": str(source),
            "input_files": [str(path) for path in files],
            "input_format": input_format,
            "source_schema": source_schema,
            "target_schema": config.target_schema,
            "total_samples": counters["total_samples"],
            "valid_samples": counters["valid_samples"],
            "invalid_samples": counters["invalid_samples"],
            "invalid_reasons": invalid_reasons,
            "output_file": "converted.jsonl",
            "parameters": {"system_prompt": config.system_prompt, "keep_metadata": config.keep_metadata},
            "invalid_policy": config.invalid_policy,
            "field_mapping": field_mapping or None,
            "exit_code": EXIT_OK,
        }
        with (temporary / "dataset_manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        if not counters["invalid_samples"]:
            (temporary / "rejected.jsonl").unlink()
        if destination.exists():
            shutil.rmtree(str(destination))
        temporary.rename(destination)
        print_summary(report, destination)
        return report
    except Exception:
        shutil.rmtree(str(temporary), ignore_errors=True)
        raise


def inject_system_prompt(output, system_prompt):
    """system_prompt 注入：仅 target=messages；已有 system 轮次时不重复注入。"""
    messages = output["messages"]
    if any(item.get("role") == "system" for item in messages):
        return
    messages.insert(0, {"role": "system", "content": system_prompt})


def print_summary(report, destination):
    print("Dataset Convert completed")
    print("Input samples: %d" % report["total_samples"])
    print("Valid samples: %d" % report["valid_samples"])
    print("Rejected samples: %d" % report["invalid_samples"])
    print("Source schema: %s" % report["source_schema"])
    print("Target schema: %s" % report["target_schema"])
    print("Output: %s" % (destination / "converted.jsonl"))


def main(argv=None):
    try:
        run_convert(build_parser().parse_args(argv))
        return EXIT_OK
    except ConvertError as error:
        print("Dataset Convert failed: %s" % error, file=sys.stderr)
        return error.exit_code
    except Exception as error:
        print("Dataset Convert failed: %s" % error, file=sys.stderr)
        return EXIT_GENERIC


if __name__ == "__main__":
    sys.exit(main())
