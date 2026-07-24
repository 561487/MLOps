#!/usr/bin/env python3
"""Build a Qwen3/ms-swift ServerOps SFT dataset from a public QA corpus.

The node is designed to run in a Kubernetes task Pod:

* source and final artifacts live on a mounted PVC;
* the temporary SQLite selection database lives on local ephemeral storage;
* source files are streamed and never loaded into memory as a whole;
* the source directory is never modified.
"""

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path


DEFAULT_SYSTEM_PROMPT = (
    "你是企业服务器、Linux、容器、Kubernetes和GPU集群运维助手。"
    "请基于用户提供的现象与证据回答；信息不足时先说明需要补充的信息。"
    "诊断建议应包含故障判断、关键证据、按优先级排列的排查步骤、"
    "可执行命令及预期结果、修复方案、风险与回滚，以及人工升级条件。"
    "不得伪造日志、命令结果或不存在的配置。"
)

SUPPORTED_SUFFIXES = {".jsonl", ".json", ".parquet"}
FIELD_ALIASES = {
    "instruction": ("instruction", "query", "prompt", "question", "input_text"),
    "input": ("input", "context"),
    "response": ("output", "response", "answer", "completion", "target"),
    "id": ("id", "uuid", "sample_id"),
}

CATEGORY_KEYWORDS = {
    "linux": {
        "linux": 2, "systemd": 3, "systemctl": 3, "journalctl": 3,
        "kernel": 2, "内核": 2, "进程": 2, "服务异常": 3, "权限": 1,
        "permission": 2, "sudo": 2, "ssh": 2, "cpu": 1, "内存": 2,
        "memory": 1, "oom": 3, "load average": 3, "文件描述符": 3,
    },
    "network": {
        "network": 2, "网络": 2, "dns": 3, "端口": 2, "port": 1,
        "firewall": 3, "防火墙": 3, "iptables": 3, "nftables": 3,
        "tcp": 2, "udp": 2, "路由": 2, "route": 1, "网卡": 2,
        "socket": 2, "connection timeout": 3, "连接超时": 3,
    },
    "container": {
        "docker": 3, "containerd": 3, "容器": 2, "container": 2,
        "imagepull": 3, "镜像拉取": 3, "dockerfile": 2, "cgroup": 2,
        "overlayfs": 3, "registry": 2, "harbor": 3,
    },
    "kubernetes": {
        "kubernetes": 3, "k8s": 3, "kubectl": 3, "pod": 2,
        "deployment": 2, "statefulset": 3, "daemonset": 3,
        "configmap": 2, "namespace": 1, "调度": 2, "scheduler": 2,
        "crashloopbackoff": 4, "imagepullbackoff": 4, "pending": 1,
        "pytorchjob": 4, "volcano": 4, "ingress": 2,
    },
    "gpu": {
        "gpu": 2, "nvidia": 3, "cuda": 3, "显存": 3,
        "nvidia-smi": 4, "driver": 1, "驱动": 2, "xid": 4,
        "nvml": 3, "cudnn": 3, "tensor core": 3,
    },
    "distributed": {
        "nccl": 5, "torchrun": 4, "deepspeed": 4, "zero-2": 4,
        "zero2": 4, "zero-3": 4, "zero3": 4, "ddp": 3,
        "distributed training": 4, "分布式训练": 4, "多机多卡": 5,
        "master_addr": 4, "world_size": 4, "rank": 1,
        "allreduce": 4, "rdma": 4,
    },
    "storage": {
        "磁盘": 2, "disk": 2, "filesystem": 2, "文件系统": 2,
        "inode": 3, "mount": 2, "挂载": 2, "nfs": 3, "ceph": 3,
        "juicefs": 4, "minio": 3, "pvc": 3, "persistentvolume": 4,
        "i/o error": 4, "io error": 3,
    },
    "observability": {
        "prometheus": 3, "grafana": 3, "监控": 2, "日志": 1,
        "log": 1, "告警": 2, "alert": 2, "metrics": 2,
        "dcgm": 4, "swanlab": 4, "trace": 1,
    },
    "security": {
        "ssh": 2, "权限": 2, "permission": 2, "证书": 2,
        "certificate": 2, "tls": 2, "firewall": 2, "防火墙": 2,
        "selinux": 3, "apparmor": 3, "secret": 1, "密钥": 2,
    },
}

CATEGORY_WEIGHTS = {
    "linux": 0.18,
    "network": 0.15,
    "container": 0.12,
    "kubernetes": 0.20,
    "gpu": 0.10,
    "distributed": 0.10,
    "storage": 0.07,
    "observability": 0.05,
    "security": 0.03,
}

OFFENSIVE_ONLY_TERMS = (
    "sql injection", "sql注入", "xss", "cross-site scripting",
    "webshell", "木马", "malware", "恶意软件", "phishing", "钓鱼",
    "password cracking", "密码爆破", "payload", "漏洞利用", "exploit",
    "metasploit", "burp suite",
)

DANGEROUS_PATTERNS = (
    r"\brm\s+-rf\b", r"\bmkfs(?:\.\w+)?\b", r"\bfdisk\b",
    r"\biptables\s+-F\b", r"\bkubectl\s+delete\b",
    r"\bdocker\s+system\s+prune\b", r"\breboot\b",
    r"\bshutdown\b", r"\bchmod\s+-R\s+777\b",
    r"\bnvidia-smi\s+--gpu-reset\b",
)

SAFETY_TERMS = (
    "风险", "回滚", "备份", "确认", "谨慎", "影响", "不可恢复",
    "risk", "rollback", "backup", "confirm", "caution", "impact",
)

STRUCTURE_TERMS = (
    "故障", "原因", "排查", "步骤", "命令", "修复", "建议",
    "预期", "风险", "回滚", "diagnos", "cause", "step",
    "command", "fix", "expected", "rollback",
)

SECRET_PATTERNS = (
    (re.compile(r"AKIA[0-9A-Z]{16}"), "<AWS_ACCESS_KEY>"),
    (re.compile(r"(?i)(access[_-]?key|secret[_-]?key|token|password)"
                r"\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{8,}"), r"\1=<REDACTED>"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?"
                r"-----END [A-Z ]*PRIVATE KEY-----", re.S), "<PRIVATE_KEY_REDACTED>"),
)


def bool_arg(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true/false")


def stable_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(value):
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False)
    value = value.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in value.splitlines()]
    value = "\n".join(lines)
    value = re.sub(r"\n{4,}", "\n\n\n", value)
    return value.strip()


def redact_secrets(text):
    for pattern, replacement in SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def first_present(record, aliases):
    for name in aliases:
        value = record.get(name)
        if value not in (None, "", []):
            return value
    return ""


def extract_pair(record):
    if not isinstance(record, dict):
        return "", "", ""

    instruction = first_present(record, FIELD_ALIASES["instruction"])
    context = first_present(record, FIELD_ALIASES["input"])
    response = first_present(record, FIELD_ALIASES["response"])
    source_id = first_present(record, FIELD_ALIASES["id"])

    # AutoPreprocessor-style messages input is also accepted.
    if (not instruction or not response) and isinstance(record.get("messages"), list):
        users = [
            normalize_text(item.get("content"))
            for item in record["messages"]
            if isinstance(item, dict) and item.get("role") == "user"
        ]
        assistants = [
            normalize_text(item.get("content"))
            for item in record["messages"]
            if isinstance(item, dict) and item.get("role") == "assistant"
        ]
        instruction = instruction or (users[-1] if users else "")
        response = response or (assistants[-1] if assistants else "")

    instruction = normalize_text(instruction)
    context = normalize_text(context)
    response = normalize_text(response)
    source_id = normalize_text(source_id)

    if context and context != instruction:
        instruction = instruction + "\n\n上下文：\n" + context
    return instruction, response, source_id


def iter_json_records(path):
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        first = ""
        while True:
            char = handle.read(1)
            if not char:
                return
            if not char.isspace():
                first = char
                break
        handle.seek(0)
        if first == "[":
            try:
                import ijson
            except ImportError as exc:
                raise RuntimeError("ijson is required for JSON array files") from exc
            yield from ijson.items(handle, "item")
            return

        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"WARNING invalid JSON: {path}:{line_number}", flush=True)


def iter_records(path, batch_size):
    if path.suffix.lower() in {".json", ".jsonl"}:
        yield from iter_json_records(path)
        return

    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("pyarrow is required for Parquet files") from exc
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size):
            yield from batch.to_pylist()


def find_data_files(raw_dir):
    result = []
    for path in Path(raw_dir).rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            result.append(path)
    return sorted(result)


def download_dataset(dataset_id, raw_dir, revision, force):
    raw_path = Path(raw_dir)
    existing = find_data_files(raw_path) if raw_path.exists() else []
    if existing and not force:
        print(f"raw data exists ({len(existing)} files), skip download", flush=True)
        return
    if force and raw_path.exists():
        raise RuntimeError(
            "force download does not delete existing raw data; use a new raw_dir"
        )

    raw_path.mkdir(parents=True, exist_ok=True)
    command = [
        "modelscope", "download", "--dataset", dataset_id,
        "--local_dir", str(raw_path),
    ]
    if revision:
        command.extend(["--revision", revision])
    print("download command:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    if not find_data_files(raw_path):
        raise RuntimeError(f"download completed but no supported files found in {raw_dir}")


def category_scores(text):
    lowered = text.lower()
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = 0
        for keyword, weight in keywords.items():
            if keyword.lower() in lowered:
                score += weight
        scores[category] = score
    return scores


def classify_domain(instruction, response):
    combined = instruction + "\n" + response[:6000]
    scores = category_scores(combined)
    category = max(scores, key=scores.get)
    return category, scores[category], scores


def has_dangerous_command(text):
    return any(re.search(pattern, text, flags=re.I) for pattern in DANGEROUS_PATTERNS)


def has_safety_context(text):
    lowered = text.lower()
    return any(term.lower() in lowered for term in SAFETY_TERMS)


def looks_offensive_only(text, domain_score):
    lowered = text.lower()
    offensive_hits = sum(term in lowered for term in OFFENSIVE_ONLY_TERMS)
    return offensive_hits >= 2 and domain_score < 6


def quality_score(instruction, response, domain_score):
    score = 0
    if domain_score >= 3:
        score += 2
    if domain_score >= 6:
        score += 1
    if 40 <= len(instruction) <= 3000:
        score += 1
    if 200 <= len(response) <= 16000:
        score += 2
    structure_hits = sum(term in response.lower() for term in STRUCTURE_TERMS)
    if structure_hits >= 2:
        score += 1
    if structure_hits >= 4:
        score += 1
    if "```" in response or re.search(r"(?m)^\s*(?:\$|#)\s+\w+", response):
        score += 1
    if not has_dangerous_command(response) or has_safety_context(response):
        score += 1
    return min(score, 10)


def simhash_tokens(text):
    lowered = re.sub(r"\s+", " ", text.lower())
    latin = re.findall(r"[a-z0-9_.:/-]{2,}", lowered)
    cjk_runs = re.findall(r"[\u3400-\u9fff]+", lowered)
    cjk = []
    for run in cjk_runs:
        if len(run) == 1:
            cjk.append(run)
        else:
            cjk.extend(run[index:index + 2] for index in range(len(run) - 1))
    return latin + cjk


def simhash64(text):
    tokens = simhash_tokens(text)
    if not tokens:
        return 0
    vector = [0] * 64
    counts = collections.Counter(tokens)
    for token, count in counts.items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        weight = min(count, 4)
        for bit in range(64):
            vector[bit] += weight if value & (1 << bit) else -weight
    result = 0
    for bit, value in enumerate(vector):
        if value >= 0:
            result |= 1 << bit
    return result


class NearDuplicateIndex:
    def __init__(self, max_distance=3):
        self.max_distance = max_distance
        self.bands = [collections.defaultdict(list) for _ in range(4)]

    @staticmethod
    def _keys(value):
        return [(value >> (index * 16)) & 0xFFFF for index in range(4)]

    def is_duplicate(self, value):
        candidates = set()
        for index, key in enumerate(self._keys(value)):
            candidates.update(self.bands[index].get(key, ()))
        return any((value ^ candidate).bit_count() <= self.max_distance
                   for candidate in candidates)

    def add(self, value):
        for index, key in enumerate(self._keys(value)):
            # Capping pathological buckets prevents unbounded comparisons.
            bucket = self.bands[index][key]
            if len(bucket) < 2000:
                bucket.append(value)


def prepare_database(path):
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS candidates (
            sample_id TEXT PRIMARY KEY,
            instruction TEXT NOT NULL,
            response TEXT NOT NULL,
            category TEXT NOT NULL,
            domain_score INTEGER NOT NULL,
            quality_score INTEGER NOT NULL,
            simhash TEXT NOT NULL,
            source_file TEXT NOT NULL,
            source_id TEXT NOT NULL,
            selected INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_candidates_category_quality "
        "ON candidates(category, quality_score DESC, domain_score DESC)"
    )
    return connection


def select_candidates(connection, target_size):
    selected = []
    for category, weight in CATEGORY_WEIGHTS.items():
        quota = max(1, round(target_size * weight))
        rows = connection.execute(
            """
            SELECT sample_id FROM candidates
            WHERE category = ?
            ORDER BY quality_score DESC, domain_score DESC, sample_id
            LIMIT ?
            """,
            (category, quota),
        ).fetchall()
        ids = [row[0] for row in rows]
        selected.extend(ids)
        connection.executemany(
            "UPDATE candidates SET selected = 1 WHERE sample_id = ?",
            [(sample_id,) for sample_id in ids],
        )

    remaining = max(0, target_size - len(selected))
    if remaining:
        rows = connection.execute(
            """
            SELECT sample_id FROM candidates
            WHERE selected = 0
            ORDER BY quality_score DESC, domain_score DESC, sample_id
            LIMIT ?
            """,
            (remaining,),
        ).fetchall()
        ids = [row[0] for row in rows]
        selected.extend(ids)
        connection.executemany(
            "UPDATE candidates SET selected = 1 WHERE sample_id = ?",
            [(sample_id,) for sample_id in ids],
        )
    connection.commit()
    return selected


def split_name(sample_id, train_ratio, validation_ratio):
    bucket = int(sample_id[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + validation_ratio:
        return "validation"
    return "test"


def write_outputs(connection, output_dir, args):
    output = Path(output_dir)
    sft_dir = output / "sft"
    metadata_dir = output / "metadata"
    reports_dir = output / "reports"
    for directory in (sft_dir, metadata_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    handles = {
        name: (sft_dir / f"{name}.jsonl").open("w", encoding="utf-8")
        for name in ("train", "validation", "test")
    }
    metadata_handle = (metadata_dir / "records.jsonl").open("w", encoding="utf-8")
    split_counts = collections.Counter()
    category_counts = collections.Counter()
    token_char_stats = []

    prefix = "<think>\n\n</think>\n\n" if args.add_empty_think else ""
    rows = connection.execute(
        """
        SELECT sample_id, instruction, response, category, domain_score,
               quality_score, simhash, source_file, source_id
        FROM candidates WHERE selected = 1
        ORDER BY category, sample_id
        """
    )
    try:
        for row in rows:
            (sample_id, instruction, response, category, domain_score,
             item_quality, fingerprint, source_file, source_id) = row
            split = split_name(sample_id, args.train_ratio, args.validation_ratio)
            assistant = response
            if prefix and not assistant.lstrip().startswith("<think>"):
                assistant = prefix + assistant
            record = {
                "id": sample_id,
                "category": category,
                "messages": [
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": assistant},
                ],
            }
            handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
            metadata = {
                "id": sample_id,
                "split": split,
                "category": category,
                "domain_score": domain_score,
                "quality_score": item_quality,
                "simhash": fingerprint,
                "source_dataset": args.dataset_id,
                "source_file": source_file,
                "source_id": source_id,
            }
            metadata_handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            split_counts[split] += 1
            category_counts[category] += 1
            token_char_stats.append(len(instruction) + len(assistant))
    finally:
        for handle in handles.values():
            handle.close()
        metadata_handle.close()

    return {
        "split_counts": dict(split_counts),
        "category_counts": dict(category_counts),
        "total_selected": sum(split_counts.values()),
        "character_length": {
            "min": min(token_char_stats) if token_char_stats else 0,
            "max": max(token_char_stats) if token_char_stats else 0,
            "mean": (
                round(sum(token_char_stats) / len(token_char_stats), 2)
                if token_char_stats else 0
            ),
        },
    }


def validate_ratios(train_ratio, validation_ratio):
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be less than 1")


def curate(args):
    validate_ratios(args.train_ratio, args.validation_ratio)
    raw_dir = Path(args.raw_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if raw_dir == output_dir or raw_dir in output_dir.parents:
        raise ValueError("output_dir must not equal or contain raw_dir")
    if (output_dir / "_SUCCESS").exists() and not args.overwrite:
        print(f"{output_dir} already completed; skip", flush=True)
        return 0
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.download_if_missing:
        download_dataset(args.dataset_id, raw_dir, args.revision, False)
    files = find_data_files(raw_dir)
    if not files:
        raise RuntimeError(f"no JSON/JSONL/Parquet files found under {raw_dir}")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    database_path = work_dir / "serverops-curation.sqlite3"
    if database_path.exists():
        database_path.unlink()
    connection = prepare_database(database_path)

    counters = collections.Counter()
    category_seen = collections.Counter()
    near_duplicates = NearDuplicateIndex(args.simhash_distance)
    rejected_path = output_dir / "reports" / "rejected_samples.jsonl"
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_handle = rejected_path.open("w", encoding="utf-8")
    random_generator = random.Random(args.seed)

    try:
        for path in files:
            relative_file = str(path.relative_to(raw_dir))
            print(f"processing {relative_file}", flush=True)
            for record in iter_records(path, args.batch_size):
                counters["total"] += 1
                if args.max_source_records and counters["total"] > args.max_source_records:
                    break
                instruction, response, source_id = extract_pair(record)
                reason = ""
                if not instruction or not response:
                    reason = "missing_pair"
                elif not args.min_instruction_chars <= len(instruction) <= args.max_instruction_chars:
                    reason = "instruction_length"
                elif not args.min_response_chars <= len(response) <= args.max_response_chars:
                    reason = "response_length"

                instruction = redact_secrets(instruction)
                response = redact_secrets(response)
                category = ""
                domain_score = 0
                all_scores = {}
                if not reason:
                    category, domain_score, all_scores = classify_domain(instruction, response)
                    if domain_score < args.min_domain_score:
                        reason = "low_domain_score"
                    elif looks_offensive_only(instruction + "\n" + response, domain_score):
                        reason = "offensive_only"
                    elif has_dangerous_command(response) and not has_safety_context(response):
                        reason = "unsafe_command_without_warning"

                item_quality = 0
                if not reason:
                    item_quality = quality_score(instruction, response, domain_score)
                    if item_quality < args.min_quality_score:
                        reason = "low_quality"

                normalized_question = re.sub(
                    r"\W+", "", instruction.lower(), flags=re.UNICODE
                )
                sample_id = stable_hash(normalized_question)
                fingerprint = simhash64(normalized_question)
                if not reason and near_duplicates.is_duplicate(fingerprint):
                    reason = "near_duplicate"

                if reason:
                    counters[f"rejected_{reason}"] += 1
                    # Bounded deterministic audit sample.
                    if counters[f"rejected_{reason}"] <= args.rejected_examples_per_reason:
                        rejected_handle.write(json.dumps({
                            "reason": reason,
                            "instruction": instruction[:2000],
                            "response": response[:4000],
                            "source_file": relative_file,
                            "scores": all_scores,
                        }, ensure_ascii=False) + "\n")
                    continue

                near_duplicates.add(fingerprint)
                category_seen[category] += 1
                counters["accepted"] += 1
                # Stable jitter prevents source ordering from deciding equal scores.
                jittered_quality = item_quality * 1000 + random_generator.randrange(1000)
                connection.execute(
                    """
                    INSERT INTO candidates(
                        sample_id, instruction, response, category, domain_score,
                        quality_score, simhash, source_file, source_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(sample_id) DO UPDATE SET
                        response = CASE
                            WHEN excluded.quality_score > quality_score
                            THEN excluded.response ELSE response END,
                        quality_score = MAX(quality_score, excluded.quality_score)
                    """,
                    (
                        sample_id, instruction, response, category, domain_score,
                        jittered_quality, f"{fingerprint:016x}", relative_file,
                        source_id,
                    ),
                )
                if counters["accepted"] % 2000 == 0:
                    connection.commit()
                    print(
                        f"scanned={counters['total']} accepted={counters['accepted']}",
                        flush=True,
                    )
            if args.max_source_records and counters["total"] >= args.max_source_records:
                break
        connection.commit()
    finally:
        rejected_handle.close()

    database_count = connection.execute(
        "SELECT COUNT(*) FROM candidates"
    ).fetchone()[0]
    target_size = min(args.target_size, database_count)
    select_candidates(connection, target_size)
    output_stats = write_outputs(connection, output_dir, args)

    report = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_id": args.dataset_id,
        "revision": args.revision,
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "source_files": [str(path.relative_to(raw_dir)) for path in files],
        "parameters": {
            "target_size": args.target_size,
            "min_domain_score": args.min_domain_score,
            "min_quality_score": args.min_quality_score,
            "simhash_distance": args.simhash_distance,
            "train_ratio": args.train_ratio,
            "validation_ratio": args.validation_ratio,
            "seed": args.seed,
            "add_empty_think": args.add_empty_think,
        },
        "scan_counts": dict(counters),
        "accepted_by_category_before_balancing": dict(category_seen),
        "unique_candidates": database_count,
        **output_stats,
    }
    reports_dir = output_dir / "reports"
    with (reports_dir / "statistics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    with (output_dir / "README.md").open("w", encoding="utf-8") as handle:
        handle.write(
            "# ServerOps SFT Dataset\n\n"
            f"- Source: `{args.dataset_id}`\n"
            f"- Created: `{report['created_at']}`\n"
            f"- Selected: `{output_stats['total_selected']}`\n"
            "- Format: ms-swift standard `messages` JSONL\n"
            "- Qwen3: train with `--loss_scale ignore_empty_think`\n"
        )
    with (output_dir / "_SUCCESS").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "created_at": report["created_at"],
            "selected": output_stats["total_selected"],
            "statistics": "reports/statistics.json",
        }, ensure_ascii=False) + "\n")
    connection.close()
    if not args.keep_work_db:
        database_path.unlink(missing_ok=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


def build_parser():
    parser = argparse.ArgumentParser("serverops-data-curation")
    parser.add_argument(
        "--dataset_id",
        default="hcnote/Cybersecurity-High-Quality-Dataset",
    )
    parser.add_argument("--revision", default="master")
    parser.add_argument(
        "--raw_dir",
        default="/mnt/storage/models-storage/dataset/cybersecurity-raw",
    )
    parser.add_argument(
        "--output_dir",
        default="/mnt/storage/models-storage/dataset/cybersecurity-serverops-v1",
    )
    parser.add_argument("--work_dir", default="/tmp/serverops-data-curation")
    parser.add_argument("--download_if_missing", type=bool_arg, default=True)
    parser.add_argument("--overwrite", type=bool_arg, default=False)
    parser.add_argument("--target_size", type=int, default=60000)
    parser.add_argument("--min_domain_score", type=int, default=3)
    parser.add_argument("--min_quality_score", type=int, default=5)
    parser.add_argument("--min_instruction_chars", type=int, default=10)
    parser.add_argument("--max_instruction_chars", type=int, default=4000)
    parser.add_argument("--min_response_chars", type=int, default=120)
    parser.add_argument("--max_response_chars", type=int, default=24000)
    parser.add_argument("--simhash_distance", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--train_ratio", type=float, default=0.90)
    parser.add_argument("--validation_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_source_records", type=int, default=0)
    parser.add_argument("--rejected_examples_per_reason", type=int, default=100)
    parser.add_argument("--add_empty_think", type=bool_arg, default=True)
    parser.add_argument("--keep_work_db", type=bool_arg, default=False)
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    return parser


def main():
    args = build_parser().parse_args()
    if args.target_size < 1:
        raise ValueError("target_size must be >= 1")
    return curate(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
