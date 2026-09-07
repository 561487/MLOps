# -*- coding: utf-8 -*-
"""dataset-convert Schema 层：源结构识别、messages 校验、role 规范化、目标结构校验、多模态检测。

- 源结构：text / alpaca / messages / sharegpt / qa / prompt_response / custom
- 目标结构：messages / text / eval_qa
- 自动识别按固定优先级（见 DETECTION_ORDER），不做机器学习式推断。
- 目前仅支持文本；发现多模态字段/内容一律明确拒绝
"""

import collections
import json

from dataset_io import detect_format, iter_records

SOURCE_SCHEMAS = ["text", "alpaca", "messages", "sharegpt", "qa", "prompt_response", "custom"]
TARGET_SCHEMAS = ["messages", "text", "eval_qa"]
ALLOWED_ROLES = {"system", "user", "assistant"}
ROLE_ALIASES = {
    "human": "user", "user": "user", "question": "user",
    "gpt": "assistant", "bot": "assistant", "assistant": "assistant", "answer": "assistant",
    "system": "system",
}
# 多模态字段探测表：顶层发现任一字段即拒绝，不允许悄悄丢弃。
MULTIMODAL_FIELDS = {"image", "images", "image_path", "video", "videos", "video_path", "audio", "audios", "audio_path"}
MULTIMODAL_REASON = "multimodal_schema_not_supported_v1"
MULTIMODAL_MESSAGE = "multimodal messages not supported in V1"

# 自动识别优先级（高 → 低）；custom 无法自动识别，必须显式指定 source_schema=custom。
DETECTION_ORDER = ["messages", "sharegpt", "alpaca", "prompt_response", "qa", "text"]
DETECT_SAMPLE_LIMIT = 50  # 自动识别仅采样前若干条记录
CONTROL_CHARACTERS = __import__("re").compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class SchemaError(Exception):
    """单条记录级校验/转换错误：reason 用于 invalid_reasons 统计，message 用于日志。"""

    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason
        self.message = message


def clean_text(value):
    """去除控制字符并 trim；dict/list 转 JSON 文本（与旧组件一致）。"""
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    elif isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)
    return CONTROL_CHARACTERS.sub("", text).strip()


def text_field(value, field):
    """内容类字段取值：list/dict 视为多模态内容，明确拒绝，不允许静默 JSON 化。"""
    if isinstance(value, (list, dict)):
        raise SchemaError(MULTIMODAL_REASON, "%s（字段 %s 为多模态内容，%s）" % (MULTIMODAL_MESSAGE, field, MULTIMODAL_REASON))
    return clean_text(value)


def require_field(record, field, reason):
    """取必填字段：缺失或为空（trim 后）抛 SchemaError。"""
    if not isinstance(record, dict) or field not in record:
        raise SchemaError(reason, "缺少字段 %s" % field)
    value = text_field(record.get(field), field)
    if not value:
        raise SchemaError(reason, "字段 %s 为空" % field)
    return value


# ---------------------------------------------------------------- 源结构自动识别

def detect_record_schema(record):
    """单条记录按 DETECTION_ORDER 优先级返回命中的源结构，未命中返回 None。"""
    if not isinstance(record, dict):
        return None
    if isinstance(record.get("messages"), list):
        return "messages"
    if isinstance(record.get("conversations"), list):
        return "sharegpt"
    if "instruction" in record and "output" in record:
        return "alpaca"
    if "prompt" in record and "response" in record:
        return "prompt_response"
    if ("question" in record and "answer" in record) or ("query" in record and "response" in record):
        return "qa"
    if isinstance(record.get("text"), str):
        return "text"
    return None


def detect_dataset_schema(files, input_format, encoding="utf-8"):
    """采样前若干条记录，按计数（同票按优先级）决定数据集源结构。

    返回 (source_schema | None, warnings)。冲突时输出 warning 供 launcher 打印。
    """
    counts = collections.Counter()
    unknown = 0
    total = 0
    for path in files:
        data_format = detect_format(path, input_format)
        for record, _metadata, read_error in iter_records(path, data_format, encoding):
            total += 1
            if total > DETECT_SAMPLE_LIMIT:
                break
            if read_error:
                unknown += 1
                continue
            detected = detect_record_schema(record)
            if detected:
                counts[detected] += 1
            else:
                unknown += 1
        if total > DETECT_SAMPLE_LIMIT:
            break
    warnings = []
    if not counts:
        return None, warnings
    final = max(counts, key=lambda key: (counts[key], -DETECTION_ORDER.index(key)))
    if len(counts) > 1 or unknown > 0:
        detail = ", ".join("%s=%d" % (key, value) for key, value in sorted(counts.items()))
        if unknown:
            detail += ", unknown=%d" % unknown
        warnings.append("自动识别出现冲突/未识别样本，按优先级取 %s（采样%d条：%s）" % (final, total, detail))
    return final, warnings


# ---------------------------------------------------------------- messages 校验

def normalize_role(raw_role):
    """role 规范化：human→user、gpt→assistant 等；未知 role 抛 invalid_role（不允许静默丢弃）。"""
    normalized = raw_role.strip().lower() if raw_role else ""
    if normalized not in ROLE_ALIASES:
        raise SchemaError("invalid_role", "未知角色 %r（允许: system/user/assistant/human/gpt/bot）" % (raw_role or ""))
    return ROLE_ALIASES[normalized]


def validate_messages(value):
    """校验并规范化 messages 列表。

    规则：必须为 list；每项 dict；必须有 role/content；role 在允许集合；content 非空；
    至少一个 user 和一个 assistant 轮次；content 为 list/dict 时明确拒绝（多模态）。
    返回规范化后的 messages 列表；失败抛 SchemaError。
    """
    if not isinstance(value, list):
        raise SchemaError("not_a_list", "messages 必须是 list")
    messages = []
    for item in value:
        if not isinstance(item, dict):
            raise SchemaError("not_a_dict", "messages 每项必须是 dict")
        if "role" not in item and "from" not in item:
            raise SchemaError("missing_role", "messages 项缺少 role")
        role = normalize_role(item.get("role") if "role" in item else item.get("from"))
        content = item.get("content") if "content" in item else item.get("value")
        if content is None:
            raise SchemaError("missing_content", "messages 项缺少 content")
        if isinstance(content, (list, dict)):
            raise SchemaError(MULTIMODAL_REASON, "%s（%s）" % (MULTIMODAL_MESSAGE, MULTIMODAL_REASON))
        if not isinstance(content, str):
            raise SchemaError("empty_content", "messages 项 content 必须为字符串")
        cleaned = clean_text(content)
        if not cleaned:
            raise SchemaError("empty_content", "messages 项 content 为空")
        messages.append({"role": role, "content": cleaned})
    if not any(item["role"] == "user" for item in messages) or not any(item["role"] == "assistant" for item in messages):
        raise SchemaError("missing_turn", "messages 至少需要一条 user 和一条 assistant 轮次")
    return messages


def check_multimodal_record(record):
    """顶层多模态字段探测：命中返回 reason，未命中返回 None。"""
    if not isinstance(record, dict):
        return None
    present = [field for field in MULTIMODAL_FIELDS if field in record]
    if present:
        return MULTIMODAL_REASON
    return None
