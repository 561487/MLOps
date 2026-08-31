# -*- coding: utf-8 -*-
"""dataset-convert 转换层：纯函数实现 V1 转换矩阵。

矩阵（严格按 V1 方案，不实现无明确语义的转换）：

    Source\Target      messages     text          eval_qa
    messages           规范化 ✅     ❌            暂不支持 ❌
    sharegpt           ✅          ❌            暂不支持 ❌
    alpaca             ✅          ❌            ✅
    qa                 ✅          ❌            ✅
    prompt_response    ✅          ❌            ✅
    text               ❌(明确报错) 透传 ✅        ❌
    custom             ✅          ✅            ✅

多模态扩展位置：未来 V1.1 在 convert() 分派前识别 multimodal_messages，
并在 media 字段上调用媒体校验/路径规范化；V1 一律抛 multimodal_schema_not_supported_v1。
"""

import json

from schema import (
    MULTIMODAL_MESSAGE,
    MULTIMODAL_REASON,
    SchemaError,
    clean_text,
    normalize_role,
    require_field,
    text_field,
    validate_messages,
)

# 允许的转换组合；text→messages 单独给出明确错误文案。
SUPPORTED_CONVERSIONS = {
    ("messages", "messages"),
    ("sharegpt", "messages"),
    ("alpaca", "messages"),
    ("alpaca", "eval_qa"),
    ("qa", "messages"),
    ("qa", "eval_qa"),
    ("prompt_response", "messages"),
    ("prompt_response", "eval_qa"),
    ("text", "text"),
    ("custom", "messages"),
    ("custom", "text"),
    ("custom", "eval_qa"),
}

TEXT_TO_MESSAGES_MESSAGE = (
    "纯文本数据无法自动确定 user / assistant 角色，请使用 target_schema=text，"
    "或通过 source_schema=custom + field_mapping 明确指定字段语义。"
)

# field_mapping 目标字段白名单（按 target_schema）
MAPPING_TARGETS = {"messages": {"user", "assistant"}, "text": {"text"}, "eval_qa": {"input", "target"}}
# 目标结构必填的映射目标
MAPPING_REQUIRED = {"messages": {"user", "assistant"}, "text": {"text"}, "eval_qa": {"input", "target"}}


def check_supported(source_schema, target_schema):
    """配置级校验：不支持的转换组合抛清晰错误。"""
    if (source_schema, target_schema) in SUPPORTED_CONVERSIONS:
        return
    if source_schema == "text" and target_schema == "messages":
        raise SchemaError("unsupported_conversion", TEXT_TO_MESSAGES_MESSAGE)
    allowed = sorted(target for src, target in SUPPORTED_CONVERSIONS if src == source_schema)
    raise SchemaError(
        "unsupported_conversion",
        "不支持从 %s 转换到 %s（允许的目标: %s）" % (source_schema, target_schema, ", ".join(allowed) if allowed else "无"),
    )


def validate_field_mapping(mapping_json, source_schema, target_schema):
    """配置级校验 field_mapping：JSON 合法性、目标字段白名单、必填目标、重复映射。

    返回 {源字段: 目标字段}；非法时抛 SchemaError（任务直接失败）。
    """
    mapping = {}
    if source_schema != "custom":
        if mapping_json:
            raise SchemaError("field_mapping_invalid", "field_mapping 仅用于 source_schema=custom")
        return mapping
    if not mapping_json:
        raise SchemaError("field_mapping_invalid", "source_schema=custom 必须提供 --field_mapping")
    try:
        parsed = json.loads(mapping_json)
    except ValueError as error:
        raise SchemaError("field_mapping_invalid", "field_mapping 不是合法 JSON: %s" % error)
    if not isinstance(parsed, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in parsed.items()):
        raise SchemaError("field_mapping_invalid", "field_mapping 必须是 JSON 对象，形如 {\"源字段\": \"user\"}")
    allowed = MAPPING_TARGETS[target_schema]
    for source_field, target_field in parsed.items():
        if target_field not in allowed:
            raise SchemaError(
                "unknown_target_field",
                "field_mapping 目标字段 %r 不符合 target_schema=%s（允许: %s）" % (target_field, target_schema, ", ".join(sorted(allowed))),
            )
    reversed_map = {}
    for source_field, target_field in parsed.items():
        if target_field in reversed_map:
            raise SchemaError("field_mapping_invalid", "field_mapping 目标字段 %s 被重复映射: %s 与 %s" % (target_field, reversed_map[target_field], source_field))
        reversed_map[target_field] = source_field
    missing = sorted(MAPPING_REQUIRED[target_schema] - set(reversed_map))
    if missing:
        raise SchemaError("field_mapping_invalid", "field_mapping 缺少目标字段: %s" % ", ".join(missing))
    return dict(parsed)


# ---------------------------------------------------------------- 各转换函数（纯函数）

def convert_messages_to_messages(record, **kwargs):
    """messages → messages：校验 + role/content 规范化。"""
    return {"messages": validate_messages(record.get("messages"))}, {"messages"}


def convert_sharegpt_to_messages(record, **kwargs):
    """sharegpt → messages：human/gpt 等 role 别名；未知 role 拒绝（invalid_role）。"""
    raw = record.get("conversations")
    if not isinstance(raw, list):
        raise SchemaError("not_a_list", "conversations 必须是 list")
    messages = []
    for item in raw:
        if not isinstance(item, dict):
            raise SchemaError("not_a_dict", "conversations 每项必须是 dict")
        if "role" not in item and "from" not in item:
            raise SchemaError("missing_role", "conversations 项缺少 role/from")
        role = normalize_role(item.get("role") if "role" in item else item.get("from"))
        content = item.get("content") if "content" in item else item.get("value")
        if content is None:
            raise SchemaError("missing_content", "conversations 项缺少 content/value")
        if isinstance(content, (list, dict)):
            raise SchemaError(MULTIMODAL_REASON, "%s（%s）" % (MULTIMODAL_MESSAGE, MULTIMODAL_REASON))
        if not isinstance(content, str):
            raise SchemaError("empty_content", "conversations 项 content 必须为字符串")
        cleaned = clean_text(content)
        if not cleaned:
            raise SchemaError("empty_content", "conversations 项内容为空")
        messages.append({"role": role, "content": cleaned})
    if not any(item["role"] == "user" for item in messages) or not any(item["role"] == "assistant" for item in messages):
        raise SchemaError("missing_turn", "conversations 至少需要一条 user 和一条 assistant 轮次")
    return {"messages": messages}, {"conversations"}


def convert_alpaca_to_messages(record, **kwargs):
    """alpaca → messages：user = instruction(+\\ninput)，assistant = output。"""
    user_content, output = alpaca_parts(record)
    return {"messages": [{"role": "user", "content": user_content}, {"role": "assistant", "content": output}]}, {"instruction", "input", "output"}


def convert_alpaca_to_eval_qa(record, **kwargs):
    """alpaca → eval_qa：input = instruction(+\\ninput)，target = output。"""
    user_content, output = alpaca_parts(record)
    return {"input": user_content, "target": output}, {"instruction", "input", "output"}


def alpaca_parts(record):
    instruction = require_field(record, "instruction", "missing_instruction")
    output = require_field(record, "output", "missing_output")
    input_part = text_field(record.get("input", ""), "input")
    user_content = instruction + ("\n" + input_part if input_part else "")
    return user_content, output


def convert_qa_to_messages(record, **kwargs):
    question, answer = qa_parts(record)
    return {"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]}, {"question", "answer", "query", "response"}


def convert_qa_to_eval_qa(record, **kwargs):
    question, answer = qa_parts(record)
    return {"input": question, "target": answer}, {"question", "answer", "query", "response"}


def qa_parts(record):
    """qa 取值：优先 question/answer，其次 query/response（严格配对，不混搭）。"""
    if "question" in record and "answer" in record:
        return require_field(record, "question", "missing_question"), require_field(record, "answer", "missing_answer")
    if "query" in record and "response" in record:
        return require_field(record, "query", "missing_question"), require_field(record, "response", "missing_answer")
    raise SchemaError("missing_question", "qa 数据需要 question/answer 或 query/response 字段")


def convert_pr_to_messages(record, **kwargs):
    """prompt_response → messages：prompt→user，response→assistant。"""
    prompt = require_field(record, "prompt", "missing_prompt")
    response = require_field(record, "response", "missing_response")
    return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]}, {"prompt", "response"}


def convert_pr_to_eval_qa(record, **kwargs):
    """prompt_response → eval_qa：prompt→input，response→target。"""
    prompt = require_field(record, "prompt", "missing_prompt")
    response = require_field(record, "response", "missing_response")
    return {"input": prompt, "target": response}, {"prompt", "response"}


def convert_text_to_text(record, **kwargs):
    """text → text：透传。"""
    return {"text": require_field(record, "text", "missing_text")}, {"text"}


def convert_custom(record, target_schema, field_mapping, **kwargs):
    """custom：按 field_mapping 取值构造目标结构（目标字段已在配置期校验）。"""
    source_by_target = {target: source for source, target in field_mapping.items()}
    values = {}
    for target_field in sorted(MAPPING_REQUIRED[target_schema]):
        source_field = source_by_target[target_field]
        if source_field not in record:
            raise SchemaError("missing_field", "缺少源字段 %s（映射目标 %s）" % (source_field, target_field))
        values[target_field] = text_field(record.get(source_field), source_field)
        if not values[target_field]:
            raise SchemaError("missing_field", "源字段 %s 为空（映射目标 %s）" % (source_field, target_field))
    if target_schema == "messages":
        return {"messages": [{"role": "user", "content": values["user"]}, {"role": "assistant", "content": values["assistant"]}]}, set(field_mapping)
    if target_schema == "eval_qa":
        return {"input": values["input"], "target": values["target"]}, set(field_mapping)
    return {"text": values["text"]}, set(field_mapping)


# ---------------------------------------------------------------- 分派

_CONVERTERS = {
    ("messages", "messages"): convert_messages_to_messages,
    ("sharegpt", "messages"): convert_sharegpt_to_messages,
    ("alpaca", "messages"): convert_alpaca_to_messages,
    ("alpaca", "eval_qa"): convert_alpaca_to_eval_qa,
    ("qa", "messages"): convert_qa_to_messages,
    ("qa", "eval_qa"): convert_qa_to_eval_qa,
    ("prompt_response", "messages"): convert_pr_to_messages,
    ("prompt_response", "eval_qa"): convert_pr_to_eval_qa,
    ("text", "text"): convert_text_to_text,
}


def convert(record, source_schema, target_schema, field_mapping=None):
    """转换单条记录：返回 (输出dict, 已消费字段set)；失败抛 SchemaError。"""
    if (source_schema, target_schema) in _CONVERTERS:
        return _CONVERTERS[(source_schema, target_schema)](record)
    if source_schema == "custom":
        return convert_custom(record, target_schema, field_mapping or {})
    raise SchemaError(
        "unsupported_conversion",
        "不支持从 %s 转换到 %s" % (source_schema, target_schema),
    )
