"""Deterministic semantic mapping; ambiguous answers require explicit encoding."""
import json
import re
from schema import SchemaError

ALIASES = {
    'question': ['question', 'query', 'problem', 'problem_text', 'prompt', 'input', '问题', '题目'],
    'answer': ['answer', 'response', 'solution', 'solution_text', 'output', 'target', '答案', '回答'],
    'context': ['context', 'background', 'passage', '背景', '背景信息'],
    'system': ['system', 'system_prompt', '系统提示词'],
    'choices': ['options', 'choices', 'candidates', '选项'],
    'choice_labels': ['option_ids', 'choice_labels', 'labels', '选项编号'],
    'correct': ['correct', 'gold', 'answer_index', 'label', '正确答案'],
    'id': ['id', 'sample_id'], 'category': ['category', 'subject', '类别'],
    'type': ['type', 'text_type', '题型'],
}

def error(message):
    raise SchemaError('semantic_mapping_error', message)

def parse_mapping(raw):
    try:
        mapping = json.loads(raw) if raw else {}
    except ValueError:
        error('field_mapping 必须为 JSON 对象')
    if not isinstance(mapping, dict):
        error('field_mapping 必须为源字段到语义字段的对象')
    names = {'user': 'question', 'input': 'question', 'assistant': 'answer', 'target': 'answer'}
    result = {}
    for source, target in mapping.items():
        if not isinstance(source, str) or not isinstance(target, str):
            error('字段映射必须为字符串')
        target = names.get(target, target)
        if target not in ALIASES or target in result:
            error('无效或重复的映射目标: %s' % target)
        result[target] = source
    return result

def resolve(record, mapping):
    if not isinstance(record, dict):
        error('样本必须是 JSON 对象')
    result, used = {}, set()
    for semantic, aliases in ALIASES.items():
        if semantic == 'correct' and 'answer' in mapping and 'correct' not in mapping:
            continue
        if semantic == 'answer' and 'correct' in mapping and 'answer' not in mapping:
            continue
        fields = [mapping[semantic]] if semantic in mapping else [k for k in record if k.casefold() in aliases and record[k] is not None]
        if len(fields) > 1:
            error('%s 存在多个候选字段 %s，请指定 field_mapping' % (semantic, fields))
        if fields:
            field = fields[0]
            if field not in record:
                error('缺少映射字段: %s' % field)
            result[semantic] = record[field]
            used.add(field)
    return result, used

def string(value, name):
    if not isinstance(value, str) or not value.strip():
        error('%s 必须是非空字符串' % name)
    return value.strip()

def label(value):
    value = re.sub(r'^选项', '', str(value).strip()).strip('(). ））、').upper()
    if not re.fullmatch('[A-Z]', value):
        error('选项标签须为 A-Z 单字母: %s' % value)
    return value

def choice_parts(parts, encoding):
    parts = dict(parts)
    for key in ('choices', 'choice_labels', 'answer', 'correct'):
        value = parts.get(key)
        if isinstance(value, str) and value.strip().startswith(('[', '{')):
            try:
                parts[key] = json.loads(value)
            except ValueError:
                error('%s 中的 JSON 数组/对象无效' % key)
    options = parts['choices']
    labels = parts.get('choice_labels')
    if isinstance(options, dict):
        if labels is not None:
            error('字典选项已有标签，不能同时指定标签数组')
        labels, options = list(options), list(options.values())
    if not isinstance(options, list) or not 2 <= len(options) <= 26:
        error('选项必须为包含 2-26 项的数组或对象')
    if labels is not None and (not isinstance(labels, list) or len(labels) != len(options)):
        error('选项标签与选项数量不一致')
    choices = []
    for i, option in enumerate(options):
        lab = labels[i] if labels is not None else chr(65 + i)
        if isinstance(option, dict):
            lab = option.get('label', lab)
            option = option.get('text', option.get('value'))
        choices.append({'label': label(lab), 'text': string(option, '选项内容')})
    available = [x['label'] for x in choices]
    if len(set(available)) != len(available):
        error('选项标签重复')
    if 'correct' in parts and 'answer' in parts:
        error('同时存在 correct 与 answer，请明确原数据答案字段')
    raw = parts.get('correct', parts.get('answer'))
    if encoding in ('auto', 'label') and isinstance(raw, str) and re.fullmatch(r'[A-Za-z](?:\s*[,，、;]\s*[A-Za-z])+', raw.strip()):
        raw = re.split(r'\s*[,，、;]\s*', raw.strip())
    values = raw if isinstance(raw, list) else [raw]
    if encoding == 'bool_array' or (encoding == 'auto' and values and all(type(x) is bool for x in values)):
        if not isinstance(raw, list) or len(raw) != len(choices) or any(type(x) is not bool for x in raw):
            error('布尔答案数组须与选项等长，且元素为 true/false')
        targets = [available[i] for i, value in enumerate(raw) if value]
    else:
        targets = []
        for value in values:
            if encoding in ('index0', 'index1'):
                if isinstance(value, bool) or not re.fullmatch(r'\d+', str(value)):
                    error('索引答案须为非负整数')
                index = int(value) - (encoding == 'index1')
                if not 0 <= index < len(choices):
                    error('答案索引超出选项范围')
                targets.append(available[index])
                continue
            matches = []
            if encoding in ('auto', 'label') and isinstance(value, str):
                cleaned = re.sub(r'^选项', '', value.strip()).strip('(). ））、').upper()
                matches += [x['label'] for x in choices if x['label'] == cleaned]
            if encoding in ('auto', 'text') and isinstance(value, str):
                matches += [x['label'] for x in choices if x['text'].casefold() == value.strip().casefold()]
            if len(set(matches)) != 1:
                error('答案无法唯一匹配选项；数字答案请指定 answer_encoding=index0/index1')
            targets.append(matches[0])
    if not targets or len(targets) != len(set(targets)):
        error('正确答案为空或重复')
    declared = str(parts.get('type', '')).lower()
    multi = declared in ('multi_choice', 'multiple_choice', '多选', '多选题') or len(targets) > 1
    if declared in ('single_choice', '单选', '单选题') and len(targets) != 1:
        error('单选题必须恰好一个正确答案')
    targets = [x for x in available if x in targets]
    return choices, targets if multi else targets[0], 'multi_choice' if multi else 'choice'

def convert_semantic(record, target, mapping, encoding):
    parts, consumed = resolve(record, mapping)
    question = string(parts.get('question'), '问题')
    context = parts.get('context')
    if context:
        question = string(context, '背景') + '\n\n' + question
    choices = None
    if 'choices' in parts:
        choices, answer, kind = choice_parts(parts, encoding)
    else:
        raw_answer = parts.get('answer')
        answer = string(str(raw_answer) if type(raw_answer) in (int, float) else raw_answer, '答案')
        kind = str(parts.get('type') or 'short_answer')
        if kind not in ('short_answer', 'numeric', 'contains'):
            error('不支持的问答题型: %s' % kind)
    if target == 'eval_qa':
        output = {'input': question, 'target': answer, 'type': kind}
        if choices:
            output['choices'] = choices
        for field in ('id', 'category'):
            if field in parts:
                output[field] = str(parts[field])
        if kind == 'contains' and 'keywords' in record:
            output['keywords'] = record['keywords']
            consumed.add('keywords')
    elif target == 'messages':
        evaluation = None
        if choices:
            evaluation = {'version': 1, 'input': question, 'choices': choices, 'type': kind}
            for field in ('id', 'category'):
                if field in parts:
                    evaluation[field] = str(parts[field])
            question += '\n\n' + '\n'.join('%s. %s' % (x['label'], x['text']) for x in choices)
        messages = []
        if parts.get('system'):
            messages.append({'role': 'system', 'content': string(parts['system'], 'system')})
        messages += [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': ', '.join(answer) if isinstance(answer, list) else answer}]
        output = {'messages': messages}
        if evaluation:
            output['metadata'] = {'evaluation': evaluation}
    else:
        error('语义转换仅支持 messages/eval_qa')
    return output, consumed


def messages_to_eval(record, turns):
    """Restore typed single-turn evaluation without exposing the assistant answer."""
    if not turns or turns[-1]['role'] != 'assistant':
        error('评测对话必须以 assistant 答案结束')
    systems = [m['content'] for m in turns[:-1] if m['role'] == 'system']
    history = [m for m in turns[:-1] if m['role'] != 'system']
    metadata = record.get('metadata') or {}
    spec = metadata.get('evaluation') if isinstance(metadata, dict) else None
    if spec is not None:
        if not isinstance(spec, dict) or spec.get('version') != 1:
            error('不支持的 evaluation 元数据版本')
        if len(history) != 1 or history[0]['role'] != 'user':
            error('选择题元数据仅支持单轮对话，禁止猜测目标轮次')
        question = string(spec.get('input'), '原始问题')
        choices, answer, kind = choice_parts(
            {'choices': spec.get('choices'), 'answer': turns[-1]['content'], 'type': spec.get('type')}, 'label')
        expected = question + '\n\n' + '\n'.join('%s. %s' % (x['label'], x['text']) for x in choices)
        if history[0]['content'].strip() != expected:
            error('messages 问题/选项与 evaluation 元数据不一致，请重新转换')
        output = dict(input=question, choices=choices, target=answer, type=kind)
        for field in ('id', 'category'):
            if field in spec:
                output[field] = spec[field]
    else:
        output = {'input': '\n\n'.join('%s: %s' % (m['role'], m['content']) for m in history),
                  'target': turns[-1]['content'], 'type': 'short_answer'}
    if systems:
        output['system'] = '\n\n'.join(systems)
    return output
