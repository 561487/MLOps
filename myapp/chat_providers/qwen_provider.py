"""
QwenChatProvider — qwen3 Provider
====================================
兼容 OpenAI Chat Completions 协议，用于对接：
  http://10.80.10.146:9998/v1/chat/completions

相比现有 view_chat.py 的 chatgpt() 方法：
  - 凭证从 credentials 字段读取（而非 service_config.llm_tokens）
  - 支持 chat_template_kwargs（enable_thinking）
  - 模块化，可独立替换
"""
import json
from .base import BaseChatProvider


class QwenChatProvider(BaseChatProvider):
    """qwen3 Provider（OpenAI 兼容协议）"""

    def build_request(self, chat_record, messages, stream=True):
        """
        构建 qwen3 API 请求。

        凭证读取优先级：
          1. credentials.apiKey / credentials.apiEndpoint（用户配置）
          2. service_config.llm_tokens / service_config.llm_url（降级兼容）
        """
        # 解析 service_config（通用模型配置）
        config = json.loads(chat_record.service_config) if chat_record.service_config else {}

        # 解析 credentials（用户凭证配置）
        creds = json.loads(chat_record.credentials) if chat_record.credentials else {}

        # ---- URL ----
        # 优先使用 credentials 中的 apiEndpoint，否则降级到 service_config.llm_url
        url = creds.get('apiEndpoint') or config.get('llm_url', '')

        # ---- Headers ----
        api_key = creds.get('apiKey', '')
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        # 合并 service_config 中的自定义 headers
        custom_headers = config.get('llm_headers', {})
        headers.update(custom_headers)

        # ---- Body ----
        # 从 service_config.llm_data 读取模型参数（与原 chatgpt() 方法兼容）
        llm_data = config.get('llm_data', {})

        body = {
            "model": llm_data.get('model', 'qwen3'),
            "messages": messages,
            "temperature": llm_data.get('temperature', 0.7),
            "max_tokens": llm_data.get('max_tokens', 512),
            "stream": stream,
        }

        # 透传 chat_template_kwargs（控制 enable_thinking 等）
        chat_tpl = llm_data.get('chat_template_kwargs')
        if chat_tpl:
            body['chat_template_kwargs'] = chat_tpl

        # 透传 credentials 中的扩展参数
        extra_params = creds.get('extraParams', {})
        if extra_params:
            body.update(extra_params)

        return url, headers, body

    def parse_chunk(self, chunk_text: str) -> str:
        """
        解析 OpenAI 兼容 SSE 格式

        输入格式: data: {"id":"...","choices":[{"delta":{"content":"你好"}}],...}
        输出: "你好"（增量文本）

        特殊处理:
          - data: [DONE] → 返回 ''（流结束标记）
          - 非 data: 开头 → 返回 ''（注释行）
        """
        if not chunk_text.startswith('data: '):
            return ''

        payload = chunk_text[6:]  # 去掉 "data: " 前缀

        if payload.strip() == '[DONE]':
            return ''

        try:
            data = json.loads(payload)
            choices = data.get('choices', [])
            if choices:
                delta = choices[0].get('delta', {})
                return delta.get('content', '')
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

        return ''
