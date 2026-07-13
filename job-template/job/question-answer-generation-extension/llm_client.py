"""OpenAI SDK adapter for JSON Chat Completions."""

import json
import time

import openai
from openai import OpenAI


class ModelCallError(RuntimeError):
    """Raised when the model response cannot be obtained or validated."""


def normalize_chat_url(base_url):
    url = base_url.strip().rstrip("/")
    if not url:
        raise ValueError("base_url 不能为空")
    suffix = "/chat/completions"
    if url.endswith(suffix):
        return url[:-len(suffix)].rstrip("/")
    if url.endswith("/v1"):
        return url
    raise ValueError("base_url 必须以 /v1 或 /chat/completions 结尾")


def extract_json_object(content):
    if not isinstance(content, str):
        raise ModelCallError("模型内容不是字符串")

    decoder = json.JSONDecoder()
    for index, char in enumerate(content):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ModelCallError("模型内容不包含合法 JSON 对象")


def _is_retryable_status(error):
    return error.status_code in (408, 409, 429) or error.status_code >= 500


def _extract_response_content(response):
    try:
        return response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as error:
        raise ModelCallError("模型响应缺少消息内容") from error


class OpenAIChatClient:
    def __init__(
        self,
        base_url,
        api_key,
        model_name,
        temperature,
        timeout,
        max_retries,
        sleep=time.sleep,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.sleep = sleep
        self.client = OpenAI(
            api_key=api_key or "not-needed",
            base_url=normalize_chat_url(base_url),
            timeout=timeout,
            max_retries=0,
        )

    def complete_json(self, system_prompt, user_prompt, validator=None):
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    temperature=self.temperature,
                    stream=False,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = _extract_response_content(response)
                value = extract_json_object(content)
                return validator(value) if validator else value
            except openai.APIStatusError as error:
                last_error = error
                if not _is_retryable_status(error) or attempt == self.max_retries:
                    break
            except (
                openai.APIError,
                ModelCallError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
            ) as error:
                last_error = error
                if attempt == self.max_retries:
                    break
            self.sleep(min(2 ** attempt, 8))

        error_type = type(last_error).__name__ if last_error else "UnknownError"
        raise ModelCallError("模型调用失败: " + error_type)
