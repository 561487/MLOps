"""
BaseChatProvider — 所有 LLM Provider 的抽象基类
====================================================
定义了两个必须实现的方法：
  1. build_request() — 构建 HTTP 请求参数
  2. parse_chunk()   — 解析 SSE 流式数据块
"""


class BaseChatProvider:
    """Provider 抽象基类，所有 LLM Provider 必须继承此类"""

    def build_request(self, chat_record, messages, stream=True):
        """
        构建 LLM API 请求参数。

        Args:
            chat_record: Chat ORM 对象（含 service_config 和 credentials）
            messages:    [{"role": "user/assistant/system", "content": "..."}]
            stream:      是否流式响应

        Returns:
            (url, headers, body_dict) 三元组
                - url:     str, LLM API 端点
                - headers: dict, HTTP 请求头
                - body:    dict, 请求体 JSON
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement build_request()"
        )

    def parse_chunk(self, chunk_text: str) -> str:
        """
        解析 LLM 返回的单个 SSE 数据块，提取文本增量。

        Args:
            chunk_text: 原始 SSE 行文本，如 'data: {"choices":[{"delta":{"content":"你好"}}]}'

        Returns:
            提取到的文本增量字符串，无内容则返回 ''
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement parse_chunk()"
        )
