"""
Chat Provider 注册表
========================
所有 LLM Provider 在此注册，通过 service_type 查找对应的 Provider 类。

新增 Provider 只需：
1. 继承 BaseChatProvider
2. 实现 build_request() 和 parse_chunk()
3. 在 CHAT_PROVIDERS 中注册
"""
from .qwen_provider import QwenChatProvider

# ===== Provider 注册表 =====
# key: service_type 字段的值（来自 chat 表）
# value: Provider 类
CHAT_PROVIDERS = {
    'openai': QwenChatProvider,   # OpenAI 兼容协议（qwen3 也是这个）
    'qwen': QwenChatProvider,     # 显式 qwen 入口
}


def get_provider(service_type: str):
    """
    根据 service_type 获取对应的 Provider 实例。

    Args:
        service_type: chat 表的 service_type 字段值

    Returns:
        BaseChatProvider 实例

    Raises:
        ValueError: 未知的 service_type
    """
    provider_cls = CHAT_PROVIDERS.get(service_type)
    if not provider_cls:
        raise ValueError(
            f"Unknown service_type: '{service_type}'. "
            f"Available: {list(CHAT_PROVIDERS.keys())}"
        )
    return provider_cls()
