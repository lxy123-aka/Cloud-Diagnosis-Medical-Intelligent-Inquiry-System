"""
utils/fallback_llm.py
=====================
带自动模型降级的 ChatOpenAI 替代品。

当主模型额度用完（429/402）时，自动切换到备用模型。
用法：将所有 ChatOpenAI(...) 替换为 FallbackChatOpenAI(...) 即可。
支持 bind_tools().ainvoke() 和直接 ainvoke() 两种调用模式。
"""

from __future__ import annotations
from loguru import logger
from langchain_openai import ChatOpenAI

from configs.settings import settings


def _is_rate_limit_error(e: Exception) -> bool:
    """判断是否为额度/限流类错误（可降级到其他模型）"""
    status = getattr(e, "status_code", None)
    err_str = str(e).lower()
    return (
        status in (429, 402)
        or "rate limit" in err_str
        or "quota" in err_str
    )


class _BoundFallback:
    """
    bind_tools() 返回的代理对象。
    在 ainvoke 时遍历模型降级链，对每个模型 bind_tools 后调用。
    """

    def __init__(self, models: list[ChatOpenAI], tools, tool_choice):
        self._models = models
        self._tools = tools
        self._tool_choice = tool_choice

    async def ainvoke(self, messages, **kwargs):
        last_error = None
        for i, model in enumerate(self._models):
            try:
                bound = model.bind_tools(
                    self._tools, tool_choice=self._tool_choice
                )
                return await bound.ainvoke(messages, **kwargs)
            except Exception as e:
                if _is_rate_limit_error(e) and i < len(self._models) - 1:
                    logger.warning(
                        f"[FallbackLLM] 模型 {model.model_name} 额度用尽，"
                        f"降级到 {self._models[i + 1].model_name}..."
                    )
                    last_error = e
                    continue
                raise
        raise last_error


class FallbackChatOpenAI:
    """
    ChatOpenAI 的降级替代品。

    自动从 settings 加载 API Key / Base URL / 模型降级列表。
    对外接口兼容 ChatOpenAI（支持 bind_tools + ainvoke）。
    """

    def __init__(self, temperature: float = 0.1, **kwargs):
        # 构建模型降级链：主模型 + 备用模型
        self._models: list[ChatOpenAI] = []
        model_names = [settings.QWEN_MODEL_NAME]
        for m in settings.QWEN_FALLBACK_MODELS.split(","):
            m = m.strip()
            if m and m not in model_names:
                model_names.append(m)

        for name in model_names:
            self._models.append(ChatOpenAI(
                model=name,
                api_key=settings.QWEN_API_KEY,
                base_url=settings.QWEN_BASE_URL,
                temperature=temperature,
                **kwargs,
            ))

    @property
    def model_name(self) -> str:
        """返回主模型名称（兼容旧代码）"""
        return self._models[0].model_name

    def bind_tools(self, tools, tool_choice=None):
        """
        绑定工具，返回降级代理对象。
        代理的 ainvoke 会在主模型失败时自动切换备用模型。
        """
        return _BoundFallback(self._models, tools, tool_choice)

    async def ainvoke(self, messages, **kwargs):
        """直接调用（无工具绑定），支持降级"""
        last_error = None
        for i, model in enumerate(self._models):
            try:
                return await model.ainvoke(messages, **kwargs)
            except Exception as e:
                if _is_rate_limit_error(e) and i < len(self._models) - 1:
                    logger.warning(
                        f"[FallbackLLM] 模型 {model.model_name} 额度用尽，"
                        f"降级到 {self._models[i + 1].model_name}..."
                    )
                    last_error = e
                    continue
                raise
        raise last_error
