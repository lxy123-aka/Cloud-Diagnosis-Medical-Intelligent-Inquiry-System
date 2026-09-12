"""
utils/llm_client.py
===================
统一大模型客户端封装。

功能：
  1. 封装 Qwen3.5 文本模型调用接口
  2. 封装 Qwen-VL 多模态模型调用接口
  3. 支持异步调用
  4. 统一错误处理和重试机制
  5. 提供 Function Calling 便捷方法
"""

from __future__ import annotations
import json
import base64
from typing import Optional
from pathlib import Path
from loguru import logger
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from configs.settings import settings


class LLMClient:
    """
    统一大模型客户端。
    封装 Qwen3.5 文本模型和 Qwen-VL 多模态模型的调用。
    """

    def __init__(self):
        self._client = AsyncOpenAI(
            api_key=settings.QWEN_API_KEY,
            base_url=settings.QWEN_BASE_URL,
        )
        self._text_model = settings.QWEN_MODEL_NAME
        self._vl_model = settings.QWEN_VL_MODEL_NAME

    # ============================================================
    # 文本模型调用
    # ============================================================

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def chat(
        self,
        messages: list[dict],
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        tools: list[dict] = None,
        tool_choice: dict = None,
    ) -> str:
        """
        通用文本对话接口。

        :param messages: 消息列表 [{"role": "user", "content": "..."}]
        :param model: 模型名称（默认使用配置中的模型）
        :param temperature: 温度参数
        :param max_tokens: 最大输出 token 数
        :param tools: Function Calling 工具列表
        :param tool_choice: 工具选择策略
        :return: 模型回复文本
        """
        model = model or self._text_model
        temperature = temperature if temperature is not None else settings.QWEN_TEMPERATURE
        max_tokens = max_tokens or settings.QWEN_MAX_TOKENS

        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        response = await self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_choice: dict = None,
        model: str = None,
        temperature: float = 0.0,
    ) -> dict:
        """
        带 Function Calling 的对话接口。

        :return: 解析后的工具调用参数 dict
        """
        model = model or self._text_model

        if tool_choice is None and tools:
            tool_choice = {
                "type": "function",
                "function": {"name": tools[0]["function"]["name"]},
            }

        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
        )

        try:
            tool_call = response.choices[0].message.tool_calls[0]
            args = tool_call["args"] if isinstance(tool_call["args"], dict) else json.loads(tool_call["args"])
            return args
        except (AttributeError, IndexError, KeyError, json.JSONDecodeError) as e:
            logger.warning(f"[LLMClient] Function Calling 解析失败: {e}")
            return {}

    async def simple_chat(self, user_message: str, system_prompt: str = "") -> str:
        """
        简单对话接口（单轮）。

        :param user_message: 用户消息
        :param system_prompt: 系统提示
        :return: 模型回复
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        return await self.chat(messages)

    # ============================================================
    # 多模态模型调用
    # ============================================================

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def vision_chat(
        self,
        prompt: str,
        image_data: bytes = None,
        image_path: str = None,
        model: str = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """
        多模态对话接口（图文混合）。

        :param prompt: 文本提示
        :param image_data: 图片二进制数据
        :param image_path: 图片文件路径
        :param model: 模型名称
        :return: 模型回复文本
        """
        model = model or self._vl_model
        image_url = self._prepare_image_url(image_data, image_path)

        response = await self._client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content

    # ============================================================
    # 内部辅助方法
    # ============================================================

    def _prepare_image_url(
        self, image_data: bytes = None, image_path: str = None
    ) -> str:
        """将图片转为 base64 data URL"""
        if image_data:
            b64 = base64.b64encode(image_data).decode("utf-8")
            return f"data:image/png;base64,{b64}"

        elif image_path:
            path = Path(image_path)
            if path.exists():
                with open(path, "rb") as f:
                    data = f.read()
                b64 = base64.b64encode(data).decode("utf-8")
                suffix = path.suffix.lower()
                mime_map = {
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".png": "image/png",
                    ".bmp": "image/bmp",
                    ".webp": "image/webp",
                }
                mime = mime_map.get(suffix, "image/png")
                return f"data:{mime};base64,{b64}"
            else:
                raise FileNotFoundError(f"图片不存在: {image_path}")
        else:
            raise ValueError("必须提供 image_data 或 image_path")

    async def close(self):
        """关闭客户端"""
        await self._client.close()


# 全局单例
llm_client = LLMClient()


# ============================================================
# 单独调试入口
# ============================================================

if __name__ == "__main__":
    """单独测试 LLM 客户端"""
    import asyncio

    async def test_client():
        client = LLMClient()

        # 测试1：简单对话
        response = await client.simple_chat(
            user_message="什么是高血压？",
            system_prompt="你是一位医学科普专家，用简洁通俗的语言回答。",
        )
        print(f"简单对话: {response[:200]}")

        # 测试2：Function Calling
        result = await client.chat_with_tools(
            messages=[{"role": "user", "content": "我头疼三天了"}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "extract_symptoms",
                    "description": "提取症状",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symptoms": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                    },
                },
            }],
        )
        print(f"Function Calling: {result}")

        await client.close()

    asyncio.run(test_client())
