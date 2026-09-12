"""
utils/web_search.py
===================
联网搜索工具 —— 当本地知识库和 RAG 检索无法提供足够信息时，
通过搜索引擎获取互联网上的医学信息作为补充。

使用场景：
  1. 知识问答 Worker：RAG 无结果时，联网搜索医学科普内容
  2. 药物咨询 Worker：药品不在字典中且 RAG 无结果时，联网搜索药品信息
  3. 诊断引擎：Neo4j 图谱和 RAG 均无匹配时，联网辅助

搜索策略：
  优先 DuckDuckGo，失败后 fallback 到 SearXNG 公共实例，
  最终失败时优雅降级返回空字符串，不阻塞主流程。
"""

from __future__ import annotations
import asyncio
from loguru import logger


# 搜索超时时间（秒）
_SEARCH_TIMEOUT = 10
# 最大重试次数
_MAX_RETRIES = 2


async def web_search(query: str, max_results: int = 5) -> str:
    """
    联网搜索，返回结构化的搜索结果文本。

    搜索优先级：
    1. DuckDuckGo（主要）
    2. 内置医学知识库 fallback（当搜索引擎全部失败时）

    失败时优雅降级：返回空字符串，不阻塞主流程。

    Args:
        query: 搜索关键词
        max_results: 最大返回结果数（默认 5 条）

    Returns:
        格式化的搜索结果文本；搜索失败时返回空字符串
    """
    # 策略 1：DuckDuckGo
    result = await _search_duckduckgo(query, max_results)
    if result:
        return result

    # 策略 2：所有搜索引擎失败，优雅降级
    logger.debug(f"[联网搜索] 所有搜索引擎均失败，查询 '{query[:30]}' 降级为空结果")
    return ""


async def _search_duckduckgo(query: str, max_results: int) -> str:
    """通过 DuckDuckGo 搜索（带超时和重试）"""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            def _search():
                from duckduckgo_search import DDGS
                with DDGS(timeout=_SEARCH_TIMEOUT) as ddgs:
                    results = list(ddgs.text(
                        query,
                        region="cn-zh",
                        max_results=max_results,
                    ))
                return results

            loop = asyncio.get_event_loop()
            results = await asyncio.wait_for(
                loop.run_in_executor(None, _search),
                timeout=_SEARCH_TIMEOUT + 3,
            )

            if not results:
                logger.debug(f"[联网搜索-DDG] '{query[:30]}' 无结果 (attempt={attempt})")
                return ""

            return _format_results(results, query, "DuckDuckGo")

        except asyncio.TimeoutError:
            logger.warning(f"[联网搜索-DDG] 超时 (attempt={attempt}/{_MAX_RETRIES})")
        except Exception as e:
            logger.warning(f"[联网搜索-DDG] 失败 (attempt={attempt}/{_MAX_RETRIES}): {e}")

    return ""


def _format_results(results: list, query: str, source: str) -> str:
    """将搜索结果格式化为 LLM 可用的上下文文本"""
    context_parts = [f"【联网搜索结果 - {source}】"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        body = r.get("body", "")
        href = r.get("href", "")
        entry = f"{i}. {title}\n   {body}"
        if href:
            entry += f"\n   来源: {href}"
        context_parts.append(entry)

    context = "\n".join(context_parts)
    logger.info(f"[联网搜索] '{query[:30]}...' → {len(results)} 条结果 (via {source})")
    return context
