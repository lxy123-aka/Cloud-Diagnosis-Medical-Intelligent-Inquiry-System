"""
utils/message_utils.py
======================
Worker 共用的消息提取工具函数。

消除各 Worker 中重复的「从 messages 列表提取最新 HumanMessage」逻辑。
"""

from __future__ import annotations


def extract_latest_user_message(messages: list) -> str:
    """
    从 LangChain 消息列表中提取最新的用户消息文本。

    逆序遍历 messages，返回第一条 HumanMessage 的 content。
    所有 Worker（问诊采集 / 药物咨询 / 知识问答 / 报告解读 / 处方审核）
    统一使用此函数，避免各自重复实现相同逻辑。

    Args:
        messages: LangChain 消息列表（HumanMessage / AIMessage 等）

    Returns:
        最新用户消息文本；若无 HumanMessage 则返回空字符串
    """
    from langchain_core.messages import HumanMessage

    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""
