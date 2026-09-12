"""
agent/worker/qa_agent.py
========================
医学知识问答 Worker Agent —— 医学科普与知识检索。

职责：
  1. 接收用户的医学知识提问
  2. 调用 RAG 检索医学知识库（Milvus 三集合混合检索）
  3. 生成专业、通俗的医学科普回答
  4. 更新 state
"""

from __future__ import annotations
from loguru import logger
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from state.state_schema import InquiryState
from configs.settings import settings
from utils.message_utils import extract_latest_user_message


# 模块级别 LLM 单例（知识问答专用）
_qa_llm = None


def _get_qa_llm() -> ChatOpenAI:
    """获取知识问答专用 LLM 实例（模块级单例）"""
    global _qa_llm
    if _qa_llm is None:
        _qa_llm = ChatOpenAI(
            model=settings.QWEN_MODEL_NAME,
            api_key=settings.QWEN_API_KEY,
            base_url=settings.QWEN_BASE_URL,
            temperature=0.4,
            max_tokens=1500,
        )
    return _qa_llm


# ============================================================
# 知识问答 Worker 节点函数
# ============================================================

async def qa_worker_node(state: InquiryState) -> dict:
    """
    知识问答 Worker 处理函数。

    执行流程：
      1. 提取用户问题
      2. 检索医学知识库（RAG 混合检索）
      3. 结合检索结果，调用 LLM 生成科普回答
      4. 更新 state

    Args:
        state: 当前 InquiryState

    Returns:
        需要更新的 state 字段字典
    """
    logger.info("[知识问答Worker] 开始执行")

    # ---------- 1. 提取问题 ----------
    messages = state.get("messages", [])
    question = extract_latest_user_message(messages)
    logger.info(f"[知识问答Worker] 问题: {question}")

    # ---------- 2. RAG 检索 ----------
    context = await _retrieve_medical_knowledge(question)

    # ---------- 2.5 RAG 无结果时，联网搜索补充 ----------
    if not context:
        from utils.web_search import web_search
        context = await web_search(question, max_results=5)
        if context:
            logger.info("[知识问答Worker] RAG无结果，已启用联网搜索补充")

    # ---------- 3. LLM 生成科普回答 ----------
    answer = await _generate_qa_answer(question, context)
    logger.info(f"[知识问答Worker] 回答生成完成，长度={len(answer)}")

    # ---------- 4. 更新 state ----------
    return {
        "qa_question": question,
        "qa_answer": answer,
        "final_response": answer,
        "messages": [AIMessage(content=answer)],
    }


# ============================================================
# RAG 知识库检索（Milvus 三集合混合检索）
# ============================================================

async def _retrieve_medical_knowledge(question: str) -> str:
    """
    从医学知识库中检索相关上下文（RAG）。

    使用 Milvus 向量库检索 medical_knowledge / disease_collection /
    symptom_collection 三个集合，混合检索后拼接为上下文。

    Args:
        question: 用户问题

    Returns:
        检索到的上下文文本（无结果时返回空字符串）
    """
    try:
        from utils.rag_retrieval import rag_retriever
        context = await rag_retriever.hybrid_retrieve(question, top_k=5)
        if context:
            logger.info(f"[知识问答Worker] RAG检索到 {len(context)} 字符的上下文")
        else:
            logger.debug("[知识问答Worker] RAG检索无结果，将使用LLM内置知识")
        return context
    except Exception as e:
        logger.warning(f"[知识问答Worker] RAG检索失败: {e}，降级为LLM内置知识")
        return ""


# ============================================================
# LLM 生成科普回答
# ============================================================

async def _generate_qa_answer(question: str, context: str) -> str:
    """
    调用 LLM 生成医学科普回答。

    Args:
        question: 用户问题
        context: RAG 检索到的上下文（可为空）

    Returns:
        科普回答文本
    """
    llm = _get_qa_llm()

    # 构造系统提示
    if context:
        # 有 RAG 上下文时，要求 LLM 严格基于检索内容回答
        system_prompt = (
            "你是一位专业的医学科普作家。请严格基于以下参考资料回答用户问题。\n"
            "回答要求：\n"
            "1. 优先使用参考资料中的信息，不要编造参考资料中没有的数据或结论\n"
            "2. 语言通俗，非医学专业人士也能理解\n"
            "3. 结构清晰，使用小标题和列表\n"
            "4. 适当举例帮助理解\n"
            "5. 如涉及疾病或治疗，提醒读者咨询专业医生\n"
            "6. 如果参考资料不足以回答问题，可以结合你的医学知识补充，但要明确标注"
        )
        user_content = f"参考资料：\n{context}\n\n用户问题：{question}"
    else:
        # 无 RAG 上下文时，使用 LLM 内置知识
        system_prompt = (
            "你是一位专业的医学科普作家，擅长将复杂的医学知识用通俗易懂的语言解释。\n"
            "回答要求：\n"
            "1. 科学准确，引用权威医学资料\n"
            "2. 语言通俗，非医学专业人士也能理解\n"
            "3. 结构清晰，使用小标题和列表\n"
            "4. 适当举例帮助理解\n"
            "5. 如涉及疾病或治疗，提醒读者咨询专业医生\n"
            "6. 不要编造数据或研究结论"
        )
        user_content = question

    response = await llm.ainvoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ])

    answer = response.content.strip()
    return f"📚 **医学科普**\n\n{answer}\n\n💡 以上内容仅供科普参考，如有健康问题请咨询专业医生。"


# ============================================================
# 辅助函数
# ============================================================

# ============================================================
# 单独调试入口
# ============================================================

if __name__ == "__main__":
    """单独测试知识问答 Worker"""
    import asyncio
    from langchain_core.messages import HumanMessage
    from state.state_schema import make_initial_state

    async def test_qa():
        state = make_initial_state(user_id="test_001", session_id="sess_004")
        state["messages"] = [HumanMessage(content="高血压患者为什么不能吃葡萄柚？")]
        state["intent"] = "knowledge_qa"
        result = await qa_worker_node(state)
        print(f"科普回答:\n{result.get('final_response')}")

    asyncio.run(test_qa())
