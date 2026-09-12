"""
agent/worker/collect_agent.py
=============================
问诊采集 Worker Agent —— 核心问诊流程执行者。

职责：
  1. 读取 state 中用户消息，提取症状
  2. 调用三层症状标准化流水线（pipeline/symptom_normalize.py）
  3. 调用置信度诊断引擎（pipeline/diagnosis_engine.py）计算候选疾病
  4. 生成追问话术（如果未收敛）
  5. 更新 state：症状列表、候选疾病、轮次计数、追问消息
"""

from __future__ import annotations
import re
from loguru import logger
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from state.state_schema import InquiryState
from pipeline.symptom_normalize import SymptomNormalizePipeline
from pipeline.diagnosis_engine import DiagnosisEngine
from configs.settings import settings
from utils.message_utils import extract_latest_user_message


# 模块级别 LLM 单例（追问生成专用，避免每次调用都新建连接）
_follow_up_llm = None

# 简短回答模式：用户回答追问时不包含新症状信息
# 只匹配明确的肯定/否定/程度回答，不要误判有实际内容的短语
_SHORT_ANSWER_PATTERNS = re.compile(
    r'^(没有|没|有|有点|有一点|是|不是|对的|不对|好的|'
    r'不清楚|不知道|不确定|还好|还好吧|偶尔|经常|'
    r'一直|有时有|没感觉|不痛|不疼|疼|痛|'
    r'一点|一些|很多|严重|不严重|轻微|'
    r'前几天|昨天|今天|最近|一直有|'
    r'嗯|哦|啊|对|错|行|不行|可以|不可以)$'
)


def _is_short_answer(text: str) -> bool:
    """判断用户输入是否为追问的简短回答（不包含新症状信息）"""
    text = text.strip()
    # 只有非常短的纯肯定/否定回答才视为简短回答（≤ 4 个字）
    # 5 个字以上通常包含实际内容（如“蹲着上完厕所”、“吃了火锅烤肉”）
    if len(text) <= 4 and _SHORT_ANSWER_PATTERNS.match(text):
        return True
    return False


def _get_follow_up_llm() -> ChatOpenAI:
    """获取追问生成专用 LLM 实例（模块级单例）"""
    global _follow_up_llm
    if _follow_up_llm is None:
        _follow_up_llm = ChatOpenAI(
            model=settings.QWEN_MODEL_NAME,
            api_key=settings.QWEN_API_KEY,
            base_url=settings.QWEN_BASE_URL,
            temperature=0.5,
            max_tokens=200,
        )
    return _follow_up_llm


# ============================================================
# 问诊采集 Worker 节点函数
# ============================================================

async def collect_worker_node(state: InquiryState) -> dict:
    """
    问诊采集 Worker 处理函数。

    执行流程：
      1. 获取用户最新消息文本
      2. 调用三层症状标准化流水线
      3. 合并到已有症状列表（去重）
      4. 调用诊断收敛引擎，获取候选疾病
      5. 判断是否收敛：
         - 未收敛 → 生成追问问题
         - 已收敛 → 生成诊断结论
      6. 更新 state

    Args:
        state: 当前 InquiryState

    Returns:
        需要更新的 state 字段字典
    """
    logger.info("[问诊采集Worker] 开始执行")

    messages = state.get("messages", [])
    current_round = state.get("inquiry_round", 0)
    existing_symptoms = state.get("standardized_symptoms", [])
    existing_candidates = state.get("disease_candidates", [])
    asked_questions = state.get("asked_questions", [])

    # 日志：显示当前状态中的症状（验证跨轮次持久化）
    logger.info(
        f"[问诊采集Worker] 状态检查: 轮次={current_round}, "
        f"已有症状={[s.get('standard_name', '') for s in existing_symptoms]}, "
        f"已有候选={[c.get('disease_name', '') for c in existing_candidates]}"
    )

    # ---------- 1. 获取用户最新消息 ----------
    user_text = extract_latest_user_message(messages)
    logger.info(f"[问诊采集Worker] 用户输入: {user_text}")

    # ---------- 1.5 检测是否为简短回答 ----------
    is_short = _is_short_answer(user_text)
    if is_short:
        logger.info(f"[问诊采集Worker] 检测到简短回答: '{user_text}'，跳过症状提取，沿用已有症状")

    # ---------- 2. 三层症状标准化 ----------
    symptom_pipeline = SymptomNormalizePipeline()
    new_symptoms = await symptom_pipeline.run(user_text, is_short_answer=is_short)
    logger.info(
        f"[问诊采集Worker] 标准化症状: "
        f"{[s.get('standard_name', '') for s in new_symptoms]}"
    )

    # ---------- 3. 合并症状（去重） ----------
    all_symptoms = _merge_symptoms(existing_symptoms, new_symptoms)

    # ---------- 4. 诊断收敛引擎 ----------
    engine = DiagnosisEngine()
    candidates, converged, top_diagnosis = await engine.evaluate(
        symptoms=all_symptoms,
        current_round=current_round + 1,
    )

    # 合并候选疾病
    all_candidates = _merge_candidates(existing_candidates, candidates)

    # ---------- 5. 生成响应 ----------
    new_round = current_round + 1
    final_response = ""

    if converged:
        # 诊断收敛 → 输出结论
        final_response = _generate_diagnosis_conclusion(
            top_diagnosis, all_candidates, all_symptoms
        )
        logger.info(f"[问诊采集Worker] 诊断收敛: {top_diagnosis}")
    else:
        # 未收敛 → 生成追问（引入对话历史 + 已问问题历史）
        final_response = await _generate_follow_up(
            candidates=all_candidates,
            current_symptoms=all_symptoms,
            current_round=new_round,
            asked_questions=asked_questions,
            messages=messages,
        )
        logger.info(f"[问诊采集Worker] 追问: {final_response}")
        # 记录本次追问到历史（避免下一轮重复提问）
        asked_questions = asked_questions + [final_response]

    # ---------- 6. 返回更新 ----------
    return {
        "standardized_symptoms": all_symptoms,
        "disease_candidates": all_candidates,
        "diagnosis_confidence": (
            all_candidates[0].get("confidence", 0) if all_candidates else 0
        ),
        "inquiry_round": new_round,
        "diagnosis_converged": converged,
        "final_diagnosis": top_diagnosis or "",
        "final_response": final_response,
        "asked_questions": asked_questions,
        "messages": [AIMessage(content=final_response)],
    }


# ============================================================
# 辅助函数
# ============================================================


def _merge_symptoms(existing: list, new: list) -> list:
    """合并症状列表，按 standard_name 去重，保留高置信度"""
    seen = {}
    for s in existing + new:
        key = s.get("standard_name", "")
        if key:
            if key not in seen or s.get("confidence", 0) > seen[key].get("confidence", 0):
                seen[key] = s
    return sorted(seen.values(), key=lambda x: x.get("confidence", 0), reverse=True)


def _merge_candidates(existing: list, new: list) -> list:
    """合并候选疾病列表，按 disease_name 去重"""
    seen = {}
    for c in existing + new:
        key = c.get("disease_name", "")
        if key:
            if key not in seen or c.get("confidence", 0) > seen[key].get("confidence", 0):
                seen[key] = c
    return sorted(seen.values(), key=lambda x: x.get("confidence", 0), reverse=True)


def _generate_diagnosis_conclusion(
    diagnosis: str, candidates: list, symptoms: list
) -> str:
    """生成诊断结论（负责任的全科医生风格）"""
    symptom_names = [s.get("standard_name", "") for s in symptoms]
    top1 = candidates[0] if candidates else {}
    conf = top1.get("confidence", 0)

    text = f"根据您描述的症状（{'、'.join(symptom_names[:5])}），"
    text += f"经过分析，初步判断您可能患有【{diagnosis}】（置信度 {conf:.0%}）。"

    if len(candidates) > 1:
        others = [c["disease_name"] for c in candidates[1:3]]
        text += f"\n\n也需要鉴别：{'、'.join(others)}等疾病。"

    # 根据置信度给出不同的建议
    if conf >= 0.85:
        text += (
            "\n\n⚠️ 以上为AI辅助分析建议，不能替代专业医生诊断。"
            "建议您尽快到医院做进一步检查。"
        )
    elif conf >= 0.60:
        text += (
            "\n\n⚠️ 以上为AI辅助分析建议，置信度中等，"
            "建议您到医院做进一步检查以明确诊断。"
        )
    else:
        text += (
            "\n\n⚠️ 以上为AI辅助分析建议，置信度较低，"
            "可能还需要更多信息才能准确判断。"
            "建议您到医院就诊，由专业医生综合评估。"
        )
    return text


async def _generate_follow_up(
    candidates: list,
    current_symptoms: list,
    current_round: int,
    asked_questions: list[str] = None,
    messages: list = None,
) -> str:
    """
    使用 LLM 生成追问话术。
    基于当前候选疾病的鉴别诊断需要，结合对话历史和已问问题，
    生成最有助于收敛的问题。
    关键原则：先问常见病因，再问罕见病；不要过度检查。
    """
    llm = _get_follow_up_llm()

    top_diseases = [c.get("disease_name", "") for c in candidates[:3]]
    symptom_names = [s.get("standard_name", "") for s in current_symptoms]

    # --- 提取对话历史中的关键信息（原始主诉 + 已回答内容） ---
    conversation_summary = _extract_conversation_context(messages or [])

    # --- RAG 检索：获取与当前症状相关的鉴别要点 ---
    rag_context = ""
    try:
        from utils.rag_retrieval import rag_retriever
        rag_context = await rag_retriever.retrieve_knowledge(
            " ".join(symptom_names), top_k=3
        )
        if rag_context:
            rag_text = "\n".join([
                f"- {item.get('title', '')}: {item.get('content', '')[:150]}"
                for item in rag_context
            ])
            logger.debug(f"[问诊采集Worker] RAG检索到 {len(rag_context)} 条鉴别知识")
        else:
            rag_text = ""
    except Exception as e:
        logger.debug(f"[问诊采集Worker] RAG检索跳过: {e}")
        rag_text = ""

    # --- 使用跨轮次追踪的已问问题列表 ---
    asked_list = asked_questions or []
    asked_text = "\n".join([f"- {q}" for q in asked_list[-8:]]) if asked_list else "无"

    # --- 构造 prompt ---
    system_prompt = (
        "你是一位经验丰富的全科医生，正在进行在线问诊。\n"
        "根据当前收集的症状、候选疾病和参考资料，生成一个最有助于鉴别诊断的追问。\n\n"
        "⚠️ 核心原则（必须遵守）：\n"
        "1. 优先询问常见病因：如果症状可能由常见、轻微的原因引起（如饮食不当、感冒、过敏），"
        "应先询问这些常见原因的相关症状，而不是直接询问罕见重症的表现\n"
        "2. 禁止过度检查：不要询问与当前症状关联性低的全身症状\n"
        "3. 问题必须与用户的主诉直接相关，不要跳跃式问诊\n"
        "4. 问题简洁通俗，每次只问一个方面\n"
        "5. 不要重复已问过的问题\n"
        "6. 直接输出问题，不要加前缀解释\n\n"
        "❌ 错误示例：用户说“拉肚子”，就问“是否出现皮肤发黄、体重下降”（跳跃到肝胆/肿瘤方向）\n"
        "✅ 正确示例：用户说“拉肚子”，先问“大便是什么性状？水样便还是糊状便？一天拉几次？”（常见消化问题）"
    )
    if rag_text:
        system_prompt += f"\n\n参考鉴别要点：\n{rag_text}"

    user_content = (
        f"用户原始主诉：{conversation_summary.get('original_complaint', '未知')}\n"
        f"当前已收集症状：{symptom_names}\n"
        f"候选疾病：{top_diseases}\n"
        f"当前追问轮数：{current_round}\n"
        f"已问过的问题：\n{asked_text}\n"
    )
    if conversation_summary.get('answers'):
        user_content += f"用户已回答的内容：\n{conversation_summary['answers']}\n"
    user_content += f"请生成下一个追问问题："

    response = await llm.ainvoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ])

    question = response.content.strip()
    return f"（第{current_round}轮追问）{question}"


def _extract_conversation_context(messages: list) -> dict:
    """
    从对话历史中提取关键上下文：
    - original_complaint: 用户最初的主诉
    - answers: 用户对追问的回答摘要
    """
    original_complaint = ""
    answers = []

    for msg in messages:
        if isinstance(msg, HumanMessage) and msg.content:
            content = msg.content.strip()
            if not original_complaint:
                original_complaint = content
            else:
                # 后续的用户消息视为对追问的回答
                answers.append(f"- 用户回答：“{content}”")

    return {
        "original_complaint": original_complaint,
        "answers": "\n".join(answers) if answers else "",
    }


def _extract_asked_questions(messages: list) -> list[str]:
    """从对话历史中提取已问过的问题（AIMessage 中包含'追问'的内容）"""
    from langchain_core.messages import AIMessage
    questions = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.content:
            content = msg.content.strip()
            # 提取追问轮次标记后的问题文本
            if "轮追问）" in content:
                # 去掉前缀标记，保留问题内容
                idx = content.find("轮追问）")
                if idx >= 0:
                    q = content[idx + len("轮追问）"):].strip()
                    if q:
                        questions.append(q)
    return questions


# ============================================================
# 单独调试入口
# ============================================================

if __name__ == "__main__":
    """单独测试问诊采集 Worker"""
    import asyncio
    from langchain_core.messages import HumanMessage
    from state.state_schema import make_initial_state

    async def test_collect():
        state = make_initial_state(user_id="test_001", session_id="sess_001")
        state["messages"] = [HumanMessage(content="我头疼三天了，还有点发烧，嗓子也疼")]
        state["intent"] = "consultation"
        result = await collect_worker_node(state)
        print(f"症状: {result.get('standardized_symptoms')}")
        print(f"候选: {result.get('disease_candidates')}")
        print(f"追问: {result.get('final_response')}")

    asyncio.run(test_collect())
