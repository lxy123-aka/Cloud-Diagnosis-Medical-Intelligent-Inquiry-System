"""
agent/supervisor_agent.py
=========================
Supervisor 调度 Agent —— 多智能体系统的“大脑”。

职责：
  1. 调用 Qwen3.5 做意图识别（Function Calling 确定性分类）
  2. 控制问诊轮次上限（最多 10 轮）
  3. 读取诊断置信度，执行收敛规则：
     - 双条件收敛：Top1 ≥ 70% 或 Top1-Top2 差值 ≥ 30%
     - 最小追问轮数门控（至少 3 轮）+ 最大轮数强制收敛（10 轮兜底）
     满足收敛条件 → 终止问诊
  4. 更新 state，输出下一步 worker 名称
"""

from __future__ import annotations
import json
from loguru import logger
from langchain_core.messages import AIMessage, SystemMessage
from utils.fallback_llm import FallbackChatOpenAI as ChatOpenAI

from state.state_schema import InquiryState
from configs.settings import settings


# ============================================================
# 模块级别 LLM 单例（避免每次节点调用都新建连接）
# ============================================================

_supervisor_llm = None


def _get_llm() -> ChatOpenAI:
    """获取 Supervisor 专用 LLM 实例（模块级单例）"""
    global _supervisor_llm
    if _supervisor_llm is None:
        _supervisor_llm = ChatOpenAI(
            temperature=0.0,  # 确定性输出
        )
    return _supervisor_llm


# ============================================================
# 意图识别 System Prompt
# ============================================================

SUPERVISOR_SYSTEM_PROMPT = """你是云诊医疗智能问诊系统的 Supervisor 调度中心。
你的任务是：
1. 分析用户消息，判断意图类别
2. 如果是问诊场景，评估当前诊断是否已经收敛

意图类别（必须选择一个）：
- consultation：在线问诊，用户描述症状、身体不适（如“我头疼”、“肚子疼”、“最近总是失眠”）
- report_analysis：报告解读，用户上传/提及检查报告、CT/B超影像、化验单
- drug_inquiry：药物咨询，用户询问具体药物的用法、剂量、副作用、禁忌，或询问“该吃什么药”（如“布洛芬怎么吃”、“阿莫西林有什么副作用”、“拉肚子了该吃什么药”、“感冒了吃什么好”）
- knowledge_qa：知识问答，用户询问医学科普、疾病原理、健康常识（如“什么是高血压”、“为什么感冒不能喝酒”、“糖尿病饮食注意事项”）
- prescription_review：处方审核，用户提交处方需要审核（通常包含多个药品名称+剂量）
- chitchat：闲聊、问候等

⚠️ 重要区分规则：
- 如果用户询问“该吃什么药”、“用什么药好”、“吃什么药管用” → drug_inquiry（询问用药建议）
- 如果用户询问的是“某种具体药物”的用法/副作用 → drug_inquiry（提到具体药物名）
- 如果用户询问的是“疾病/健康知识的科普解释” → knowledge_qa（询问疾病饮食知识）
- 如果用户只是描述症状而没有问药 → consultation
- 例如：“拉肚子了，该吃什么药” → drug_inquiry（询问用药建议）
- 例如：“布洛芬有什么副作用？” → drug_inquiry（提到具体药物名）
- 例如：“我头疼” → consultation（只描述症状）
- 例如：“高血压患者为什么不能吃葡萄柚？” → knowledge_qa（询问疾病饮食知识）

你必须使用 classify_intent 工具输出分类结果。"""


# ============================================================
# Supervisor 节点函数
# ============================================================

async def supervisor_node(state: InquiryState) -> dict:
    """
    Supervisor 节点处理函数。

    执行流程：
      1. 获取最新消息，判断是否首次进入（需要做意图识别）
      2. 如果是从 Worker 返回的（问诊场景），检查收敛条件
      3. 更新 state 中的 intent / should_end / final_response

    Args:
        state: 当前 InquiryState

    Returns:
        需要更新的 state 字段字典
    """
    messages = state.get("messages", [])
    current_round = state.get("inquiry_round", 0)
    max_rounds = state.get("max_rounds", settings.MAX_INQUIRY_ROUNDS)
    current_intent = state.get("intent", "")

    logger.info(f"[Supervisor] 进入，当前轮次={current_round}，当前intent={current_intent}")

    # ---------- 获取 LLM 单例 ----------
    llm = _get_llm()

    # ---------- 判断是否是首次进入（Worker 还未执行） ----------
    # 使用 current_worker 判断：如果为空，说明 Worker 还没执行过，需要意图分类；
    # 如果不为空，说明是从 Worker 返回的，需要检查收敛。
    # 注意：不能用 intent 判断，因为在同一轮 chat() 内的图循环中，
    # supervisor 第一次执行会设置 intent，导致第二次进入时误判为"Worker已返回"。
    current_worker = state.get("current_worker", "")
    is_first_entry = current_worker == ""

    if is_first_entry:
        # ====== 首次进入：做意图识别 ======
        result = await _classify_intent(llm, messages)
        intent = result["intent"]
        confidence = result.get("confidence", 0.8)

        logger.info(f"[Supervisor] 意图识别结果: {intent} (置信度={confidence})")

        # 闲聊直接结束
        should_end = intent == "chitchat"
        final_response = ""
        if should_end:
            final_response = (
                "您好！我是云诊医疗智能问诊系统，可以为您提供在线问诊、"
                "报告解读、药物咨询等服务。请问有什么可以帮您？"
            )

        return {
            "intent": intent,
            "intent_confidence": confidence,
            "current_worker": _intent_to_worker(intent),
            "should_end": should_end,
            "final_response": final_response,
            "messages": [AIMessage(content=final_response)] if final_response else [],
        }

    else:
        # ====== Worker 返回后：检查是否继续 ======
        # 对于问诊场景，检查收敛条件
        if current_intent == "consultation":
            return _check_convergence(state)
        else:
            # 非问诊场景，Worker 执行一次即可结束
            logger.info(f"[Supervisor] {current_intent} 任务完成，结束对话")
            return {
                "should_end": True,
                "current_worker": "",
            }


# ============================================================
# 意图识别（Function Calling）
# ============================================================

async def _classify_intent(llm: ChatOpenAI, messages: list) -> dict:
    """
    使用 LLM Function Calling 做确定性意图分类。

    Returns:
        {"intent": str, "confidence": float}
    """
    # 定义 Function Calling 工具
    tools = [
        {
            "type": "function",
            "function": {
                "name": "classify_intent",
                "description": "对用户消息进行意图分类",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "intent": {
                            "type": "string",
                            "enum": [
                                "consultation",
                                "report_analysis",
                                "drug_inquiry",
                                "knowledge_qa",
                                "prescription_review",
                                "chitchat",
                            ],
                            "description": "意图类别",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "分类置信度 0-1",
                        },
                    },
                    "required": ["intent", "confidence"],
                },
            },
        }
    ]

    # 绑定工具并调用
    llm_with_tools = llm.bind_tools(
        tools, tool_choice={"type": "function", "function": {"name": "classify_intent"}}
    )

    # 构造消息列表（只取最近5条避免超长）
    chat_messages = [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)] + messages[-5:]

    response = await llm_with_tools.ainvoke(chat_messages)

    # 解析 Function Calling 结果
    try:
        tool_call = response.tool_calls[0]
        args = tool_call["args"] if isinstance(tool_call["args"], dict) else json.loads(tool_call["args"])
        return {
            "intent": args["intent"],
            "confidence": float(args.get("confidence", 0.8)),
        }
    except (IndexError, KeyError, json.JSONDecodeError, TypeError) as e:
        logger.warning(f"[Supervisor] 意图解析失败: {e}，降级为 chitchat")
        return {"intent": "chitchat", "confidence": 0.3}


# ============================================================
# 收敛判断
# ============================================================

def _check_convergence(state: InquiryState) -> dict:
    """
    检查诊断收敛条件。

    收敛规则：双条件收敛 + 最小轮数门控 + 最大轮数兜底
      双条件（满足任一即终止）：
        1. Top1 置信度 ≥ CONFIDENCE_THRESHOLD（默认 70%）
        2. Top1 - Top2 差值 ≥ CONFIDENCE_GAP（默认 30%）
      过程控制：
        - 当前轮数 < 最小追问轮数 → 强制继续
        - 达到最大追问轮数 → 强制收敛

    Returns:
        需要更新的 state 字段
    """
    candidates = state.get("disease_candidates", [])
    current_round = state.get("inquiry_round", 0)
    max_rounds = state.get("max_rounds", settings.MAX_INQUIRY_ROUNDS)
    final_response = state.get("final_response", "")

    # 按置信度排序
    sorted_candidates = sorted(
        candidates, key=lambda x: x.get("confidence", 0), reverse=True
    )

    # ---- 条件3：达到最大轮数，强制收敛 ----
    if current_round >= max_rounds:
        logger.warning(f"[Supervisor] 达到最大轮数 {max_rounds}，强制收敛")
        top1_name = sorted_candidates[0]["disease_name"] if sorted_candidates else "未知"
        return {
            "diagnosis_converged": True,
            "should_end": True,
            "final_diagnosis": top1_name,
            "final_response": final_response or (
                f"经过{max_rounds}轮问诊，初步判断您可能患有【{top1_name}】。"
                f"建议尽快到医院做进一步检查确认。"
            ),
        }

    if not sorted_candidates:
        # 还没有候选疾病，退出图循环等待用户回答
        return {"should_end": True}

    top1 = sorted_candidates[0]
    top1_conf = top1.get("confidence", 0)
    top2_conf = sorted_candidates[1].get("confidence", 0) if len(sorted_candidates) > 1 else 0
    gap = top1_conf - top2_conf

    logger.info(
        f"[Supervisor] 收敛检查: Top1={top1['disease_name']}({top1_conf:.2f}), "
        f"Top2_conf={top2_conf:.2f}, 差值={gap:.2f}, 轮次={current_round}/{max_rounds}"
    )

    # ---- 条件1：Top1 置信度 ≥ 阈值（默认 70%） ----
    if top1_conf >= settings.CONFIDENCE_THRESHOLD:
        logger.info(f"[Supervisor] Top1 置信度 {top1_conf:.2f} ≥ {settings.CONFIDENCE_THRESHOLD}，诊断收敛")
        return {
            "diagnosis_converged": True,
            "should_end": True,
            "final_diagnosis": top1["disease_name"],
            "diagnosis_confidence": top1_conf,
            "final_response": final_response or (
                f"根据您描述的症状，初步判断您可能患有【{top1['disease_name']}】"
                f"（置信度 {top1_conf:.0%}）。建议前往{_get_department(top1)}就诊，"
                f"并做进一步检查确认。"
            ),
        }

    # ---- 条件2：Top1 - Top2 差值 ≥ 30% ----
    if gap >= settings.CONFIDENCE_GAP:
        logger.info(f"[Supervisor] Top1-Top2 差值 {gap:.2f} ≥ {settings.CONFIDENCE_GAP}，诊断收敛")
        return {
            "diagnosis_converged": True,
            "should_end": True,
            "final_diagnosis": top1["disease_name"],
            "diagnosis_confidence": top1_conf,
            "final_response": final_response or (
                f"根据您描述的症状，初步判断您可能患有【{top1['disease_name']}】"
                f"（置信度 {top1_conf:.0%}）。建议前往{_get_department(top1)}就诊。"
            ),
        }

    # ---- 未收敛：退出图循环，将追问返回给用户 ----
    # 设置 should_end=True 让图停止循环，collect_worker 生成的追问
    # 已通过 final_response 保存在 state 中，会被返回给用户。
    # 用户回答后，下一次 chat() 调用会重新启动图。
    logger.info(f"[Supervisor] 未收敛，退出图循环等待用户回答（已完成第{current_round}轮）")
    return {"should_end": True}


# ============================================================
# 辅助函数
# ============================================================

def _intent_to_worker(intent: str) -> str:
    """意图 → Worker 名称映射"""
    mapping = {
        "consultation": "collect_worker",
        "report_analysis": "report_worker",
        "drug_inquiry": "drug_worker",
        "knowledge_qa": "qa_worker",
        "prescription_review": "prescription_worker",
    }
    return mapping.get(intent, "")


def _get_department(candidate: dict) -> str:
    """从候选疾病中获取推荐科室"""
    return candidate.get("department", "") or "全科"


# ============================================================
# 单独调试入口
# ============================================================

if __name__ == "__main__":
    """单独测试 Supervisor 意图识别"""
    import asyncio
    from langchain_core.messages import HumanMessage
    from state.state_schema import make_initial_state

    async def test_supervisor():
        # 测试1：问诊意图
        state = make_initial_state(user_id="test_001", session_id="sess_001")
        state["messages"] = [HumanMessage(content="我头疼三天了，还有点发烧")]
        result = await supervisor_node(state)
        print(f"意图识别结果: intent={result.get('intent')}, worker={result.get('current_worker')}")

    asyncio.run(test_supervisor())
