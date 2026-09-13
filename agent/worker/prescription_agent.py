"""
agent/worker/prescription_agent.py
===================================
处方审核 Worker Agent —— 处方安全性校验。

职责：
  1. 接收处方信息（药品列表、剂量、频次）
  2. 校验药品冲突（药物相互作用）
  3. 校验禁忌症（过敏史、基础疾病）
  4. 校验剂量风险（超量、不足量）
  5. 输出审核结果与风险提示，更新 state
"""

from __future__ import annotations
import json
import re
from loguru import logger
from langchain_core.messages import AIMessage
from utils.fallback_llm import FallbackChatOpenAI as ChatOpenAI

from state.state_schema import InquiryState
from configs.settings import settings
from utils.message_utils import extract_latest_user_message


# 模块级别 LLM 单例（处方审核专用，两个不同配置）
_prescription_parse_llm = None
_prescription_review_llm = None


def _get_prescription_parse_llm() -> ChatOpenAI:
    """获取处方解析专用 LLM（temperature=0.0，确定性输出）"""
    global _prescription_parse_llm
    if _prescription_parse_llm is None:
        _prescription_parse_llm = ChatOpenAI(
            temperature=0.0,
        )
    return _prescription_parse_llm


def _get_prescription_review_llm() -> ChatOpenAI:
    """获取处方审核专用 LLM（temperature=0.2，max_tokens=1024）"""
    global _prescription_review_llm
    if _prescription_review_llm is None:
        _prescription_review_llm = ChatOpenAI(
            temperature=0.2,
            max_tokens=1024,
        )
    return _prescription_review_llm


# ============================================================
# 药品相互作用规则库（内置基础规则，后续可扩展为图谱查询）
# ============================================================

# 禁忌组合：(药A, 药B) → 风险描述
DRUG_INTERACTION_RULES: list[dict] = [
    {
        "drug_a": "阿莫西林",
        "drug_b": "甲氨蝶呤",
        "risk_level": "high",
        "description": "阿莫西林可减少甲氨蝶呤排泄，增加毒性风险",
    },
    {
        "drug_a": "布洛芬",
        "drug_b": "阿司匹林",
        "risk_level": "medium",
        "description": "布洛芬可能降低阿司匹林的心血管保护作用",
    },
    {
        "drug_a": "布洛芬",
        "drug_b": "华法林",
        "risk_level": "high",
        "description": "NSAIDs与抗凝药合用增加消化道出血风险",
    },
    {
        "drug_a": "二甲双胍",
        "drug_b": "碘造影剂",
        "risk_level": "critical",
        "description": "合用增加乳酸酸中毒风险，造影前后48h需停用二甲双胍",
    },
    {
        "drug_a": "头孢克洛",
        "drug_b": "氨基糖苷类",
        "risk_level": "medium",
        "description": "合用增加肾毒性风险",
    },
    {
        "drug_a": "奥美拉唑",
        "drug_b": "氯吡格雷",
        "risk_level": "high",
        "description": "奥美拉唑抑制CYP2C19，降低氯吡格雷活性代谢产物",
    },
]

# 药品剂量上限（单次最大剂量，单位mg）
MAX_DOSAGE_RULES: dict[str, float] = {
    "布洛芬": 400.0,
    "阿莫西林": 1000.0,
    "头孢克洛": 500.0,
    "奥美拉唑": 40.0,
    "二甲双胍": 1000.0,
}


# ============================================================
# 处方审核 Worker 节点函数
# ============================================================

async def prescription_worker_node(state: InquiryState) -> dict:
    """
    处方审核 Worker 处理函数。

    执行流程：
      1. 从 state 中提取处方信息
      2. 解析处方中的药品列表
      3. 逐项检查：药物相互作用、禁忌症、剂量风险
      4. 使用 LLM 生成审核报告
      5. 更新 state

    Args:
        state: 当前 InquiryState

    Returns:
        需要更新的 state 字段字典
    """
    logger.info("[处方审核Worker] 开始执行")

    # ---------- 1. 提取处方信息 ----------
    messages = state.get("messages", [])
    prescription_text = extract_latest_user_message(messages)
    prescription_items = state.get("prescription_items", [])

    logger.info(f"[处方审核Worker] 处方内容: {prescription_text[:200]}")

    # ---------- 2. 解析处方药品 ----------
    if not prescription_items:
        prescription_items = await _parse_prescription(prescription_text)

    drug_names = [item.get("drug_name", "") for item in prescription_items]
    logger.info(f"[处方审核Worker] 药品列表: {drug_names}")

    # ---------- 3. 规则校验 ----------
    issues = []
    suggestions = []
    risk_level = "low"

    # 3.1 药物相互作用检查
    interaction_issues = _check_drug_interactions(drug_names)
    issues.extend(interaction_issues)

    # 3.2 剂量检查
    dosage_issues = _check_dosage(prescription_items)
    issues.extend(dosage_issues)

    # 3.3 使用 LLM 做进一步审核
    llm_review = await _llm_prescription_review(prescription_items, prescription_text)
    if llm_review.get("additional_issues"):
        # 确保 additional_issues 中的每一项都是字典
        for item in llm_review["additional_issues"]:
            if isinstance(item, dict):
                issues.append(item)
            else:
                # 如果是字符串，转换为字典
                issues.append({"description": str(item)[:200], "risk_level": "medium"})
    suggestions.extend(llm_review.get("suggestions", []))

    # 确定最终风险等级
    for issue in issues:
        if issue.get("risk_level") == "critical":
            risk_level = "critical"
            break
        elif issue.get("risk_level") == "high" and risk_level != "critical":
            risk_level = "high"
        elif issue.get("risk_level") == "medium" and risk_level == "low":
            risk_level = "medium"

    is_safe = len(issues) == 0

    # ---------- 4. 构造审核结果 ----------
    review_result = {
        "is_safe": is_safe,
        "risk_level": risk_level,
        "issues": [issue.get("description", str(issue)) for issue in issues],
        "suggestions": suggestions,
    }

    # ---------- 5. 生成面向用户的响应 ----------
    response = _format_review_response(review_result, prescription_items)

    return {
        "prescription_items": prescription_items,
        "review_result": review_result,
        "final_response": response,
        "messages": [AIMessage(content=response)],
    }


# ============================================================
# 规则校验函数
# ============================================================

def _check_drug_interactions(drug_names: list[str]) -> list[dict]:
    """检查药物相互作用"""
    issues = []
    for rule in DRUG_INTERACTION_RULES:
        if rule["drug_a"] in drug_names and rule["drug_b"] in drug_names:
            issues.append({
                "type": "interaction",
                "risk_level": rule["risk_level"],
                "description": (
                    f"⚠️ 【{rule['risk_level'].upper()}风险】"
                    f"{rule['drug_a']} + {rule['drug_b']}：{rule['description']}"
                ),
            })
    return issues


def _check_dosage(prescription_items: list[dict]) -> list[dict]:
    """检查剂量是否超标"""
    issues = []
    for item in prescription_items:
        drug_name = item.get("drug_name", "")
        dosage_str = item.get("dosage", "")
        if drug_name in MAX_DOSAGE_RULES and dosage_str:
            # 简单提取数值
            try:
                numbers = re.findall(r"(\d+\.?\d*)", dosage_str)
                if numbers:
                    dosage_mg = float(numbers[0])
                    # 假设提取的是mg单位的单次剂量
                    max_mg = MAX_DOSAGE_RULES[drug_name]
                    if dosage_mg > max_mg:
                        issues.append({
                            "type": "dosage",
                            "risk_level": "high",
                            "description": (
                                f"⚠️ 【HIGH风险】{drug_name} 单次剂量 {dosage_mg}mg "
                                f"超过推荐最大单次剂量 {max_mg}mg"
                            ),
                        })
            except (ValueError, IndexError):
                pass
    return issues


# ============================================================
# LLM 辅助审核
# ============================================================

async def _parse_prescription(text: str) -> list[dict]:
    """使用 LLM 从文本中解析处方信息"""
    llm = _get_prescription_parse_llm()

    response = await llm.ainvoke([
        {
            "role": "system",
            "content": (
                "从处方文本中提取药品信息，使用 parse_prescription 工具输出。\n"
                "每个药品包括：drug_name（药品名）、dosage（单次剂量）、"
                "frequency（频次）、route（给药途径）"
            ),
        },
        {"role": "user", "content": text},
    ])

    # 尝试解析结构化结果
    try:
        tool_call = response.tool_calls[0]
        args = tool_call["args"] if isinstance(tool_call["args"], dict) else json.loads(tool_call["args"])
        return args.get("items", [])
    except (AttributeError, IndexError, KeyError, json.JSONDecodeError):
        # 降级：简单正则提取药品名
        items = []
        for drug_name in MAX_DOSAGE_RULES.keys():
            if drug_name in text:
                items.append({"drug_name": drug_name, "dosage": "", "frequency": "", "route": "口服"})
        return items


async def _llm_prescription_review(
    prescription_items: list[dict], original_text: str
) -> dict:
    """使用 LLM 做进一步处方审核"""
    llm = _get_prescription_review_llm()

    items_text = "\n".join(
        [f"- {item.get('drug_name', '')}: {item.get('dosage', '')} {item.get('frequency', '')}"
         for item in prescription_items]
    )

    response = await llm.ainvoke([
        {
            "role": "system",
            "content": (
                "你是一位资深临床药师，负责审核处方安全性。\n"
                "请检查以下处方的合理性，包括：\n"
                "1. 药物相互作用\n"
                "2. 剂量合理性\n"
                "3. 给药途径合理性\n"
                "4. 其他潜在风险\n"
                "输出 JSON 格式：{\"additional_issues\": [...], \"suggestions\": [...]}"
            ),
        },
        {"role": "user", "content": f"处方内容：\n{items_text}\n\n原始处方：{original_text}"},
    ])

    try:
        # 尝试提取 JSON（可能包含在 markdown 代码块中）
        text = response.content
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(text)
        return result
    except (json.JSONDecodeError, AttributeError) as e:
        logger.warning(f"[处方审核] LLM返回JSON解析失败: {e}")
        return {
            "additional_issues": [{"description": str(response.content)[:200], "risk_level": "medium"}],
            "suggestions": ["建议咨询临床药师进一步审核"],
        }


# ============================================================
# 格式化输出
# ============================================================

def _format_review_response(review_result: dict, prescription_items: list) -> str:
    """格式化处方审核结果"""
    is_safe = review_result.get("is_safe", True)
    risk_level = review_result.get("risk_level", "low")
    issues = review_result.get("issues", [])
    suggestions = review_result.get("suggestions", [])

    drug_names = [item.get("drug_name", "") for item in prescription_items]

    # 风险等级图标
    risk_icons = {
        "low": "✅",
        "medium": "⚡",
        "high": "⚠️",
        "critical": "🚨",
    }
    icon = risk_icons.get(risk_level, "✅")

    response = f"📋 **处方审核报告** {icon}\n\n"
    response += f"**药品列表**：{'、'.join(drug_names)}\n\n"

    if is_safe:
        response += "**审核结果**：处方审核通过，未发现明显安全风险。\n"
    else:
        response += f"**风险等级**：{risk_level.upper()}\n\n"
        response += "**发现的问题**：\n"
        for issue in issues:
            response += f"  • {issue}\n"

    if suggestions:
        response += "\n**修改建议**：\n"
        for sug in suggestions:
            response += f"  • {sug}\n"

    response += "\n⚠️ 以上为AI辅助审核结果，最终处方请以临床医师/药师意见为准。"
    return response


# ============================================================
# 辅助函数
# ============================================================

# ============================================================
# 单独调试入口
# ============================================================

if __name__ == "__main__":
    """单独测试处方审核 Worker"""
    import asyncio
    from langchain_core.messages import HumanMessage
    from state.state_schema import make_initial_state

    async def test_prescription():
        state = make_initial_state(user_id="test_001", session_id="sess_005")
        state["messages"] = [HumanMessage(content=(
            "处方审核：阿莫西林 0.5g 每日3次口服，"
            "布洛芬 0.4g 每日3次口服，"
            "奥美拉唑 20mg 每日1次口服"
        ))]
        state["intent"] = "prescription_review"
        result = await prescription_worker_node(state)
        print(f"审核结果:\n{result.get('final_response')}")
        print(f"风险等级: {result.get('review_result', {}).get('risk_level')}")

    asyncio.run(test_prescription())
