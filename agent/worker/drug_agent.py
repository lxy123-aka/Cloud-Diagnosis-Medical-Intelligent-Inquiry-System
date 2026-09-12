"""
agent/worker/drug_agent.py
==========================
药物咨询 Worker Agent —— 药品信息查询与用药建议。

职责：
  1. 解析用户药物咨询问题
  2. 查询药品信息（适应症、用法用量、禁忌、副作用、相互作用）
  3. 生成专业用药建议
  4. 更新 state
"""

from __future__ import annotations
from loguru import logger
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from state.state_schema import InquiryState
from configs.settings import settings
from utils.message_utils import extract_latest_user_message


# 模块级别 LLM 单例（药物咨询专用）
_drug_llm = None


def _get_drug_llm() -> ChatOpenAI:
    """获取药物咨询专用 LLM 实例（模块级单例）"""
    global _drug_llm
    if _drug_llm is None:
        _drug_llm = ChatOpenAI(
            model=settings.QWEN_MODEL_NAME,
            api_key=settings.QWEN_API_KEY,
            base_url=settings.QWEN_BASE_URL,
            temperature=0.3,
            max_tokens=1024,
        )
    return _drug_llm


# ============================================================
# 药物信息知识库（内置基础数据，后续可替换为 RAG 检索）
# ============================================================

DRUG_KNOWLEDGE_BASE: dict[str, dict] = {
    "阿莫西林": {
        "name": "阿莫西林",
        "trade_names": ["阿莫仙", "弗莱莫星"],
        "category": "青霉素类抗生素",
        "indications": ["上呼吸道感染", "中耳炎", "泌尿道感染", "皮肤软组织感染"],
        "dosage": "成人：0.5g/次，每8小时一次",
        "contraindications": ["青霉素过敏者禁用", "传染性单核细胞增多症患者禁用"],
        "side_effects": ["皮疹", "腹泻", "恶心", "过敏反应"],
        "interactions": ["与丙磺舒合用可升高血药浓度", "与甲氨蝶呤合用增加毒性"],
        "pregnancy_category": "B",
    },
    "布洛芬": {
        "name": "布洛芬",
        "trade_names": ["芬必得", "美林"],
        "category": "非甾体抗炎药(NSAIDs)",
        "indications": ["发热", "头痛", "关节痛", "痛经", "肌肉痛"],
        "dosage": "成人：0.2-0.4g/次，每4-6小时一次，日最大量1.2g",
        "contraindications": ["活动性消化道溃疡", "严重肝肾功能不全", "阿司匹林过敏"],
        "side_effects": ["胃肠道不适", "头晕", "皮疹", "水肿"],
        "interactions": ["与阿司匹林合用降低布洛芬疗效", "与抗凝药合用增加出血风险"],
        "pregnancy_category": "B/D（孕晚期D）",
    },
    "头孢克洛": {
        "name": "头孢克洛",
        "trade_names": ["希刻劳"],
        "category": "二代头孢菌素",
        "indications": ["呼吸道感染", "尿路感染", "中耳炎", "皮肤感染"],
        "dosage": "成人：0.25g/次，每8小时一次",
        "contraindications": ["头孢类过敏者禁用", "青霉素过敏者慎用"],
        "side_effects": ["腹泻", "恶心", "皮疹", "头痛"],
        "interactions": ["与丙磺舒合用升高血药浓度", "与氨基糖苷类合用增加肾毒性"],
        "pregnancy_category": "B",
    },
    "奥美拉唑": {
        "name": "奥美拉唑",
        "trade_names": ["洛赛克", "奥克"],
        "category": "质子泵抑制剂(PPI)",
        "indications": ["胃溃疡", "十二指肠溃疡", "胃食管反流病"],
        "dosage": "成人：20mg/次，每日1-2次，晨起空腹服用",
        "contraindications": ["对本品过敏者禁用"],
        "side_effects": ["头痛", "腹泻", "恶心", "长期使用可致维生素B12缺乏"],
        "interactions": ["抑制CYP2C19，影响氯吡格雷活化", "增加地高辛吸收"],
        "pregnancy_category": "C",
    },
    "二甲双胍": {
        "name": "二甲双胍",
        "trade_names": ["格华止"],
        "category": "双胍类降糖药",
        "indications": ["2型糖尿病", "胰岛素抵抗"],
        "dosage": "成人：起始0.5g/次，每日2-3次，逐渐加量，最大2g/日",
        "contraindications": ["严重肾功能不全", "代谢性酸中毒", "严重感染"],
        "side_effects": ["胃肠道反应", "乳酸酸中毒（罕见）", "维生素B12缺乏"],
        "interactions": ["与碘造影剂合用增加乳酸酸中毒风险", "与酒精合用增加风险"],
        "pregnancy_category": "B",
    },
}


# ============================================================
# 药物咨询 Worker 节点函数
# ============================================================

async def drug_worker_node(state: InquiryState) -> dict:
    """
    药物咨询 Worker 处理函数。

    执行流程：
      1. 从 state 中提取用户的药物咨询问题
      2. 使用规则匹配药品名称和咨询类型
      3. 查询药品知识库
      4. 使用 LLM 生成用药建议
      5. 更新 state

    Args:
        state: 当前 InquiryState

    Returns:
        需要更新的 state 字段字典
    """
    logger.info("[药物咨询Worker] 开始执行")

    # ---------- 1. 提取用户问题 ----------
    messages = state.get("messages", [])
    user_text = extract_latest_user_message(messages)
    logger.info(f"[药物咨询Worker] 用户问题: {user_text}")

    # ---------- 2. 解析药品和意图 ----------
    drug_name, query_type = _parse_drug_query(user_text)
    logger.info(f"[药物咨询Worker] 药品={drug_name}, 查询类型={query_type}")

    # ---------- 3. 查询药品知识库 ----------
    drug_info = DRUG_KNOWLEDGE_BASE.get(drug_name, {})
    if not drug_info:
        # 尝试模糊匹配
        drug_info = _fuzzy_match_drug(drug_name)

    # ---------- 4. LLM 生成用药建议 ----------
    if drug_info:
        response_text = await _generate_drug_advice(drug_info, query_type, user_text)
    else:
        # 字典未匹配，尝试 RAG 检索
        rag_context = ""
        try:
            from utils.rag_retrieval import rag_retriever
            rag_context = await rag_retriever.hybrid_retrieve(user_text, top_k=5)
            logger.info(f"[药物咨询Worker] 字典未匹配，RAG检索到 {len(rag_context)} 字符上下文")
        except Exception as e:
            logger.warning(f"[药物咨询Worker] RAG检索失败: {e}")

        if rag_context:
            response_text = await _generate_drug_advice_from_rag(
                drug_name, query_type, user_text, rag_context
            )
        else:
            # RAG 也无结果，尝试联网搜索
            from utils.web_search import web_search
            web_context = await web_search(f"{drug_name} 药品说明书 用法用量 副作用", max_results=5)
            if web_context:
                logger.info(f"[药物咨询Worker] 字典+RAG无结果，已启用联网搜索")
                response_text = await _generate_drug_advice_from_rag(
                    drug_name, query_type, user_text, web_context
                )
            else:
                response_text = (
                    f"抱歉，暂未查询到「{drug_name}」的详细药品信息。\n"
                    f"建议您：\n"
                    f"1. 确认药品通用名称\n"
                    f"2. 咨询药师或医师获取专业用药指导\n"
                    f"3. 仔细阅读药品说明书"
                )

    # ---------- 5. 更新 state ----------
    return {
        "drug_query": user_text,
        "drug_info": [drug_info] if drug_info else [],
        "final_response": response_text,
        "messages": [AIMessage(content=response_text)],
    }


# ============================================================
# 辅助函数
# ============================================================

def _parse_drug_query(text: str) -> tuple[str, str]:
    """
    从用户文本中解析药品名称和咨询类型。

    Returns:
        (药品名称, 查询类型)
        查询类型：dosage / side_effect / interaction / contraindication / general
    """
    # 1. 遍历已知药品名进行精确匹配
    for drug_name in DRUG_KNOWLEDGE_BASE.keys():
        if drug_name in text:
            # 判断查询类型
            if any(kw in text for kw in ["用法", "用量", "怎么吃", "怎么用", "剂量"]):
                return drug_name, "dosage"
            elif any(kw in text for kw in ["副作用", "不良反应", "有什么反应"]):
                return drug_name, "side_effect"
            elif any(kw in text for kw in ["禁忌", "不能吃", "不能用", "禁用"]):
                return drug_name, "contraindication"
            elif any(kw in text for kw in ["相互作用", "一起吃", "合用", "同时吃"]):
                return drug_name, "interaction"
            else:
                return drug_name, "general"

    # 2. 未匹配到已知药品，尝试模糊匹配
    fuzzy_match = _fuzzy_match_drug(text)
    if fuzzy_match:
        drug_name = fuzzy_match.get("name", "")
        if any(kw in text for kw in ["用法", "用量", "怎么吃", "怎么用", "剂量"]):
            return drug_name, "dosage"
        elif any(kw in text for kw in ["副作用", "不良反应", "有什么反应"]):
            return drug_name, "side_effect"
        elif any(kw in text for kw in ["禁忌", "不能吃", "不能用", "禁用"]):
            return drug_name, "contraindication"
        elif any(kw in text for kw in ["相互作用", "一起吃", "合用", "同时吃"]):
            return drug_name, "interaction"
        else:
            return drug_name, "general"

    # 3. 仍然未匹配，使用简单启发式提取药品名
    # 移除常见问句模式，提取可能的药品名
    import re
    
    # 匹配模式："XX有什么功效" -> XX
    pattern1 = r"(.+?)(?:有什么功效|是什么药|有什么用|能治什么)"
    match = re.search(pattern1, text)
    if match:
        return match.group(1).strip(), "general"
    
    # 匹配模式："XX怎么吃" -> XX
    pattern2 = r"(.+?)(?:怎么吃|怎么用|如何服用|用法用量)"
    match = re.search(pattern2, text)
    if match:
        return match.group(1).strip(), "dosage"
    
    # 匹配模式："XX的副作用" -> XX
    pattern3 = r"(.+?)(?:的副作用|的不良反应|有什么反应)"
    match = re.search(pattern3, text)
    if match:
        return match.group(1).strip(), "side_effect"
    
    # 默认：返回原文的前20个字符作为药品名（避免过长）
    return text[:20].strip(), "general"


def _fuzzy_match_drug(name: str) -> dict:
    """模糊匹配药品名称（包含关系匹配）"""
    for drug_name, info in DRUG_KNOWLEDGE_BASE.items():
        if name in drug_name or drug_name in name:
            return info
        # 检查商品名
        for trade in info.get("trade_names", []):
            if trade in name or name in trade:
                return info
    return {}


async def _generate_drug_advice_from_rag(
    drug_name: str, query_type: str, user_text: str, rag_context: str
) -> str:
    """当字典匹配失败时，基于 RAG 检索结果生成用药建议"""
    llm = _get_drug_llm()

    response = await llm.ainvoke([
        {
            "role": "system",
            "content": (
                "你是一位专业的临床药师。请基于以下参考资料回答用户的用药咨询。\n"
                "要求：\n"
                "1. 优先使用参考资料中的信息\n"
                "2. 回答专业准确，语言通俗易懂\n"
                "3. 包含必要的安全提示\n"
                "4. 如有严重不良反应或禁忌，要特别强调\n"
                "5. 最后提醒用户遵医嘱用药\n"
                "6. 不要编造参考资料中没有的信息"
            ),
        },
        {
            "role": "user",
            "content": (
                f"用户问题：{user_text}\n\n"
                f"药品名称：{drug_name}\n\n"
                f"参考资料：\n{rag_context}"
            ),
        },
    ])

    return f"💊 **用药咨询回复**\n\n{response.content}\n\n⚠️ 以上信息仅供参考，具体用药请遵医嘱。"


async def _generate_drug_advice(drug_info: dict, query_type: str, user_text: str) -> str:
    """使用 LLM 生成专业的用药建议（基于字典数据）"""
    llm = _get_drug_llm()

    # 根据查询类型组织药品信息
    info_text = f"药品名称：{drug_info['name']}\n"
    info_text += f"药品分类：{drug_info.get('category', '')}\n"

    if query_type in ("dosage", "general"):
        info_text += f"用法用量：{drug_info.get('dosage', '')}\n"
        info_text += f"适应症：{', '.join(drug_info.get('indications', []))}\n"

    if query_type in ("side_effect", "general"):
        info_text += f"不良反应：{', '.join(drug_info.get('side_effects', []))}\n"

    if query_type in ("contraindication", "general"):
        info_text += f"禁忌症：{', '.join(drug_info.get('contraindications', []))}\n"

    if query_type in ("interaction", "general"):
        info_text += f"药物相互作用：{', '.join(drug_info.get('interactions', []))}\n"

    info_text += f"妊娠分级：{drug_info.get('pregnancy_category', '未知')}\n"

    response = await llm.ainvoke([
        {
            "role": "system",
            "content": (
                "你是一位专业的临床药师。根据药品信息回答用户的用药咨询。\n"
                "要求：\n"
                "1. 回答专业准确，引用药品说明书信息\n"
                "2. 语言通俗易懂\n"
                "3. 包含必要的安全提示\n"
                "4. 如有严重不良反应或禁忌，要特别强调\n"
                "5. 最后提醒用户遵医嘱用药"
            ),
        },
        {
            "role": "user",
            "content": f"用户问题：{user_text}\n\n药品信息：\n{info_text}",
        },
    ])

    return f"💊 **用药咨询回复**\n\n{response.content}\n\n⚠️ 以上信息仅供参考，具体用药请遵医嘱。"


# ============================================================
# 单独调试入口
# ============================================================

if __name__ == "__main__":
    """单独测试药物咨询 Worker"""
    import asyncio
    from langchain_core.messages import HumanMessage
    from state.state_schema import make_initial_state

    async def test_drug():
        state = make_initial_state(user_id="test_001", session_id="sess_003")
        state["messages"] = [HumanMessage(content="布洛芬有什么副作用？能和阿莫西林一起吃吗？")]
        state["intent"] = "drug_inquiry"
        result = await drug_worker_node(state)
        print(f"用药建议:\n{result.get('final_response')}")

    asyncio.run(test_drug())
