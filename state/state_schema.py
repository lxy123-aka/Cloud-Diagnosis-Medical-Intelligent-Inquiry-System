"""
LangGraph 全局状态 Schema 定义
所有 Agent / Pipeline / Memory 共享此状态结构。
继承 MessagesState 获得消息历史管理能力。
"""

from __future__ import annotations
from typing import Optional, Any
from typing_extensions import TypedDict
from langgraph.graph import MessagesState


# ============================================================
# 辅助数据结构（TypedDict，便于 JSON 序列化）
# ============================================================

class SymptomInfo(TypedDict, total=False):
    """标准化症状信息"""
    raw_text: str            # 用户原始描述
    standard_name: str       # 标准化医学术语
    snomed_code: str         # SNOMED-CT 编码
    category: str            # 症状分类（全身/呼吸/消化...）
    severity: str            # 严重程度：轻度/中度/重度
    confidence: float        # 置信度 0-1
    source: str              # 来源：llm_extract / graph_match / vector_recall
    duration: str            # 持续时间
    questions: list[str]     # 关联追问模板


class DiseaseCandidate(TypedDict, total=False):
    """候选疾病"""
    disease_name: str              # 疾病名称
    icd_code: str                  # ICD-10 编码
    department: str                # 就诊科室
    confidence: float              # 置信度 0-1
    matched_symptoms: list[str]    # 已匹配的症状列表
    total_symptoms: int            # 该疾病总症状数
    symptom_coverage: float        # 症状覆盖率
    evidence_paths: list[str]      # 图谱证据路径


class DrugInfo(TypedDict, total=False):
    """药品信息"""
    name: str                      # 药品通用名
    trade_names: list[str]         # 商品名
    category: str                  # 药物分类
    indications: list[str]         # 适应症
    dosage: str                    # 用法用量
    contraindications: list[str]   # 禁忌症
    side_effects: list[str]        # 不良反应
    interactions: list[str]        # 药物相互作用
    pregnancy_category: str        # 妊娠分级


class PrescriptionItem(TypedDict, total=False):
    """处方条目"""
    drug_name: str
    dosage: str
    frequency: str
    route: str                     # 给药途径


class ReviewResult(TypedDict, total=False):
    """处方审核结果"""
    is_safe: bool
    risk_level: str                # low / medium / high / critical
    issues: list[str]              # 发现的问题列表
    suggestions: list[str]         # 修改建议


class MedicalHistory(TypedDict, total=False):
    """历史病历摘要"""
    record_id: str
    visit_date: str
    chief_complaint: str
    symptoms: list[str]
    diagnosis: str
    treatment: str


# ============================================================
# LangGraph 主状态定义
# ============================================================

class InquiryState(MessagesState):
    """
    LangGraph 主状态。
    继承 MessagesState 获得 messages 列表管理。
    所有 Agent 节点共享读写此状态。
    """
    # ---------- 用户与会话 ----------
    user_id: str
    session_id: str

    # ---------- 意图分类 ----------
    intent: str                    # consultation / report_analysis / drug_inquiry / knowledge_qa / prescription_review / chitchat
    intent_confidence: float

    # ---------- 症状标准化 ----------
    standardized_symptoms: list[dict]

    # ---------- 诊断收敛 ----------
    disease_candidates: list[dict]
    diagnosis_confidence: float
    inquiry_round: int
    max_rounds: int
    diagnosis_converged: bool
    final_diagnosis: str
    asked_questions: list[str]       # 已问过的问题列表（跨轮次追踪，避免重复追问）

    # ---------- 多模态 ----------
    has_image: bool
    image_paths: list[str]
    image_analysis: str

    # ---------- 药物 & 处方 ----------
    drug_query: str
    drug_info: list[dict]
    prescription_items: list[dict]
    review_result: dict

    # ---------- 知识问答 ----------
    qa_question: str
    qa_answer: str

    # ---------- 记忆 ----------
    medical_history: list[dict]

    # ---------- 最终输出 ----------
    final_response: str
    current_worker: str            # 当前路由到的 worker 名称
    should_end: bool               # 是否结束对话


def make_initial_state(
    user_id: str = "",
    session_id: str = "",
    max_rounds: int = 10,
) -> dict:
    """构造初始状态字典，供图 invoke 使用"""
    return {
        "user_id": user_id,
        "session_id": session_id,
        "intent": "",
        "intent_confidence": 0.0,
        "standardized_symptoms": [],
        "disease_candidates": [],
        "diagnosis_confidence": 0.0,
        "inquiry_round": 0,
        "max_rounds": max_rounds,
        "diagnosis_converged": False,
        "final_diagnosis": "",
        "asked_questions": [],
        "has_image": False,
        "image_paths": [],
        "image_analysis": "",
        "drug_query": "",
        "drug_info": [],
        "prescription_items": [],
        "review_result": {},
        "qa_question": "",
        "qa_answer": "",
        "medical_history": [],
        "final_response": "",
        "current_worker": "",
        "should_end": False,
    }
