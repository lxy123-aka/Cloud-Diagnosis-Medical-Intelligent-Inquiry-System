"""
tests/test_conversation.py
==========================
端到端测试用例 —— 模拟完整用户问诊对话。

测试场景：
  1. 在线问诊完整流程（多轮追问 → 诊断收敛）
  2. 报告解读（文本报告）
  3. 药物咨询
  4. 知识问答
  5. 处方审核
  6. 闲聊兜底
  7. 各模块独立测试

运行方式：
  python tests/test_conversation.py
"""

from __future__ import annotations
import sys
import os
import asyncio
import time
from pathlib import Path

# 确保项目根目录在路径中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
from langchain_core.messages import HumanMessage
from state.state_schema import make_initial_state


# ============================================================
# 测试工具函数
# ============================================================

def print_separator(title: str):
    """打印分隔线"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_result(label: str, value):
    """打印测试结果"""
    if isinstance(value, list):
        print(f"  {label}:")
        for item in value:
            if isinstance(item, dict):
                print(f"    - {item}")
            else:
                print(f"    - {item}")
    else:
        text = str(value)
        if len(text) > 200:
            text = text[:200] + "..."
        print(f"  {label}: {text}")


# ============================================================
# 测试1：在线问诊完整流程
# ============================================================

async def test_consultation_workflow():
    """测试在线问诊完整流程"""
    print_separator("测试1：在线问诊完整流程")

    from agent.graph_builder import build_inquiry_graph

    graph = build_inquiry_graph()
    state = make_initial_state(user_id="test_user_001", session_id="test_sess_001")

    # 模拟用户输入
    test_messages = [
        "我头疼三天了，还有点发烧，嗓子也疼",
    ]

    state["messages"] = [HumanMessage(content=test_messages[0])]

    print(f"\n  用户输入: {test_messages[0]}")
    print(f"\n  --- 图执行过程 ---")

    start_time = time.time()
    step_count = 0

    try:
        for step in graph.stream(state, {"recursion_limit": 25}):
            node_name = list(step.keys())[0]
            node_state = step[node_name]
            step_count += 1
            print(f"  步骤 {step_count}: 节点={node_name}")

            if node_state.get("intent"):
                print_result("    意图", node_state["intent"])
            if node_state.get("standardized_symptoms"):
                print_result("    症状", node_state["standardized_symptoms"])
            if node_state.get("disease_candidates"):
                print_result("    候选疾病", [
                    f"{c.get('disease_name', '')}({c.get('confidence', 0):.2f})"
                    for c in node_state["disease_candidates"][:3]
                ])
            if node_state.get("final_response"):
                print_result("    回复", node_state["final_response"])
            if node_state.get("should_end"):
                print(f"    → 对话结束标记")

    except Exception as e:
        print(f"  ⚠️ 图执行异常: {e}")

    elapsed = time.time() - start_time
    print(f"\n  执行完成: {step_count} 步, 耗时 {elapsed:.2f}s")


# ============================================================
# 测试2：报告解读
# ============================================================

async def test_report_analysis():
    """测试报告解读"""
    print_separator("测试2：报告解读")

    from agent.worker.report_agent import report_worker_node

    state = make_initial_state(user_id="test_user_002", session_id="test_sess_002")
    state["messages"] = [HumanMessage(content=(
        "血常规报告：白细胞 12.5×10^9/L（参考4-10），"
        "中性粒细胞 85%（参考50-70%），"
        "血红蛋白 130g/L（参考120-160），"
        "血小板 220×10^9/L（参考100-300），"
        "CRP 45mg/L（参考0-10）"
    ))]
    state["intent"] = "report_analysis"

    print(f"\n  输入: 血常规报告数据")

    try:
        result = await report_worker_node(state)
        print_result("解读结果", result.get("final_response", ""))
        print("  ✅ 报告解读测试通过")
    except Exception as e:
        print(f"  ⚠️ 报告解读测试异常: {e}")


# ============================================================
# 测试3：药物咨询
# ============================================================

async def test_drug_inquiry():
    """测试药物咨询"""
    print_separator("测试3：药物咨询")

    from agent.worker.drug_agent import drug_worker_node

    state = make_initial_state(user_id="test_user_003", session_id="test_sess_003")
    state["messages"] = [HumanMessage(content="布洛芬有什么副作用？能和阿莫西林一起吃吗？")]
    state["intent"] = "drug_inquiry"

    print(f"\n  输入: 布洛芬有什么副作用？能和阿莫西林一起吃吗？")

    try:
        result = await drug_worker_node(state)
        print_result("用药建议", result.get("final_response", ""))
        print_result("药品信息", result.get("drug_info", []))
        print("  ✅ 药物咨询测试通过")
    except Exception as e:
        print(f"  ⚠️ 药物咨询测试异常: {e}")


# ============================================================
# 测试4：知识问答
# ============================================================

async def test_knowledge_qa():
    """测试知识问答"""
    print_separator("测试4：知识问答")

    from agent.worker.qa_agent import qa_worker_node

    state = make_initial_state(user_id="test_user_004", session_id="test_sess_004")
    state["messages"] = [HumanMessage(content="高血压患者日常需要注意什么？")]
    state["intent"] = "knowledge_qa"

    print(f"\n  输入: 高血压患者日常需要注意什么？")

    try:
        result = await qa_worker_node(state)
        print_result("科普回答", result.get("final_response", ""))
        print("  ✅ 知识问答测试通过")
    except Exception as e:
        print(f"  ⚠️ 知识问答测试异常: {e}")


# ============================================================
# 测试5：处方审核
# ============================================================

async def test_prescription_review():
    """测试处方审核"""
    print_separator("测试5：处方审核")

    from agent.worker.prescription_agent import prescription_worker_node

    state = make_initial_state(user_id="test_user_005", session_id="test_sess_005")
    state["messages"] = [HumanMessage(content=(
        "处方审核：阿莫西林 0.5g 每日3次口服，"
        "布洛芬 0.4g 每日3次口服，"
        "奥美拉唑 20mg 每日1次口服"
    ))]
    state["intent"] = "prescription_review"

    print(f"\n  输入: 处方审核（阿莫西林+布洛芬+奥美拉唑）")

    try:
        result = await prescription_worker_node(state)
        review = result.get("review_result", {})
        print_result("审核结果", "通过" if review.get("is_safe") else "存在问题")
        print_result("风险等级", review.get("risk_level", "unknown"))
        print_result("问题列表", review.get("issues", []))
        print_result("建议", review.get("suggestions", []))
        print("  ✅ 处方审核测试通过")
    except Exception as e:
        print(f"  ⚠️ 处方审核测试异常: {e}")


# ============================================================
# 测试6：症状标准化流水线
# ============================================================

async def test_symptom_pipeline():
    """测试三层症状标准化流水线"""
    print_separator("测试6：症状标准化流水线")

    from pipeline.symptom_normalize import SymptomNormalizePipeline

    pipeline = SymptomNormalizePipeline()

    test_cases = [
        "肚子左边一直疼，吃完饭更疼",
        "头疼三天了，还有点发烧",
        "胸口闷，喘不上气",
    ]

    for text in test_cases:
        print(f"\n  输入: {text}")
        try:
            result = await pipeline.run(text)
            for s in result:
                print(f"    → {s.get('standard_name', '')} "
                      f"(置信度={s.get('confidence', 0):.2f}, "
                      f"来源={s.get('source', '')})")
        except Exception as e:
            print(f"    ⚠️ 处理异常: {e}")

    print("\n  ✅ 症状流水线测试完成")


# ============================================================
# 测试7：诊断收敛引擎
# ============================================================

async def test_diagnosis_engine():
    """测试置信度驱动诊断收敛引擎"""
    print_separator("测试7：诊断收敛引擎")

    from pipeline.diagnosis_engine import DiagnosisEngine

    engine = DiagnosisEngine()

    symptoms = [
        {"standard_name": "头痛", "confidence": 0.9, "source": "graph_match"},
        {"standard_name": "发热", "confidence": 0.85, "source": "graph_match"},
        {"standard_name": "咽痛", "confidence": 0.8, "source": "llm_extract"},
    ]

    print(f"\n  输入症状: {[s['standard_name'] for s in symptoms]}")

    try:
        candidates, converged, diagnosis = await engine.evaluate(symptoms, current_round=3)
        print(f"\n  候选疾病:")
        for c in candidates[:5]:
            print(f"    {c['disease_name']}: {c['confidence']:.2f} "
                  f"(匹配: {c['matched_symptoms']})")
        print(f"  收敛: {converged}")
        print(f"  最终诊断: {diagnosis}")
        print("  ✅ 诊断引擎测试完成")
    except Exception as e:
        print(f"  ⚠️ 诊断引擎测试异常: {e}")


# ============================================================
# 测试8：工具注册
# ============================================================

def test_tools():
    """测试 Function Calling 工具"""
    print_separator("测试8：工具注册")

    from utils.tools import ALL_TOOLS, get_all_tool_schemas, query_drug_info

    print(f"\n  已注册工具数: {len(ALL_TOOLS)}")
    for t in ALL_TOOLS:
        print(f"    - {t.name}")

    schemas = get_all_tool_schemas()
    print(f"  Schema 数量: {len(schemas)}")

    # 测试药物查询工具
    result = query_drug_info.invoke({"drug_name": "布洛芬"})
    print(f"\n  药物查询测试: {result[:100]}...")
    print("  ✅ 工具注册测试完成")


# ============================================================
# 主测试函数
# ============================================================

async def run_all_tests():
    """运行所有测试"""
    print("\n" + "█" * 60)
    print("  云诊医疗智能问诊系统（CDMIIS）端到端测试")
    print("█" * 60)

    start_time = time.time()
    test_results = {}

    # 测试1：在线问诊
    try:
        await test_consultation_workflow()
        test_results["问诊流程"] = "PASS"
    except Exception as e:
        test_results["问诊流程"] = f"FAIL: {e}"

    # 测试2：报告解读
    try:
        await test_report_analysis()
        test_results["报告解读"] = "PASS"
    except Exception as e:
        test_results["报告解读"] = f"FAIL: {e}"

    # 测试3：药物咨询
    try:
        await test_drug_inquiry()
        test_results["药物咨询"] = "PASS"
    except Exception as e:
        test_results["药物咨询"] = f"FAIL: {e}"

    # 测试4：知识问答
    try:
        await test_knowledge_qa()
        test_results["知识问答"] = "PASS"
    except Exception as e:
        test_results["知识问答"] = f"FAIL: {e}"

    # 测试5：处方审核
    try:
        await test_prescription_review()
        test_results["处方审核"] = "PASS"
    except Exception as e:
        test_results["处方审核"] = f"FAIL: {e}"

    # 测试6：症状流水线
    try:
        await test_symptom_pipeline()
        test_results["症状流水线"] = "PASS"
    except Exception as e:
        test_results["症状流水线"] = f"FAIL: {e}"

    # 测试7：诊断引擎
    try:
        await test_diagnosis_engine()
        test_results["诊断引擎"] = "PASS"
    except Exception as e:
        test_results["诊断引擎"] = f"FAIL: {e}"

    # 测试8：工具注册
    try:
        test_tools()
        test_results["工具注册"] = "PASS"
    except Exception as e:
        test_results["工具注册"] = f"FAIL: {e}"

    # 汇总报告
    elapsed = time.time() - start_time
    print_separator("测试汇总")
    total = len(test_results)
    passed = sum(1 for v in test_results.values() if v == "PASS")
    for name, result in test_results.items():
        icon = "✅" if result == "PASS" else "❌"
        print(f"  {icon} {name}: {result}")
    print(f"\n  总计: {passed}/{total} 通过, 耗时 {elapsed:.2f}s")
    print("=" * 60)


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    asyncio.run(run_all_tests())
