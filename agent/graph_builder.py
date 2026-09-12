"""
agent/graph_builder.py
======================
使用 LangGraph 构建 Supervisor + Worker 多智能体协作图。

架构说明：
  - Supervisor 节点：意图识别 → 路由到对应 Worker
  - 5 个 Worker 节点：问诊采集 / 报告解读 / 药物咨询 / 知识问答 / 处方审核
  - 条件路由：根据 state.intent 决定走哪个 Worker
  - Worker 完成后回到 Supervisor，由 Supervisor 决定是否结束

图结构：
  START → supervisor →(条件路由)→ worker_x → supervisor → ... → END
"""

from __future__ import annotations
from loguru import logger
from langgraph.graph import StateGraph, START, END

from state.state_schema import InquiryState
from agent.supervisor_agent import supervisor_node
from agent.worker.collect_agent import collect_worker_node
from agent.worker.report_agent import report_worker_node
from agent.worker.drug_agent import drug_worker_node
from agent.worker.qa_agent import qa_worker_node
from agent.worker.prescription_agent import prescription_worker_node


# ============================================================
# 条件路由函数
# ============================================================

def route_by_intent(state: InquiryState) -> str:
    """
    根据 Supervisor 识别出的 intent 路由到对应 Worker 节点。

    路由表：
      consultation        → collect_worker
      report_analysis     → report_worker
      drug_inquiry        → drug_worker
      knowledge_qa        → qa_worker
      prescription_review → prescription_worker
      chitchat / 未知     → END（闲聊直接结束）

    Returns:
        目标节点名称字符串
    """
    intent = state.get("intent", "chitchat")
    should_end = state.get("should_end", False)

    # 如果 Supervisor 标记了结束（诊断收敛 / 任务完成），直接结束
    if should_end:
        logger.info(f"[路由] should_end=True，对话结束")
        return "end"

    # 意图 → 节点名称映射
    intent_to_node = {
        "consultation": "collect_worker",
        "report_analysis": "report_worker",
        "drug_inquiry": "drug_worker",
        "knowledge_qa": "qa_worker",
        "prescription_review": "prescription_worker",
    }

    target = intent_to_node.get(intent, "")
    if target:
        logger.info(f"[路由] intent={intent} → {target}")
        return target
    else:
        # 闲聊或无法识别的意图，直接结束
        logger.info(f"[路由] intent={intent} 无匹配Worker，结束对话")
        return "end"


# ============================================================
# 构建 LangGraph 状态图
# ============================================================

def build_inquiry_graph(checkpointer=None):
    """
    构建并编译多智能体协作状态图。

    节点列表：
      - supervisor:          Supervisor 调度 Agent（意图识别 + 收敛判断）
      - collect_worker:      问诊采集 Worker
      - report_worker:       报告解读 Worker
      - drug_worker:         药物咨询 Worker
      - qa_worker:           知识问答 Worker
      - prescription_worker: 处方审核 Worker

    边定义：
      START → supervisor
      supervisor →(条件路由)→ worker / end
      每个 worker → supervisor（回到 Supervisor 继续判断）

    Args:
        checkpointer: 可选的 LangGraph Checkpointer（用于状态持久化）

    Returns:
        编译后的 LangGraph CompiledGraph 实例
    """
    # 创建状态图，绑定 InquiryState
    graph = StateGraph(InquiryState)

    # ---------- 添加节点 ----------
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("collect_worker", collect_worker_node)
    graph.add_node("report_worker", report_worker_node)
    graph.add_node("drug_worker", drug_worker_node)
    graph.add_node("qa_worker", qa_worker_node)
    graph.add_node("prescription_worker", prescription_worker_node)

    # ---------- 添加边 ----------
    # 入口边：START → supervisor
    graph.add_edge(START, "supervisor")

    # Supervisor 条件路由：根据 intent 分发到 Worker 或 END
    graph.add_conditional_edges(
        "supervisor",
        route_by_intent,
        {
            "collect_worker": "collect_worker",
            "report_worker": "report_worker",
            "drug_worker": "drug_worker",
            "qa_worker": "qa_worker",
            "prescription_worker": "prescription_worker",
            "end": END,
        },
    )

    # 每个 Worker 完成后回到 Supervisor
    graph.add_edge("collect_worker", "supervisor")
    graph.add_edge("report_worker", "supervisor")
    graph.add_edge("drug_worker", "supervisor")
    graph.add_edge("qa_worker", "supervisor")
    graph.add_edge("prescription_worker", "supervisor")

    # ---------- 编译图 ----------
    if checkpointer is not None:
        compiled = graph.compile(checkpointer=checkpointer)
        logger.info("[GraphBuilder] 多智能体协作图编译完成（带 Checkpointer）")
    else:
        compiled = graph.compile()
        logger.info("[GraphBuilder] 多智能体协作图编译完成")
    return compiled


# ============================================================
# 便捷函数：获取编译好的图实例（单例）
# ============================================================

_graph_instance = None


def get_graph():
    """获取全局编译图实例（单例模式）"""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = build_inquiry_graph()
    return _graph_instance


# ============================================================
# 单独调试入口
# ============================================================

if __name__ == "__main__":
    """直接运行此文件可打印图结构并测试运行"""
    from langchain_core.messages import HumanMessage
    from state.state_schema import make_initial_state

    graph = build_inquiry_graph()

    # 打印图结构
    try:
        mermaid = graph.get_graph().draw_mermaid()
        print("=== 图结构 (Mermaid) ===")
        print(mermaid)
    except Exception:
        print("=== 图结构 ===")
        print(graph.get_graph())

    # 简单测试运行
    print("\n=== 测试运行 ===")
    initial_state = make_initial_state(user_id="test_001", session_id="sess_001")
    initial_state["messages"] = [HumanMessage(content="我最近总是头疼，还有点发烧")]

    # 使用 stream 模式逐步执行
    for step in graph.stream(initial_state, {"recursion_limit": 25}):
        node_name = list(step.keys())[0]
        print(f"\n>>> 执行节点: {node_name}")
        state_snapshot = step[node_name]
        if state_snapshot.get("final_response"):
            print(f"    响应: {state_snapshot['final_response'][:200]}")
