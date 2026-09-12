"""
agent/worker/report_agent.py
============================
报告解读 Worker Agent —— 医学报告 / 影像分析。

职责：
  1. 接收 state，判断输入是文本报告还是图片
  2. 文本报告：调用 LLM 提取医学实体并解读
  3. 图片输入：调用 Qwen-VL 多模态模型解析 CT/B超/化验单图片
  4. 生成结构化解读结论写入 state
"""

from __future__ import annotations
from loguru import logger
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from state.state_schema import InquiryState
from configs.settings import settings
from utils.message_utils import extract_latest_user_message


# 模块级别 LLM 单例（报告解读专用）
_report_llm = None


def _get_report_llm() -> ChatOpenAI:
    """获取报告解读专用 LLM 实例（模块级单例）"""
    global _report_llm
    if _report_llm is None:
        _report_llm = ChatOpenAI(
            model=settings.QWEN_MODEL_NAME,
            api_key=settings.QWEN_API_KEY,
            base_url=settings.QWEN_BASE_URL,
            temperature=0.2,
            max_tokens=2048,
        )
    return _report_llm


# ============================================================
# 报告解读 Worker 节点函数
# ============================================================

async def report_worker_node(state: InquiryState) -> dict:
    """
    报告解读 Worker 处理函数。

    执行流程：
      1. 检查 state 中是否有图片（has_image / image_paths）
      2. 有图片 → 调用 Qwen-VL 多模态推理
      3. 无图片 → 从消息中提取报告文本，用 LLM 解读
      4. 生成解读结论，更新 state

    Args:
        state: 当前 InquiryState

    Returns:
        需要更新的 state 字段字典
    """
    logger.info("[报告解读Worker] 开始执行")

    has_image = state.get("has_image", False)
    image_paths = state.get("image_paths", [])

    if has_image and image_paths:
        # ====== 图片输入：调用 Qwen-VL 多模态 ======
        logger.info(f"[报告解读Worker] 检测到图片输入: {image_paths}")
        analysis = await _analyze_image(image_paths)
    else:
        # ====== 文本报告：LLM 解读 ======
        logger.info("[报告解读Worker] 文本报告解读")
        analysis = await _analyze_text_report(state)

    # 生成面向用户的解读结论
    response = _format_report_response(analysis)

    return {
        "image_analysis": analysis if has_image else "",
        "final_response": response,
        "messages": [AIMessage(content=response)],
    }


# ============================================================
# 图片分析（Qwen-VL 多模态）
# ============================================================

async def _analyze_image(image_paths: list[str]) -> str:
    """
    调用 Qwen-VL 模型分析医学影像图片。
    支持格式：CT、B超、化验单、检验报告等。

    Args:
        image_paths: 图片文件路径列表

    Returns:
        图片分析结果文本
    """
    from multimodal.vl_infer import QwenVLInfer
    from pathlib import Path

    vl_infer = QwenVLInfer()
    all_analysis = []
    
    # 限制图片大小（10MB）
    MAX_IMAGE_SIZE = 10 * 1024 * 1024

    for img_path in image_paths:
        try:
            # 检查文件大小
            file_size = Path(img_path).stat().st_size
            if file_size > MAX_IMAGE_SIZE:
                logger.warning(f"[报告解读Worker] 图片过大 ({file_size} bytes)，跳过")
                all_analysis.append(f"图片 {img_path} 过大，暂不支持分析")
                continue
            
            # 读取图片并编码为 base64
            with open(img_path, "rb") as f:
                img_data = f.read()

            # 调用 Qwen-VL 推理
            result = await vl_infer.analyze_medical_image(
                image_data=img_data,
                prompt=(
                    "请详细分析这张医学图片，包括：\n"
                    "1. 图片类型（CT/B超/化验单等）\n"
                    "2. 关键发现\n"
                    "3. 异常指标\n"
                    "4. 临床意义"
                ),
            )
            all_analysis.append(result)
            logger.info(f"[报告解读Worker] 图片分析完成: {img_path}")

        except Exception as e:
            logger.error(f"[报告解读Worker] 图片分析失败 {img_path}: {e}")
            all_analysis.append(f"图片 {img_path} 分析失败: {str(e)}")

    return "\n\n".join(all_analysis)


# ============================================================
# 文本报告分析
# ============================================================

async def _analyze_text_report(state: InquiryState) -> str:
    """
    使用 LLM 解读文本形式的医学报告。
    从消息历史中提取报告内容，调用 LLM 做结构化解读。
    """
    messages = state.get("messages", [])

    # 提取用户消息中的报告文本
    report_text = extract_latest_user_message(messages)

    if not report_text:
        return "未检测到报告内容，请您提供需要解读的检查报告。"

    # 调用 LLM 解读
    llm = _get_report_llm()

    response = await llm.ainvoke([
        {
            "role": "system",
            "content": (
                "你是一位资深医学检验报告解读专家。请对用户提供的医学报告进行专业解读。\n"
                "解读要求：\n"
                "1. 提取报告中的关键指标和异常值\n"
                "2. 解释每个异常指标的临床意义\n"
                "3. 给出可能的健康风险提示\n"
                "4. 提供就医建议\n"
                "5. 使用通俗易懂的语言，同时保持医学专业性\n"
                "6. 结构化输出：【报告类型】【关键发现】【异常指标】【临床意义】【建议】"
            ),
        },
        {"role": "user", "content": f"请解读以下医学报告：\n\n{report_text}"},
    ])

    return response.content


# ============================================================
# 格式化输出
# ============================================================

def _format_report_response(analysis: str) -> str:
    """格式化报告解读结果，面向用户友好展示"""
    response = "📋 **报告解读结果**\n\n"
    response += analysis
    response += "\n\n⚠️ 以上为AI辅助解读，仅供参考。具体诊断请咨询专业医生。"
    return response


# ============================================================
# 单独调试入口
# ============================================================

if __name__ == "__main__":
    """单独测试报告解读 Worker"""
    import asyncio
    from langchain_core.messages import HumanMessage
    from state.state_schema import make_initial_state

    async def test_report():
        state = make_initial_state(user_id="test_001", session_id="sess_002")
        state["messages"] = [HumanMessage(content=(
            "血常规报告：白细胞 12.5×10^9/L（参考4-10），"
            "中性粒细胞 85%（参考50-70%），"
            "血红蛋白 130g/L（参考120-160），"
            "血小板 220×10^9/L（参考100-300），"
            "CRP 45mg/L（参考0-10）"
        ))]
        state["intent"] = "report_analysis"
        result = await report_worker_node(state)
        print(f"解读结果:\n{result.get('final_response')}")

    asyncio.run(test_report())
