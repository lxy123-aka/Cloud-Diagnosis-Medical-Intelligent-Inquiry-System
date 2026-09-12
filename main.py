"""
main.py
=======
云诊医疗智能问诊系统（CDMIIS）启动入口。

功能：
  1. 加载配置，初始化日志
  2. 初始化数据库连接（Redis、PostgreSQL）
  3. 构建 LangGraph 多智能体对话图
  4. 提供对话调用入口
  5. 支持文本/图片输入
  6. 启动智能问诊会话

使用方式：
  命令行交互模式：python main.py
  API 服务模式：python main.py --api
"""

from __future__ import annotations
import sys
import os
import uuid
import asyncio
from pathlib import Path
from loguru import logger

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import settings
from state.state_schema import InquiryState, make_initial_state


# ============================================================
# 初始化
# ============================================================

def setup_logging():
    """初始化日志"""
    logger.remove()
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <7}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
            "<level>{message}</level>"
        ),
        level="INFO",
        colorize=True,
    )
    # 文件日志
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.add(
        str(log_dir / "app_{time:YYYY-MM-DD}.log"),
        rotation="00:00",
        retention="30 days",
        encoding="utf-8",
    )


async def init_services():
    """初始化外部服务连接"""
    logger.info("=" * 50)
    logger.info(f"  {settings.PROJECT_NAME} v{settings.VERSION}")
    logger.info("=" * 50)

    # 初始化 Redis（短期记忆）
    try:
        from memory.short_memory import redis_saver
        await redis_saver._get_redis()
        logger.info("✅ Redis 短期记忆连接成功")
    except Exception as e:
        logger.warning(f"⚠️ Redis 连接失败（可选）: {e}")

    # 初始化 PostgreSQL（长期记忆）
    try:
        from memory.long_memory import long_term_memory
        await long_term_memory.initialize()
        logger.info("✅ PostgreSQL 长期记忆连接成功")
    except Exception as e:
        logger.warning(f"⚠️ PostgreSQL 连接失败（可选）: {e}")

    logger.info("✅ 服务初始化完成")


# ============================================================
# 对话引擎
# ============================================================

class ConversationEngine:
    """
    对话引擎 —— 封装 LangGraph 图的调用逻辑。
    支持文本和图片输入，管理会话状态。
    """

    def __init__(self):
        self._graph = None
        self._user_id = ""
        self._session_id = ""
        self._state = None  # 持久化 state，保存对话历史

    def initialize(self, user_id: str = None, session_id: str = None):
        """初始化对话引擎"""
        from agent.graph_builder import get_graph
        self._graph = get_graph()
        self._user_id = user_id or str(uuid.uuid4())[:8]
        self._session_id = session_id or str(uuid.uuid4())[:8]
        logger.info(
            f"[对话引擎] 初始化完成: user={self._user_id}, session={self._session_id}"
        )

    async def chat(self, user_input: str, image_paths: list[str] = None) -> str:
        """
        发送用户消息并获取系统回复。

        :param user_input: 用户文本输入
        :param image_paths: 图片路径列表（可选）
        :return: 系统回复文本
        """
        if self._graph is None:
            self.initialize()

        from langchain_core.messages import HumanMessage

        # 如果是首次对话，创建初始状态或从 Redis 恢复
        if self._state is None:
            # 先尝试从 Redis 恢复状态（支持服务重启后继续问诊）
            restored = False
            try:
                from memory.short_memory import redis_saver
                saved_state = await redis_saver.load_full_state(self._session_id)
                if saved_state:
                    self._state = saved_state
                    # 确保 messages 是列表
                    if not isinstance(self._state.get("messages"), list):
                        self._state["messages"] = []
                    restored = True
                    logger.info(
                        f"[对话引擎] 从 Redis 恢复状态: session={self._session_id}, "
                        f"轮次={self._state.get('inquiry_round', 0)}, "
                        f"症状={[s.get('standard_name', '') for s in self._state.get('standardized_symptoms', [])]}"
                    )
            except Exception as e:
                logger.debug(f"[对话引擎] Redis 状态恢复跳过: {e}")

            if not restored:
                self._state = make_initial_state(
                    user_id=self._user_id,
                    session_id=self._session_id,
                )
                self._state["messages"] = []

        # 添加用户消息到现有 state
        self._state["messages"].append(HumanMessage(content=user_input))

        # 处理图片输入
        if image_paths:
            self._state["has_image"] = True
            self._state["image_paths"] = image_paths

        # 加载历史病历（长期记忆）
        try:
            from memory.long_memory import long_term_memory
            history = await long_term_memory.load_history_for_state(self._user_id)
            if history:
                self._state["medical_history"] = history
                logger.info(f"[对话引擎] 加载了 {len(history)} 条历史病历")
        except Exception as e:
            logger.debug(f"[对话引擎] 历史病历加载跳过: {e}")

        # 执行图（传入当前 state + thread_id config）
        accumulated_updates = {}  # 累积所有节点的状态更新（避免只取最后一个节点丢失中间状态）
        worker_responses = {}  # 收集各 Worker 的响应
        thread_config = {
            "recursion_limit": 30,
            "configurable": {"thread_id": self._session_id},
        }
        try:
            async for step in self._graph.astream(
                self._state, thread_config
            ):
                node_name = list(step.keys())[0]
                node_state = step[node_name]

                # 累积所有节点的状态更新
                accumulated_updates.update(node_state)

                # 收集 Worker 节点的响应（优先使用 Worker 的 final_response）
                if node_name.endswith("_worker") and node_state.get("final_response"):
                    worker_responses[node_name] = node_state["final_response"]

                logger.debug(f"[对话引擎] 节点执行: {node_name}")

        except Exception as e:
            logger.error(f"[对话引擎] 图执行失败: {e}")
            return f"抱歉，系统处理您的请求时出现错误：{str(e)}"

        # 更新持久化 state（使用累积的所有节点更新，而非仅最后一个节点）
        if accumulated_updates:
            self._state.update(accumulated_updates)

        # 提取回复（优先使用 Worker 的响应，其次是 accumulated_updates 中的 final_response）
        if worker_responses:
            # 使用最后一个 Worker 的响应（通常是当前执行的 Worker）
            response = list(worker_responses.values())[-1]
            logger.info(f"[对话引擎] 使用 Worker 响应: {list(worker_responses.keys())[-1]}")
        elif accumulated_updates.get("final_response"):
            response = accumulated_updates["final_response"]
        else:
            response = "您好，请问还有什么可以帮您的吗？"

        # 重置 intent 和 current_worker，确保下次 chat() 调用时 Supervisor 
        # 重新做意图分类，而不是错误地进入"Worker返回后检查收敛"分支导致图循环不退出
        self._state["intent"] = ""
        self._state["current_worker"] = ""

        # 保存到短期记忆 + Redis 状态持久化
        try:
            from memory.short_memory import redis_saver
            await redis_saver.save_conversation_turn(
                session_id=self._session_id,
                user_message=user_input,
                assistant_message=response,
            )
            # 保存完整状态到 Redis（支持重启恢复）
            await redis_saver.save_full_state(self._session_id, self._state)
        except Exception as e:
            logger.debug(f"[对话引擎] 短期记忆保存跳过: {e}")

        return response

    async def end_session(self):
        """结束会话，保存病历"""
        try:
            from memory.long_memory import long_term_memory
            # 这里可以保存完整病历到 PostgreSQL
            logger.info(f"[对话引擎] 会话结束: {self._session_id}")
        except Exception as e:
            logger.debug(f"[对话引擎] 会话结束处理跳过: {e}")


# ============================================================
# 命令行交互模式
# ============================================================

async def interactive_mode():
    """命令行交互模式"""
    print("\n" + "=" * 60)
    print(f"  欢迎使用 {settings.PROJECT_NAME} v{settings.VERSION}")
    print("  输入您的问题开始问诊，输入 'quit' 或 'exit' 退出")
    print("  输入 'image:图片路径' 上传医学图片")
    print("=" * 60 + "\n")

    engine = ConversationEngine()
    engine.initialize()

    while True:
        try:
            user_input = input("🧑 您: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！祝您健康！")
            break

        # 处理图片输入
        image_paths = None
        if user_input.lower().startswith("image:"):
            img_path = user_input[6:].strip()
            if os.path.exists(img_path):
                image_paths = [img_path]
                user_input = input("🧑 请描述您想了解的关于这张图片的问题: ").strip()
            else:
                print(f"❌ 图片文件不存在: {img_path}")
                continue

        # 调用对话引擎
        print("🤖 系统处理中...\n")
        response = await engine.chat(user_input, image_paths)
        print(f"🤖 系统: {response}\n")

    await engine.end_session()


# ============================================================
# API 服务模式
# ============================================================

def start_api_server():
    """启动 FastAPI 服务"""
    import uvicorn

    logger.info(f"启动 API 服务: {settings.API_HOST}:{settings.API_PORT}")

    # 延迟导入避免循环依赖
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

    class ChatRequest(BaseModel):
        message: str
        user_id: str = ""
        session_id: str = ""
        image_paths: list[str] = []

    class ChatResponse(BaseModel):
        response: str
        intent: str = ""
        converged: bool = False

    # 全局引擎池（简化版，生产环境应使用连接池）
    engines: dict[str, ConversationEngine] = {}

    @app.post("/chat", response_model=ChatResponse)
    async def chat_endpoint(req: ChatRequest):
        session_key = req.session_id or str(uuid.uuid4())[:8]
        if session_key not in engines:
            engine = ConversationEngine()
            engine.initialize(user_id=req.user_id, session_id=session_key)
            engines[session_key] = engine

        engine = engines[session_key]
        response = await engine.chat(req.message, req.image_paths or None)

        return ChatResponse(response=response)

    @app.get("/health")
    async def health_check():
        return {"status": "ok", "version": settings.VERSION}

    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)


# ============================================================
# 主入口
# ============================================================

async def main():
    """主入口"""
    setup_logging()
    await init_services()

    # 检查命令行参数
    if "--api" in sys.argv:
        start_api_server()
    else:
        await interactive_mode()


if __name__ == "__main__":
    asyncio.run(main())
