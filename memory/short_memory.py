"""
memory/short_memory.py
======================
短期会话记忆 —— 基于 RedisStack。

功能：
  1. 实现 AsyncRedisSaver 作为 LangGraph Checkpointer
  2. 使用 RedisJSON 存储结构化对话状态
  3. 支持会话历史快速读写
  4. 对接 LangGraph 的 checkpoint 机制
"""

from __future__ import annotations
import json
from typing import Any, Optional, AsyncIterator
from datetime import datetime, timezone
from loguru import logger
from langchain_core.runnables import RunnableConfig

from configs.settings import settings


class AsyncRedisSaver:
    """
    基于 RedisStack 的 LangGraph Checkpointer。
    用于保存对话状态、支持多轮会话记忆。
    使用 RedisJSON 模块存储结构化数据。

    实现 LangGraph BaseCheckpointSaver 接口。
    """

    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.REDIS_URL
        self._redis = None

    async def _get_redis(self):
        """获取 Redis 连接（懒加载）"""
        if self._redis is None:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(
                self.redis_url,
                decode_responses=True,
                encoding="utf-8",
            )
            # 测试连接
            await self._redis.ping()
            logger.info(f"RedisStack 短期记忆连接成功: {self.redis_url}")
        return self._redis

    def _make_key(self, thread_id: str, checkpoint_ns: str = "") -> str:
        """生成 Redis key"""
        return f"checkpoint:{thread_id}:{checkpoint_ns}"

    def _make_writes_key(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> str:
        """生成 pending writes 的 key"""
        return f"writes:{thread_id}:{checkpoint_ns}:{checkpoint_id}"

    # ============================================================
    # LangGraph Checkpointer 接口实现
    # ============================================================

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: dict,
        metadata: dict,
        new_versions: dict,
    ) -> RunnableConfig:
        """保存 checkpoint 到 Redis"""
        redis = await self._get_redis()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        key = self._make_key(thread_id, checkpoint_ns)

        # 序列化 checkpoint 和 metadata
        data = {
            "checkpoint": self._serialize_checkpoint(checkpoint),
            "metadata": dict(metadata) if metadata else {},
            "versions": dict(new_versions),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # 存储为 JSON 字符串
        await redis.set(key, json.dumps(data, ensure_ascii=False))

        # 设置过期时间（7天）
        await redis.expire(key, 7 * 24 * 3600)

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint.get("id", ""),
            }
        }

    async def aget_tuple(self, config: RunnableConfig) -> Optional[dict]:
        """获取最新 checkpoint"""
        redis = await self._get_redis()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        key = self._make_key(thread_id, checkpoint_ns)
        raw = await redis.get(key)
        if raw is None:
            return None

        data = json.loads(raw)
        checkpoint = self._deserialize_checkpoint(data["checkpoint"])

        return {
            "config": {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint.get("id", ""),
                }
            },
            "checkpoint": checkpoint,
            "metadata": data.get("metadata"),
            "created_at": data.get("updated_at"),
            "parent_config": None,
        }

    async def alist(self, config: RunnableConfig) -> AsyncIterator[dict]:
        """列出所有 checkpoint（简化实现：只返回最新一个）"""
        result = await self.aget_tuple(config)
        if result:
            yield result

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: list[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """保存中间写入"""
        redis = await self._get_redis()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id", "")

        key = self._make_writes_key(thread_id, checkpoint_ns, checkpoint_id)
        data = [{"channel": ch, "value": str(val)} for ch, val in writes]
        await redis.set(key, json.dumps(data, ensure_ascii=False))
        await redis.expire(key, 24 * 3600)

    # ============================================================
    # 序列化 / 反序列化
    # ============================================================

    def _serialize_checkpoint(self, checkpoint: dict) -> dict:
        """序列化 checkpoint 为可 JSON 化的 dict"""
        channel_values = {}
        for k, v in checkpoint.get("channel_values", {}).items():
            if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                channel_values[k] = v
            else:
                channel_values[k] = str(v)

        return {
            "id": checkpoint.get("id", ""),
            "ts": checkpoint.get("ts", ""),
            "channel_values": channel_values,
            "channel_versions": dict(checkpoint.get("channel_versions", {})),
            "versions_seen": {
                k: dict(v) if isinstance(v, dict) else {}
                for k, v in checkpoint.get("versions_seen", {}).items()
            },
            "pending_sends": [],
        }

    def _deserialize_checkpoint(self, data: dict) -> dict:
        """反序列化 checkpoint"""
        return {
            "id": data.get("id", ""),
            "ts": data.get("ts", ""),
            "channel_values": data.get("channel_values", {}),
            "channel_versions": data.get("channel_versions", {}),
            "versions_seen": data.get("versions_seen", {}),
            "pending_sends": [],
        }

    # ============================================================
    # 会话历史便捷方法
    # ============================================================

    async def save_conversation_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ):
        """保存一轮对话到 Redis（用于会话上下文快速读取）"""
        redis = await self._get_redis()
        key = f"conversation:{session_id}"
        turn = {
            "user": user_message,
            "assistant": assistant_message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await redis.rpush(key, json.dumps(turn, ensure_ascii=False))
        await redis.expire(key, 24 * 3600)

    async def get_conversation_history(
        self, session_id: str, limit: int = 20
    ) -> list[dict]:
        """获取会话历史"""
        redis = await self._get_redis()
        key = f"conversation:{session_id}"
        raw_list = await redis.lrange(key, -limit, -1)
        return [json.loads(r) for r in raw_list]

    # ============================================================
    # 全量状态持久化（用于 LangGraph 对话状态跨轮次/跨重启恢复）
    # ============================================================

    async def save_full_state(self, session_id: str, state: dict) -> None:
        """
        保存完整的对话状态到 Redis。
        用于服务重启后恢复多轮问诊状态。
        """
        redis = await self._get_redis()
        key = f"full_state:{session_id}"

        # 序列化状态（处理不可 JSON 序列化的字段）
        serializable = {}
        for k, v in state.items():
            if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                serializable[k] = v
            else:
                # 尝试序列化，失败则转字符串
                try:
                    json.dumps(v)
                    serializable[k] = v
                except (TypeError, ValueError):
                    serializable[k] = str(v)

        await redis.set(key, json.dumps(serializable, ensure_ascii=False, default=str))
        await redis.expire(key, 7 * 24 * 3600)  # 7天过期
        logger.debug(f"[Redis] 状态已保存: session={session_id}, keys={list(serializable.keys())}")

    async def load_full_state(self, session_id: str) -> Optional[dict]:
        """
        从 Redis 加载完整的对话状态。
        服务重启后调用此方法恢复状态。
        """
        redis = await self._get_redis()
        key = f"full_state:{session_id}"
        raw = await redis.get(key)
        if raw is None:
            return None

        try:
            state = json.loads(raw)
            logger.info(f"[Redis] 状态已恢复: session={session_id}")
            return state
        except json.JSONDecodeError as e:
            logger.warning(f"[Redis] 状态反序列化失败: {e}")
            return None

    async def close(self):
        """关闭 Redis 连接"""
        if self._redis:
            await self._redis.close()
            self._redis = None
            logger.info("RedisStack 短期记忆连接已关闭")


# 全局单例
redis_saver = AsyncRedisSaver()


# ============================================================
# 单独调试入口
# ============================================================

if __name__ == "__main__":
    """单独测试 Redis 短期记忆"""
    import asyncio

    async def test_redis():
        saver = AsyncRedisSaver()

        # 测试保存对话
        await saver.save_conversation_turn(
            session_id="test_sess_001",
            user_message="我头疼三天了",
            assistant_message="请问头痛的具体位置在哪里？",
        )
        print("对话保存成功")

        # 测试读取对话历史
        history = await saver.get_conversation_history("test_sess_001")
        print(f"对话历史: {history}")

        # 测试 checkpoint 保存
        config = {
            "configurable": {
                "thread_id": "test_thread_001",
                "checkpoint_ns": "",
            }
        }
        checkpoint = {
            "id": "cp_001",
            "ts": datetime.now(timezone.utc).isoformat(),
            "channel_values": {"messages": ["test"]},
            "channel_versions": {},
            "versions_seen": {},
        }
        result = await saver.aput(config, checkpoint, {"source": "test"}, {})
        print(f"Checkpoint 保存结果: {result}")

        # 测试 checkpoint 读取
        loaded = await saver.aget_tuple(config)
        print(f"Checkpoint 读取结果: {loaded is not None}")

        await saver.close()

    asyncio.run(test_redis())
