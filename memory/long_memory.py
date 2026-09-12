"""
memory/long_memory.py
=====================
长期病历记忆模块 —— 基于 PostgreSQL。

功能：
  1. 用户病历、问诊记录增删改查
  2. 问诊启动时自动加载历史病历写入 state
  3. 使用 asyncpg 异步连接池
  4. 支持按症状搜索历史病历
"""

from __future__ import annotations
import uuid
from datetime import datetime
from typing import Optional
from loguru import logger

from configs.settings import settings


class LongTermMemory:
    """长期病历记忆管理器"""

    def __init__(self):
        self._pool = None  # asyncpg 连接池

    async def initialize(self):
        """初始化数据库连接池和表"""
        import asyncpg

        self._pool = await asyncpg.create_pool(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            database=settings.POSTGRES_DB,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            min_size=2,
            max_size=10,
        )
        logger.info("PostgreSQL 连接池创建成功")

        # 自动建表
        await self._create_tables()

    async def _create_tables(self):
        """创建所需的数据库表"""
        async with self._pool.acquire() as conn:
            # 用户表
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id VARCHAR(64) PRIMARY KEY,
                    phone VARCHAR(20) UNIQUE NOT NULL,
                    name VARCHAR(64) DEFAULT '',
                    gender VARCHAR(10) DEFAULT '',
                    birth_date VARCHAR(20) DEFAULT '',
                    allergies JSONB DEFAULT '[]',
                    chronic_diseases JSONB DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)

            # 病历表
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS medical_records (
                    record_id VARCHAR(64) PRIMARY KEY,
                    user_id VARCHAR(64) NOT NULL REFERENCES users(user_id),
                    session_id VARCHAR(64) NOT NULL,
                    chief_complaint TEXT DEFAULT '',
                    symptoms JSONB DEFAULT '[]',
                    disease_candidates JSONB DEFAULT '[]',
                    final_diagnosis TEXT DEFAULT '',
                    prescriptions JSONB DEFAULT '[]',
                    report_images JSONB DEFAULT '[]',
                    report_analysis TEXT DEFAULT '',
                    doctor_notes TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)

            # 创建索引
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_records_user_created
                ON medical_records(user_id, created_at DESC)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_records_user_diagnosis
                ON medical_records(user_id, final_diagnosis)
                WHERE final_diagnosis != ''
            """)

        logger.info("数据库表创建/验证完成")

    # ============================================================
    # 病历 CRUD
    # ============================================================

    async def save_record(self, record: dict) -> str:
        """
        保存病历记录。

        :param record: 病历数据字典
        :return: record_id
        """
        if not self._pool:
            await self.initialize()

        record_id = record.get("record_id") or str(uuid.uuid4())
        record["record_id"] = record_id

        import json
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO medical_records
                    (record_id, user_id, session_id, chief_complaint,
                     symptoms, disease_candidates, final_diagnosis,
                     prescriptions, report_images, report_analysis, doctor_notes)
                VALUES
                    ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7, $8::jsonb, $9::jsonb, $10, $11)
                ON CONFLICT (record_id) DO UPDATE SET
                    symptoms = EXCLUDED.symptoms,
                    disease_candidates = EXCLUDED.disease_candidates,
                    final_diagnosis = EXCLUDED.final_diagnosis,
                    prescriptions = EXCLUDED.prescriptions,
                    report_analysis = EXCLUDED.report_analysis,
                    updated_at = NOW()
                """,
                record_id,
                record.get("user_id", ""),
                record.get("session_id", ""),
                record.get("chief_complaint", ""),
                json.dumps(record.get("symptoms", []), ensure_ascii=False),
                json.dumps(record.get("disease_candidates", []), ensure_ascii=False),
                record.get("final_diagnosis", ""),
                json.dumps(record.get("prescriptions", []), ensure_ascii=False),
                json.dumps(record.get("report_images", []), ensure_ascii=False),
                record.get("report_analysis", ""),
                record.get("doctor_notes", ""),
            )
        logger.info(f"病历保存成功: {record_id}")
        return record_id

    async def get_user_records(self, user_id: str, limit: int = 10) -> list[dict]:
        """获取用户历史病历"""
        if not self._pool:
            await self.initialize()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT record_id, session_id, chief_complaint, symptoms,
                       final_diagnosis, prescriptions, created_at
                FROM medical_records
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                user_id,
                limit,
            )
            return [dict(row) for row in rows]

    async def get_recent_diagnosis(self, user_id: str) -> Optional[dict]:
        """获取用户最近的诊断结果"""
        if not self._pool:
            await self.initialize()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT final_diagnosis, symptoms, prescriptions
                FROM medical_records
                WHERE user_id = $1 AND final_diagnosis != ''
                ORDER BY created_at DESC
                LIMIT 1
                """,
                user_id,
            )
            return dict(row) if row else None

    async def delete_record(self, record_id: str) -> bool:
        """删除病历记录"""
        if not self._pool:
            await self.initialize()

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM medical_records WHERE record_id = $1",
                record_id,
            )
            return "DELETE 1" in result

    async def update_record(self, record_id: str, updates: dict) -> bool:
        """更新病历记录"""
        if not self._pool:
            await self.initialize()

        import json
        set_clauses = []
        values = []
        idx = 1

        for key, value in updates.items():
            if key in ("symptoms", "disease_candidates", "prescriptions", "report_images"):
                set_clauses.append(f"{key} = ${idx}::jsonb")
                values.append(json.dumps(value, ensure_ascii=False))
            elif key in ("chief_complaint", "final_diagnosis", "report_analysis", "doctor_notes"):
                set_clauses.append(f"{key} = ${idx}")
                values.append(value)
            idx += 1

        if not set_clauses:
            return False

        set_clauses.append("updated_at = NOW()")
        values.append(record_id)

        query = f"UPDATE medical_records SET {', '.join(set_clauses)} WHERE record_id = ${idx}"

        async with self._pool.acquire() as conn:
            result = await conn.execute(query, *values)
            return "UPDATE 1" in result

    # ============================================================
    # 用户 CRUD
    # ============================================================

    async def save_user(self, user: dict) -> str:
        """保存或更新用户信息"""
        if not self._pool:
            await self.initialize()

        user_id = user.get("user_id") or str(uuid.uuid4())
        user["user_id"] = user_id

        import json
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO users (user_id, phone, name, gender, birth_date, allergies, chronic_diseases)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb)
                ON CONFLICT (user_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    gender = EXCLUDED.gender,
                    allergies = EXCLUDED.allergies,
                    chronic_diseases = EXCLUDED.chronic_diseases,
                    updated_at = NOW()
                """,
                user_id,
                user.get("phone", ""),
                user.get("name", ""),
                user.get("gender", ""),
                user.get("birth_date", ""),
                json.dumps(user.get("allergies", []), ensure_ascii=False),
                json.dumps(user.get("chronic_diseases", []), ensure_ascii=False),
            )
        logger.info(f"用户保存成功: {user_id}")
        return user_id

    async def get_user(self, user_id: str) -> Optional[dict]:
        """获取用户信息"""
        if not self._pool:
            await self.initialize()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM users WHERE user_id = $1", user_id
            )
            return dict(row) if row else None

    # ============================================================
    # 搜索与辅助
    # ============================================================

    async def search_records_by_symptom(self, symptom: str, limit: int = 5) -> list[dict]:
        """按症状搜索历史病历（用于辅助诊断参考）"""
        if not self._pool:
            await self.initialize()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT record_id, user_id, final_diagnosis, symptoms, created_at
                FROM medical_records
                WHERE symptoms::text ILIKE $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                f"%{symptom}%",
                limit,
            )
            return [dict(row) for row in rows]

    async def load_history_for_state(self, user_id: str) -> list[dict]:
        """
        问诊启动时加载用户历史病历，用于写入 state。
        返回最近10条病历摘要。
        """
        records = await self.get_user_records(user_id, limit=10)
        history = []
        for r in records:
            history.append({
                "record_id": r.get("record_id", ""),
                "visit_date": str(r.get("created_at", "")),
                "chief_complaint": r.get("chief_complaint", ""),
                "symptoms": r.get("symptoms", []),
                "diagnosis": r.get("final_diagnosis", ""),
            })
        return history

    async def close(self):
        """关闭数据库连接"""
        if self._pool:
            await self._pool.close()
            logger.info("PostgreSQL 连接池已关闭")


# 全局单例
long_term_memory = LongTermMemory()


# ============================================================
# 单独调试入口
# ============================================================

if __name__ == "__main__":
    """单独测试长期病历记忆"""
    import asyncio

    async def test_long_memory():
        mem = LongTermMemory()
        await mem.initialize()

        # 测试保存用户
        user_id = await mem.save_user({
            "phone": "13800138000",
            "name": "测试用户",
            "gender": "男",
            "allergies": ["青霉素"],
        })
        print(f"用户ID: {user_id}")

        # 测试保存病历
        record_id = await mem.save_record({
            "user_id": user_id,
            "session_id": "test_sess_001",
            "chief_complaint": "头痛三天",
            "symptoms": [{"standard_name": "头痛"}, {"standard_name": "发热"}],
            "final_diagnosis": "上呼吸道感染",
        })
        print(f"病历ID: {record_id}")

        # 测试查询
        records = await mem.get_user_records(user_id)
        print(f"病历数量: {len(records)}")

        # 测试加载历史
        history = await mem.load_history_for_state(user_id)
        print(f"历史病历: {history}")

        await mem.close()

    asyncio.run(test_long_memory())
