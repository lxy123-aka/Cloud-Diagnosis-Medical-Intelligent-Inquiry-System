"""执行 PostgreSQL consultations 表创建"""
import asyncio
import asyncpg


async def main():
    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        user="postgres",
        password="cdmiis_password",
        database="cloud_diagnosis",
    )

    # 检查现有表
    tables = await conn.fetch(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
    )
    print(f"现有表: {[t['tablename'] for t in tables]}")

    # 创建 consultations 表
    create_sql = """
    CREATE TABLE IF NOT EXISTS consultations (
        consultation_id     VARCHAR(64)   PRIMARY KEY,
        user_id             VARCHAR(64)   NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
        session_id          VARCHAR(64)   NOT NULL,
        record_id           VARCHAR(64)   REFERENCES medical_records(record_id) ON DELETE SET NULL,
        intent              VARCHAR(32)   DEFAULT '',
        intent_confidence   DECIMAL(5,4)  DEFAULT 0.0000,
        inquiry_rounds      INTEGER       DEFAULT 0,
        max_rounds          INTEGER       DEFAULT 10,
        converged           BOOLEAN       DEFAULT FALSE,
        final_diagnosis     TEXT          DEFAULT '',
        diagnosis_confidence DECIMAL(5,4) DEFAULT 0.0000,
        standardized_symptoms JSONB       DEFAULT '[]'::jsonb,
        disease_candidates  JSONB         DEFAULT '[]'::jsonb,
        conversation_log    JSONB         DEFAULT '[]'::jsonb,
        has_image           BOOLEAN       DEFAULT FALSE,
        image_paths         JSONB         DEFAULT '[]'::jsonb,
        image_analysis      TEXT          DEFAULT '',
        started_at          TIMESTAMP     DEFAULT NOW(),
        ended_at            TIMESTAMP,
        duration_seconds    INTEGER,
        created_at          TIMESTAMP     DEFAULT NOW(),
        updated_at          TIMESTAMP     DEFAULT NOW()
    );
    """
    await conn.execute(create_sql)
    print("✅ consultations 表创建成功")

    # 创建索引
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_consultations_user_created ON consultations(user_id, created_at DESC);",
        "CREATE INDEX IF NOT EXISTS idx_consultations_session ON consultations(session_id);",
        "CREATE INDEX IF NOT EXISTS idx_consultations_record ON consultations(record_id);",
        "CREATE INDEX IF NOT EXISTS idx_consultations_diagnosis ON consultations(final_diagnosis) WHERE final_diagnosis != '';",
    ]
    for idx_sql in indexes:
        await conn.execute(idx_sql)
    print("✅ 索引创建成功")

    # 验证
    tables = await conn.fetch(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
    )
    print(f"当前表: {[t['tablename'] for t in tables]}")

    # 查看 consultations 列信息
    cols = await conn.fetch(
        "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'consultations' ORDER BY ordinal_position"
    )
    print(f"\nconsultations 表结构 ({len(cols)} 列):")
    for c in cols:
        print(f"  {c['column_name']:30s} {c['data_type']}")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
