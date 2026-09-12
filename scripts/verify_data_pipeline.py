"""
scripts/verify_data_pipeline.py
================================
数据流水线端到端验证脚本。

对典型症状输入（"头痛 发热 恶心"）跑完整三层流水线，
输出每层命中情况，验证 Neo4j 匹配 + Milvus 向量召回均正常。

运行方式：
  python scripts/verify_data_pipeline.py
"""

from __future__ import annotations
import sys
import asyncio
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger


async def verify_neo4j():
    """验证 Neo4j 连接和数据完整性"""
    print("\n" + "=" * 60)
    print("  1. Neo4j 数据验证")
    print("=" * 60)

    from neo4j import AsyncGraphDatabase
    from configs.settings import settings

    driver = AsyncGraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )

    try:
        async with driver.session() as session:
            # 统计节点
            result = await session.run("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt")
            records = await result.data()
            print("\n  节点统计:")
            for r in records:
                print(f"    {r['label']}: {r['cnt']} 个")

            # 统计关系
            result = await session.run(
                "MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS cnt"
            )
            records = await result.data()
            print("\n  关系统计:")
            for r in records:
                print(f"    {r['rel_type']}: {r['cnt']} 条")

            # 测试症状查询
            test_symptoms = ["头痛", "发热", "恶心", "腹泻", "腹痛"]
            print(f"\n  症状精确匹配测试: {test_symptoms}")
            for name in test_symptoms:
                result = await session.run(
                    "MATCH (s:Symptom) WHERE s.name = $name RETURN s.name AS name, s.aliases AS aliases",
                    {"name": name},
                )
                records = await result.data()
                if records:
                    r = records[0]
                    print(f"    ✅ '{name}' → 匹配成功 (别名: {r.get('aliases', [])})")
                else:
                    print(f"    ❌ '{name}' → 未匹配")

            # 测试疾病-症状关系查询
            print("\n  疾病-症状关系查询测试 (输入: ['腹泻', '腹痛', '恶心', '呕吐']):")
            result = await session.run(
                """
                MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
                WHERE s.name IN $symptoms
                RETURN d.name AS disease, count(s) AS match_count,
                       collect(s.name) AS matched_symptoms
                ORDER BY match_count DESC
                LIMIT 10
                """,
                {"symptoms": ["腹泻", "腹痛", "恶心", "呕吐"]},
            )
            records = await result.data()
            if records:
                for r in records:
                    print(
                        f"    {r['disease']}: 匹配 {r['match_count']} 个症状 "
                        f"{r['matched_symptoms']}"
                    )
            else:
                print("    ❌ 无匹配疾病")

    finally:
        await driver.close()


async def verify_symptom_pipeline():
    """验证症状标准化三层流水线"""
    print("\n" + "=" * 60)
    print("  2. 症状标准化流水线验证")
    print("=" * 60)

    from pipeline.symptom_normalize import SymptomNormalizePipeline

    pipeline = SymptomNormalizePipeline()

    test_cases = [
        "我头疼，还有点发热",
        "肚子疼，拉了三次肚子，还有点恶心",
        "嗓子疼，吞咽的时候特别疼",
        "没有不舒服",  # 否定测试
        "昨天吃了火锅",  # 非症状测试
    ]

    for text in test_cases:
        print(f"\n  输入: '{text}'")
        try:
            result = await pipeline.run(text)
            if result:
                for s in result:
                    print(
                        f"    → {s.get('standard_name', '')} "
                        f"(置信度={s.get('confidence', 0):.2f}, "
                        f"来源={s.get('source', '')})"
                    )
            else:
                print("    → (无症状提取)")
        except Exception as e:
            print(f"    ❌ 错误: {e}")

    await pipeline.close()


async def verify_milvus():
    """验证 Milvus 集合状态"""
    print("\n" + "=" * 60)
    print("  3. Milvus 向量库验证")
    print("=" * 60)

    try:
        from pymilvus import MilvusClient
        from configs.settings import settings

        uri = f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
        client = MilvusClient(uri=uri)

        collections = ["symptom_collection", "disease_collection", "medical_knowledge"]
        for col in collections:
            try:
                if client.has_collection(col):
                    stats = client.get_collection_stats(col)
                    row_count = stats.get("row_count", 0)
                    print(f"  ✅ {col}: {row_count} 条数据")
                else:
                    print(f"  ❌ {col}: 集合不存在")
            except Exception as e:
                print(f"  ❌ {col}: 查询失败 - {e}")

        # 测试向量搜索
        print("\n  向量搜索测试 ('头痛'):")
        try:
            from utils.embedding_client import embedding_client
            query_vec = await embedding_client.encode_single("头痛", normalize=True)
            results = client.search(
                collection_name="symptom_collection",
                data=[query_vec],
                limit=5,
                output_fields=["symptom_name", "category"],
            )
            for hit in results[0]:
                name = hit["entity"]["symptom_name"]
                score = hit["distance"]
                print(f"    {name}: score={score:.4f}")
        except Exception as e:
            print(f"    ❌ 搜索失败: {e}")

    except Exception as e:
        print(f"  ❌ Milvus 连接失败: {e}")


async def verify_redis():
    """验证 Redis 连接"""
    print("\n" + "=" * 60)
    print("  4. Redis 连接验证")
    print("=" * 60)

    try:
        from memory.short_memory import redis_saver

        # 测试连接
        r = await redis_saver._get_redis()
        info = await r.info("server")
        print(f"  ✅ Redis 连接成功: {info.get('redis_version', '?')}")

        # 测试状态保存/恢复
        test_state = {
            "user_id": "test_verify",
            "session_id": "verify_session",
            "inquiry_round": 2,
            "standardized_symptoms": [{"standard_name": "头痛", "confidence": 0.9}],
            "messages": [],
        }
        await redis_saver.save_full_state("verify_session", test_state)
        loaded = await redis_saver.load_full_state("verify_session")
        if loaded and loaded.get("inquiry_round") == 2:
            print("  ✅ 状态保存/恢复测试通过")
        else:
            print("  ❌ 状态恢复数据不匹配")

        # 清理测试数据
        await r.delete("full_state:verify_session")

    except Exception as e:
        print(f"  ❌ Redis 验证失败: {e}")


async def main():
    """运行所有验证"""
    print("\n" + "🔍" + " CDMIIS 数据流水线验证 " + "🔍")
    print("=" * 60)

    await verify_neo4j()
    await verify_symptom_pipeline()
    await verify_milvus()
    await verify_redis()

    print("\n" + "=" * 60)
    print("  验证完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
