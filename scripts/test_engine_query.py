"""直接测试 diagnosis_engine 的图谱查询是否工作"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
from pipeline.diagnosis_engine import DiagnosisEngine


async def main():
    engine = DiagnosisEngine()

    # 直接调用图谱查询
    symptom_names = ["腹泻", "腹痛", "恶心", "呕吐"]
    print(f"查询症状: {symptom_names}")

    diseases = await engine._query_diseases_from_graph(symptom_names)
    print(f"\n图谱查询结果: {len(diseases)} 条")
    for d in diseases[:5]:
        print(f"  {d.get('disease_name')}: 匹配 {d.get('match_count')} 个症状 {d.get('matched_symptoms')}")

    if not diseases:
        print("\n*** 图谱查询返回空！测试 result.data() 是否可用 ***")

        # 直接用 Neo4j 异步驱动测试
        from neo4j import AsyncGraphDatabase
        from configs.settings import settings

        driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
        async with driver.session() as session:
            result = await session.run(
                "MATCH (s:Symptom) WHERE s.name = '腹泻' RETURN s.name AS name"
            )
            # 方式1: await result.data()
            try:
                data = await result.data()
                print(f"  result.data() = {data}")
            except Exception as e:
                print(f"  result.data() 失败: {type(e).__name__}: {e}")

        async with driver.session() as session:
            result = await session.run(
                "MATCH (s:Symptom) WHERE s.name = '腹泻' RETURN s.name AS name"
            )
            # 方式2: fetch + record.data()
            try:
                records = await result.fetch(n=-1)
                print(f"  result.fetch(n=-1) 返回 {len(records)} 条")
                for r in records:
                    print(f"    record.data() = {r.data()}")
            except Exception as e:
                print(f"  fetch(n=-1) 失败: {type(e).__name__}: {e}")

        async with driver.session() as session:
            result = await session.run(
                "MATCH (s:Symptom) WHERE s.name = '腹泻' RETURN s.name AS name"
            )
            # 方式3: 直接迭代
            try:
                records = []
                async for record in result:
                    records.append(record.data())
                print(f"  async for 迭代返回 {len(records)} 条: {records}")
            except Exception as e:
                print(f"  async for 失败: {type(e).__name__}: {e}")

        await driver.close()

    # 测试共现关系查询
    co = await engine._get_co_occurrences(symptom_names)
    print(f"\n共现关系查询: {len(co)} 条")

    await engine.close()


asyncio.run(main())
