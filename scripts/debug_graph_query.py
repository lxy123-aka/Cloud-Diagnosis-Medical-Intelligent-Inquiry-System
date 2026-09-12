"""诊断知识图谱查询为什么总是失败"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
from neo4j import AsyncGraphDatabase
from configs.settings import settings


async def run_query(session, query, params=None):
    """执行查询并返回所有记录"""
    result = await session.run(query, params or {})
    records = await result.data()  # 直接获取所有数据
    return records


async def main():
    driver = AsyncGraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )

    async with driver.session() as session:
        # 1. 检查所有节点标签和数量
        labels_data = await run_query(session, "CALL db.labels()")
        print(f"1. 所有节点标签: {[d.get('label') for d in labels_data]}")

        # 2. 检查各标签节点数
        for label in ['Symptom', 'Disease', 'Drug', 'Department']:
            data = await run_query(session, f"MATCH (n:{label}) RETURN count(n) AS cnt")
            print(f"   {label}: {data[0]['cnt'] if data else 0} 个节点")

        # 3. 查看 Disease 节点前 5 个及其属性
        diseases = await run_query(session,
            "MATCH (d:Disease) RETURN d.name AS name, d.typical_symptoms AS symptoms LIMIT 5"
        )
        print(f"\n2. Disease 节点前 5 个:")
        for d in diseases:
            print(f"   {d.get('name')} - symptoms: {str(d.get('symptoms', []))[:80]}")

        # 4. 查看 Symptom 节点前 10 个
        symptoms = await run_query(session,
            "MATCH (s:Symptom) RETURN s.name AS name LIMIT 10"
        )
        print(f"\n3. Symptom 节点前 10 个:")
        for s in symptoms:
            print(f"   - {s.get('name')}")

        # 5. 检查 HAS_SYMPTOM 关系指向哪里
        has_symptom = await run_query(session,
            "MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s:Symptom) "
            "RETURN d.name AS disease, s.name AS symptom LIMIT 10"
        )
        print(f"\n4. HAS_SYMPTOM 关系示例:")
        for r in has_symptom:
            print(f"   {r.get('disease')} → {r.get('symptom')}")
        if not has_symptom:
            print("   *** 空！HAS_SYMPTOM 关系存在但 Disease→Symptom 路径不通 ***")

        # 6. 检查 HAS_SYMPTOM 指向的目标标签
        targets = await run_query(session,
            "MATCH (d:Disease)-[:HAS_SYMPTOM]->(n) "
            "RETURN DISTINCT labels(n) AS lbls LIMIT 5"
        )
        print(f"\n5. HAS_SYMPTOM 目标节点标签:")
        for t in targets:
            print(f"   {t.get('lbls')}")

        # 7. 模拟诊断引擎查询
        diag = await run_query(session,
            """
            MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
            WHERE s.name IN $symptom_names
            WITH d, COLLECT(s.name) AS matched, COUNT(s) AS cnt
            RETURN d.name AS disease, matched, cnt
            ORDER BY cnt DESC LIMIT 10
            """,
            {"symptom_names": ["腹泻", "腹痛", "恶心", "呕吐", "发热"]}
        )
        print(f"\n6. 诊断引擎模拟查询:")
        for r in diag:
            print(f"   {r.get('disease')}: 匹配 {r.get('cnt')} 个症状 {r.get('matched')}")
        if not diag:
            print("   *** 未找到匹配！***")

        # 8. 检查常见症状是否在图谱中
        common = await run_query(session,
            "MATCH (s:Symptom) WHERE s.name IN $names RETURN s.name AS name",
            {"names": ["腹泻", "腹痛", "腹胀", "恶心", "呕吐", "发热", "头痛", "咳嗽"]}
        )
        print(f"\n7. 常见症状查询: {[r.get('name') for r in common]}")

    await driver.close()


asyncio.run(main())
