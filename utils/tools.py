"""
utils/tools.py
==============
Function Calling 工具注册模块。

功能：
  1. 封装图谱查询工具（Neo4j 症状匹配、疾病查询）
  2. 封装向量检索工具（Milvus 症状相似度检索）
  3. 封装药物查询工具
  4. 统一注册为 LangChain Tool 格式，供给 Agent 调用
  5. 支持后续扩展 RAG 知识库工具
"""

from __future__ import annotations
import json
from typing import Optional
from loguru import logger

from langchain_core.tools import tool


# ============================================================
# 图谱查询工具
# ============================================================

@tool
def search_symptom_in_knowledge_graph(symptom_name: str) -> str:
    """
    在 Neo4j 医学知识图谱中精确匹配症状。
    输入症状名称（可以是口语化描述），返回标准医学术语和关联信息。

    :param symptom_name: 症状名称
    :return: JSON 格式的匹配结果
    """
    try:
        from neo4j import AsyncGraphDatabase
        import asyncio
        from configs.settings import settings

        async def _query():
            driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            )
            async with driver.session() as session:
                result = await session.run(
                    """
                    MATCH (s:Symptom)
                    WHERE s.name = $name OR $name IN s.aliases
                    RETURN s.name AS name, s.snomed_code AS snomed_code,
                           s.category AS category, s.aliases AS synonyms
                    LIMIT 5
                    """,
                    {"name": symptom_name},
                )
                records = await result.data()
            await driver.close()
            return records

        results = asyncio.run(_query())
        return json.dumps(results, ensure_ascii=False)

    except Exception as e:
        logger.warning(f"[工具] 图谱症状查询失败: {e}")
        return json.dumps({"error": str(e), "results": []})


@tool
def query_diseases_by_symptoms(symptoms_json: str) -> str:
    """
    根据症状列表查询 Neo4j 知识图谱中的关联疾病。
    输入 JSON 格式的症状名称数组，返回候选疾病及匹配度。

    :param symptoms_json: JSON 格式症状列表，如 '["头痛", "发热"]'
    :return: JSON 格式的疾病列表
    """
    try:
        symptom_names = json.loads(symptoms_json)
        from neo4j import AsyncGraphDatabase
        import asyncio
        from configs.settings import settings

        async def _query():
            driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            )
            async with driver.session() as session:
                result = await session.run(
                    """
                    MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
                    WHERE s.name IN $symptom_names
                    WITH d, COLLECT(s.name) AS matched, COUNT(s) AS cnt
                    RETURN d.name AS disease_name, d.icd_code AS icd_code,
                           d.department AS department, matched, cnt
                    ORDER BY cnt DESC LIMIT 10
                    """,
                    {"symptom_names": symptom_names},
                )
                records = await result.data()
            await driver.close()
            return records

        results = asyncio.run(_query())
        return json.dumps(results, ensure_ascii=False)

    except Exception as e:
        logger.warning(f"[工具] 图谱疾病查询失败: {e}")
        return json.dumps({"error": str(e), "results": []})


# ============================================================
# 向量检索工具
# ============================================================

@tool
def search_similar_symptoms_vector(query_text: str, top_k: int = 5) -> str:
    """
    使用 Milvus 向量库进行症状语义相似度检索。
    输入口语化症状描述，返回语义最相似的标准症状列表。

    :param query_text: 症状描述文本
    :param top_k: 返回前 k 个最相似结果
    :return: JSON 格式的检索结果
    """
    try:
        from sentence_transformers import SentenceTransformer
        from pymilvus import MilvusClient
        from configs.settings import settings

        # 向量化
        model = SentenceTransformer(settings.EMBEDDING_MODEL)
        query_vec = model.encode([query_text], normalize_embeddings=True)[0].tolist()

        # Milvus 检索
        client = MilvusClient(uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
        results = client.search(
            collection_name="symptom_collection",
            data=[query_vec],
            limit=top_k,
            output_fields=["symptom_name", "category"],
        )

        matches = [
            {
                "symptom_name": hit["entity"]["symptom_name"],
                "category": hit["entity"]["category"],
                "score": round(hit["distance"], 4),
            }
            for hit in results[0]
        ]
        return json.dumps(matches, ensure_ascii=False)

    except Exception as e:
        logger.warning(f"[工具] 向量检索失败: {e}")
        return json.dumps({"error": str(e), "results": []})


# ============================================================
# 药物查询工具
# ============================================================

@tool
def query_drug_info(drug_name: str) -> str:
    """
    查询药品详细信息，包括适应症、用法用量、禁忌、副作用、相互作用。

    :param drug_name: 药品名称
    :return: JSON 格式的药品信息
    """
    from agent.worker.drug_agent import DRUG_KNOWLEDGE_BASE

    # 精确匹配
    if drug_name in DRUG_KNOWLEDGE_BASE:
        return json.dumps(DRUG_KNOWLEDGE_BASE[drug_name], ensure_ascii=False)

    # 模糊匹配
    for name, info in DRUG_KNOWLEDGE_BASE.items():
        if drug_name in name or name in drug_name:
            return json.dumps(info, ensure_ascii=False)
        for trade in info.get("trade_names", []):
            if trade in drug_name or drug_name in trade:
                return json.dumps(info, ensure_ascii=False)

    return json.dumps({"error": f"未找到药品: {drug_name}"})


# ============================================================
# RAG 知识库工具（预留接口）
# ============================================================

@tool
def search_medical_knowledge(query: str, top_k: int = 5) -> str:
    """
    从医学知识库中检索相关文献/指南内容（RAG 向量检索）。
    使用 Milvus 向量库检索 medical_knowledge 集合中的医学知识，
    包括症状鉴别、药物科普、疾病管理、检查指标解读等。

    :param query: 检索查询文本
    :param top_k: 返回前 k 条结果
    :return: JSON 格式的检索结果
    """
    try:
        import asyncio
        from utils.rag_retrieval import rag_retriever

        retriever = rag_retriever

        # 在同步工具中运行异步检索
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果已有事件循环在运行，创建任务
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    results = pool.submit(
                        asyncio.run, retriever.retrieve_knowledge(query, top_k=top_k)
                    ).result()
            else:
                results = loop.run_until_complete(
                    retriever.retrieve_knowledge(query, top_k=top_k)
                )
        except RuntimeError:
            results = asyncio.run(
                retriever.retrieve_knowledge(query, top_k=top_k)
            )

        if results:
            logger.info(f"[工具] RAG 知识检索 '{query[:30]}...' → {len(results)} 条结果")
            return json.dumps(results, ensure_ascii=False)
        else:
            return json.dumps({"message": "未检索到相关医学知识", "query": query, "results": []})

    except Exception as e:
        logger.warning(f"[工具] RAG 知识检索失败: {e}")
        return json.dumps({"error": str(e), "query": query, "results": []})


# ============================================================
# 工具注册表
# ============================================================

# 所有可用工具列表（供 Agent 绑定使用）
ALL_TOOLS = [
    search_symptom_in_knowledge_graph,
    query_diseases_by_symptoms,
    search_similar_symptoms_vector,
    query_drug_info,
    search_medical_knowledge,
]

# 工具名称 → 工具对象映射
TOOL_MAP = {t.name: t for t in ALL_TOOLS}


def get_tools_by_names(names: list[str]) -> list:
    """根据工具名称列表获取工具对象"""
    return [TOOL_MAP[name] for name in names if name in TOOL_MAP]


def get_all_tool_schemas() -> list[dict]:
    """获取所有工具的 OpenAI Function Calling Schema"""
    schemas = []
    for t in ALL_TOOLS:
        schemas.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.args_schema.schema() if hasattr(t, "args_schema") else {},
            },
        })
    return schemas


# ============================================================
# 单独调试入口
# ============================================================

if __name__ == "__main__":
    """单独测试工具注册"""
    print("=== 已注册工具 ===")
    for t in ALL_TOOLS:
        print(f"  - {t.name}: {t.description[:80]}...")

    print("\n=== 工具 Schema ===")
    schemas = get_all_tool_schemas()
    for s in schemas:
        print(f"  {s['function']['name']}")

    # 测试药物查询
    print("\n=== 测试药物查询 ===")
    result = query_drug_info.invoke({"drug_name": "布洛芬"})
    print(result[:200])
