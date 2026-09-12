"""
utils/rag_retrieval.py
======================
统一医学 RAG 向量检索服务。

封装 Milvus 向量检索逻辑，为所有 Agent/Worker 提供统一的知识检索接口。
支持三种检索模式：
  1. retrieve_knowledge  —— 从 medical_knowledge 集合检索医学知识
  2. retrieve_diseases   —— 从 disease_collection 集合检索疾病信息
  3. retrieve_symptoms   —— 从 symptom_collection 集合检索标准症状
  4. hybrid_retrieve     —— 混合检索，同时查询三个集合并拼接为上下文

底层使用 utils/embedding_client.py 做向量化，连接 Milvus 做 ANN 检索。
"""

from __future__ import annotations
from loguru import logger
from configs.settings import settings


class MedicalRAGRetriever:
    """
    医学 RAG 检索器。
    封装 Milvus 向量检索，为各 Agent/Worker 提供统一的知识检索接口。
    """

    def __init__(self):
        self._client = None

    def _get_client(self):
        """延迟初始化 Milvus 客户端（单例）"""
        if self._client is None:
            from pymilvus import MilvusClient
            uri = f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
            self._client = MilvusClient(uri=uri)
        return self._client

    async def _encode(self, text: str) -> list[float]:
        """使用统一 Embedding 客户端编码文本"""
        from utils.embedding_client import embedding_client
        return await embedding_client.encode_single(text, normalize=True)

    async def retrieve_knowledge(
        self, query: str, top_k: int = 5, min_score: float = 0.3
    ) -> list[dict]:
        """
        从 medical_knowledge 集合检索医学知识。

        :param query: 查询文本
        :param top_k: 返回前 k 条结果
        :param min_score: 最低相似度阈值
        :return: [{"title": str, "content": str, "source": str, "score": float}]
        """
        try:
            query_vector = await self._encode(query)
            client = self._get_client()

            results = client.search(
                collection_name="medical_knowledge",
                data=[query_vector],
                limit=top_k,
                output_fields=["title", "content", "source"],
            )

            matches = []
            for hit in results[0]:
                score = hit.get("distance", 0)
                if score < min_score:
                    continue
                matches.append({
                    "title": hit["entity"].get("title", ""),
                    "content": hit["entity"].get("content", ""),
                    "source": hit["entity"].get("source", ""),
                    "score": round(score, 4),
                })

            logger.debug(f"[RAG] 知识检索 '{query[:30]}...' → {len(matches)} 条结果")
            return matches

        except Exception as e:
            logger.warning(f"[RAG] 知识检索失败（Milvus可能未启动）: {e}")
            return []

    async def retrieve_diseases(
        self, query: str, top_k: int = 5, min_score: float = 0.3
    ) -> list[dict]:
        """
        从 disease_collection 集合检索疾病信息。

        :param query: 查询文本（通常是症状描述）
        :param top_k: 返回前 k 条结果
        :param min_score: 最低相似度阈值
        :return: [{"disease_name": str, "icd_code": str, "description": str, "score": float}]
        """
        try:
            query_vector = await self._encode(query)
            client = self._get_client()

            results = client.search(
                collection_name="disease_collection",
                data=[query_vector],
                limit=top_k,
                output_fields=["disease_name", "icd_code", "description"],
            )

            matches = []
            for hit in results[0]:
                score = hit.get("distance", 0)
                if score < min_score:
                    continue
                matches.append({
                    "disease_name": hit["entity"].get("disease_name", ""),
                    "icd_code": hit["entity"].get("icd_code", ""),
                    "description": hit["entity"].get("description", ""),
                    "score": round(score, 4),
                })

            logger.debug(f"[RAG] 疾病检索 '{query[:30]}...' → {len(matches)} 条结果")
            return matches

        except Exception as e:
            logger.warning(f"[RAG] 疾病检索失败（Milvus可能未启动）: {e}")
            return []

    async def retrieve_symptoms(
        self, query: str, top_k: int = 5, min_score: float = 0.4
    ) -> list[dict]:
        """
        从 symptom_collection 集合检索标准症状。

        :param query: 查询文本（口语化症状描述）
        :param top_k: 返回前 k 条结果
        :param min_score: 最低相似度阈值
        :return: [{"symptom_name": str, "category": str, "score": float}]
        """
        try:
            query_vector = await self._encode(query)
            client = self._get_client()

            results = client.search(
                collection_name="symptom_collection",
                data=[query_vector],
                limit=top_k,
                output_fields=["symptom_name", "category"],
            )

            matches = []
            for hit in results[0]:
                score = hit.get("distance", 0)
                if score < min_score:
                    continue
                matches.append({
                    "symptom_name": hit["entity"].get("symptom_name", ""),
                    "category": hit["entity"].get("category", ""),
                    "score": round(score, 4),
                })

            logger.debug(f"[RAG] 症状检索 '{query[:30]}...' → {len(matches)} 条结果")
            return matches

        except Exception as e:
            logger.warning(f"[RAG] 症状检索失败（Milvus可能未启动）: {e}")
            return []

    async def hybrid_retrieve(self, query: str, top_k: int = 5) -> str:
        """
        混合检索：同时查询三个集合，将结果拼接为可用于 LLM prompt 的上下文字符串。

        :param query: 查询文本
        :param top_k: 每个集合返回前 k 条结果
        :return: 拼接后的上下文文本（如无结果返回空字符串）
        """
        # 并行检索三个集合
        knowledge_results, disease_results, symptom_results = await _parallel_retrieve(
            query, top_k
        )

        # 拼接为结构化上下文
        context_parts: list[str] = []

        if knowledge_results:
            context_parts.append("【医学知识】")
            for item in knowledge_results:
                title = item.get("title", "")
                content = item.get("content", "")
                source = item.get("source", "")
                entry = f"- {title}：{content}"
                if source:
                    entry += f"（来源：{source}）"
                context_parts.append(entry)

        if disease_results:
            context_parts.append("\n【相关疾病】")
            for item in disease_results:
                name = item.get("disease_name", "")
                desc = item.get("description", "")
                icd = item.get("icd_code", "")
                entry = f"- {name}"
                if icd:
                    entry += f"（{icd}）"
                if desc:
                    entry += f"：{desc}"
                context_parts.append(entry)

        if symptom_results:
            context_parts.append("\n【相关症状】")
            for item in symptom_results:
                name = item.get("symptom_name", "")
                cat = item.get("category", "")
                entry = f"- {name}"
                if cat:
                    entry += f"（{cat}）"
                context_parts.append(entry)

        context = "\n".join(context_parts)
        if context:
            logger.info(
                f"[RAG] 混合检索完成: 知识{len(knowledge_results)}条 "
                f"疾病{len(disease_results)}条 症状{len(symptom_results)}条"
            )
        return context


async def _parallel_retrieve(
    query: str, top_k: int
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    并行执行三个集合的检索。
    使用 asyncio.gather 并发请求，减少总延迟。
    单个集合失败不影响其他集合的检索结果。
    """
    import asyncio

    retriever = MedicalRAGRetriever()

    try:
        knowledge_task = retriever.retrieve_knowledge(query, top_k=top_k)
        disease_task = retriever.retrieve_diseases(query, top_k=top_k)
        symptom_task = retriever.retrieve_symptoms(query, top_k=top_k)

        results = await asyncio.gather(
            knowledge_task, disease_task, symptom_task, return_exceptions=True
        )

        # 逐个检查异常，记录失败集合但不中断整体流程
        knowledge_results = results[0] if not isinstance(results[0], Exception) else []
        disease_results = results[1] if not isinstance(results[1], Exception) else []
        symptom_results = results[2] if not isinstance(results[2], Exception) else []

        # 记录失败集合的具体信息
        failed_collections = []
        if isinstance(results[0], Exception):
            failed_collections.append(f"medical_knowledge({results[0]})")
        if isinstance(results[1], Exception):
            failed_collections.append(f"disease_collection({results[1]})")
        if isinstance(results[2], Exception):
            failed_collections.append(f"symptom_collection({results[2]})")
        if failed_collections:
            logger.warning(f"[RAG] 部分集合检索失败: {', '.join(failed_collections)}")

        return knowledge_results, disease_results, symptom_results

    except Exception as e:
        logger.warning(f"[RAG] 并行检索异常: {e}")
        return [], [], []


# ============================================================
# 全局单例（避免重复创建 Milvus 连接）
# ============================================================

rag_retriever = MedicalRAGRetriever()


# ============================================================
# 单独调试入口
# ============================================================

if __name__ == "__main__":
    """测试 RAG 检索"""
    import asyncio

    async def test_rag():
        retriever = MedicalRAGRetriever()

        # 测试知识检索
        print("=== 知识检索测试 ===")
        results = await retriever.retrieve_knowledge("头痛怎么办", top_k=3)
        for r in results:
            print(f"  [{r['score']:.2f}] {r['title']}: {r['content'][:80]}...")

        # 测试疾病检索
        print("\n=== 疾病检索测试 ===")
        results = await retriever.retrieve_diseases("腹痛 腹泻 恶心", top_k=3)
        for r in results:
            print(f"  [{r['score']:.2f}] {r['disease_name']}: {r['description'][:80]}...")

        # 测试混合检索
        print("\n=== 混合检索测试 ===")
        context = await retriever.hybrid_retrieve("高血压患者饮食注意事项", top_k=3)
        print(context[:500] if context else "（无结果，Milvus可能未启动或知识库为空）")

    asyncio.run(test_rag())
