"""
configs/milvus_init.py
======================
Milvus 向量库初始化脚本（使用 Qwen Embedding API 生成真实向量）。

功能：
  1. 删除旧集合，重新创建症状/疾病/知识三个向量集合
  2. 使用 Qwen Embedding API 为所有数据生成真实语义向量
  3. 插入 200 症状 + 200 疾病 + 100 条医学知识

运行方式：
  python configs/milvus_init.py
"""

from __future__ import annotations
import json
import asyncio
from loguru import logger

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import settings
from configs.neo4j_symptoms import SYMPTOMS
from configs.neo4j_diseases import DISEASES
from configs.milvus_knowledge import KNOWLEDGE_DATA


# ============================================================
# 批量 embedding 工具函数
# ============================================================

async def batch_embed(texts: list[str], batch_size: int = 6) -> list[list[float]]:
    """
    分批调用 Qwen Embedding API 生成向量。
    DashScope text-embedding-v3 每次最多 10 条，这里用 batch_size=6 留余量。
    如果 API 调用失败，直接抛出异常（不使用随机向量降级）。
    """
    from utils.embedding_client import embedding_client

    all_embeddings: list[list[float]] = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        vecs = await embedding_client.encode(batch, normalize=True)
        # 检查是否降级为随机向量（通过检查方差判断）
        import numpy as np
        mean_norm = np.mean([np.linalg.norm(v) for v in vecs])
        if abs(mean_norm - 1.0) > 0.1:  # 归一化向量范数应接近1
            raise RuntimeError(f"Embedding 降级为随机向量，请检查 API 配置。batch={i//batch_size}")
        all_embeddings.extend(vecs.tolist())
        done = min(i + batch_size, total)
        logger.info(f"  嵌入进度: {done}/{total}")

    return all_embeddings


# ============================================================
# 主初始化函数
# ============================================================

def init_milvus():
    """初始化 Milvus 向量库（全部使用真实 embedding）"""
    from pymilvus import MilvusClient, DataType

    uri = f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
    logger.info(f"连接 Milvus: {uri}")
    client = MilvusClient(uri=uri)

    symptom_col = "symptom_collection"
    disease_col = "disease_collection"
    knowledge_col = "medical_knowledge"

    # ============================================================
    # 1. 删除旧集合（清除随机向量数据）
    # ============================================================
    for col_name in [symptom_col, disease_col, knowledge_col]:
        if client.has_collection(col_name):
            client.drop_collection(col_name)
            logger.info(f"已删除旧集合: {col_name}")

    # ============================================================
    # 2. 创建症状向量集合
    # ============================================================
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("symptom_name", DataType.VARCHAR, max_length=256)
    schema.add_field("category", DataType.VARCHAR, max_length=64)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=settings.EMBEDDING_DIMENSION)
    schema.add_field("disease_ids", DataType.JSON)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="embedding", metric_type="COSINE",
                           index_type="IVF_FLAT", params={"nlist": 128})
    client.create_collection(collection_name=symptom_col, schema=schema, index_params=index_params)
    logger.info(f"✅ 症状向量集合 {symptom_col} 创建成功")

    # ============================================================
    # 3. 创建疾病向量集合
    # ============================================================
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("disease_name", DataType.VARCHAR, max_length=256)
    schema.add_field("icd_code", DataType.VARCHAR, max_length=32)
    schema.add_field("description", DataType.VARCHAR, max_length=1024)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=settings.EMBEDDING_DIMENSION)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="embedding", metric_type="COSINE",
                           index_type="IVF_FLAT", params={"nlist": 128})
    client.create_collection(collection_name=disease_col, schema=schema, index_params=index_params)
    logger.info(f"✅ 疾病向量集合 {disease_col} 创建成功")

    # ============================================================
    # 4. 创建医学知识集合
    # ============================================================
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("title", DataType.VARCHAR, max_length=512)
    schema.add_field("content", DataType.VARCHAR, max_length=4096)
    schema.add_field("source", DataType.VARCHAR, max_length=256)
    schema.add_field("category", DataType.VARCHAR, max_length=64)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=settings.EMBEDDING_DIMENSION)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="embedding", metric_type="COSINE",
                           index_type="IVF_FLAT", params={"nlist": 256})
    client.create_collection(collection_name=knowledge_col, schema=schema, index_params=index_params)
    logger.info(f"✅ 医学知识集合 {knowledge_col} 创建成功")

    # ============================================================
    # 5. 生成 embedding 并插入数据（异步，使用 Qwen API）
    # ============================================================
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # ---- 5a. 症状向量 ----
        logger.info(f"\n开始为 {len(SYMPTOMS)} 个症状生成 embedding...")
        symptom_texts = [f"{s[0]} {s[2]}" for s in SYMPTOMS]  # 名称 + 分类
        symptom_embeddings = loop.run_until_complete(batch_embed(symptom_texts))

        symptom_data = []
        for i, s in enumerate(SYMPTOMS):
            symptom_data.append({
                "id": i + 1,
                "symptom_name": s[0],
                "category": s[2],
                "embedding": symptom_embeddings[i],
                "disease_ids": json.dumps([], ensure_ascii=False),
            })
        client.insert(collection_name=symptom_col, data=symptom_data)
        logger.info(f"✅ 插入 {len(symptom_data)} 条症状向量数据")

        # ---- 5b. 疾病向量 ----
        logger.info(f"\n开始为 {len(DISEASES)} 个疾病生成 embedding...")
        disease_texts = [f"{d[0]} {d[3]}" for d in DISEASES]  # 名称 + 描述
        disease_embeddings = loop.run_until_complete(batch_embed(disease_texts))

        disease_data = []
        for i, d in enumerate(DISEASES):
            disease_data.append({
                "id": i + 1,
                "disease_name": d[0],
                "icd_code": d[1],
                "description": d[3][:1000],  # 截断防止超长
                "embedding": disease_embeddings[i],
            })
        client.insert(collection_name=disease_col, data=disease_data)
        logger.info(f"✅ 插入 {len(disease_data)} 条疾病向量数据")

        # ---- 5c. 医学知识向量 ----
        logger.info(f"\n开始为 {len(KNOWLEDGE_DATA)} 条知识生成 embedding...")
        knowledge_texts = [f"{k['title']} {k['content'][:200]}" for k in KNOWLEDGE_DATA]
        knowledge_embeddings = loop.run_until_complete(batch_embed(knowledge_texts))

        knowledge_data = []
        for i, k in enumerate(KNOWLEDGE_DATA):
            knowledge_data.append({
                "id": i + 1,
                "title": k["title"],
                "content": k["content"],
                "source": k["source"],
                "category": k.get("category", "通用"),
                "embedding": knowledge_embeddings[i],
            })
        client.insert(collection_name=knowledge_col, data=knowledge_data)
        logger.info(f"✅ 插入 {len(knowledge_data)} 条医学知识向量数据")

    finally:
        loop.close()

    # ============================================================
    # 6. 打印集合信息
    # ============================================================
    logger.info(f"\n{'='*50}")
    logger.info(f"  Milvus 集合统计")
    logger.info(f"{'='*50}")
    for col_name in [symptom_col, disease_col, knowledge_col]:
        client.flush(collection_name=col_name)  # 确保落盘，row_count 才会更新
        stats = client.get_collection_stats(col_name)
        logger.info(f"  {col_name}: {stats.get('row_count', 0)} 条数据")
    logger.info(f"{'='*50}")
    logger.info("✅ Milvus 初始化完成（全部使用真实 embedding）")


if __name__ == "__main__":
    init_milvus()
