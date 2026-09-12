"""验证 Milvus 数据是否真正存在（通过搜索测试）"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
from pymilvus import MilvusClient
from configs.settings import settings

async def main():
    uri = f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
    client = MilvusClient(uri=uri)

    # 检查集合是否存在
    for col in ["symptom_collection", "disease_collection", "medical_knowledge"]:
        exists = client.has_collection(col)
        print(f"  {col}: exists={exists}")

    # 通过搜索验证数据
    from utils.embedding_client import embedding_client

    print("\n--- 症状搜索测试 ---")
    query_vec = await embedding_client.encode_single("腹泻", normalize=True)
    results = client.search(
        collection_name="symptom_collection",
        data=[query_vec],
        limit=5,
        output_fields=["symptom_name", "category"],
    )
    for hit in results[0]:
        print(f"  {hit['entity']['symptom_name']:15s} score={hit['distance']:.4f}")

    print("\n--- 疾病搜索测试 ---")
    query_vec = await embedding_client.encode_single("急性胃肠炎", normalize=True)
    results = client.search(
        collection_name="disease_collection",
        data=[query_vec],
        limit=5,
        output_fields=["disease_name", "icd_code"],
    )
    for hit in results[0]:
        print(f"  {hit['entity']['disease_name']:15s} score={hit['distance']:.4f}")

    print("\n--- 知识搜索测试 ---")
    query_vec = await embedding_client.encode_single("感冒怎么办", normalize=True)
    results = client.search(
        collection_name="medical_knowledge",
        data=[query_vec],
        limit=3,
        output_fields=["title", "category"],
    )
    for hit in results[0]:
        print(f"  {hit['entity']['title'][:30]:30s} score={hit['distance']:.4f}")

    # 用 query 获取实际行数
    print("\n--- 集合行数（通过 query）---")
    for col in ["symptom_collection", "disease_collection", "medical_knowledge"]:
        res = client.query(collection_name=col, filter="", output_fields=["id"], limit=1)
        # 尝试 stats
        stats = client.get_collection_stats(col)
        print(f"  {col}: stats={stats}")

    print("\n✅ Milvus 数据验证完成")

asyncio.run(main())
