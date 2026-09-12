"""验证 Milvus 数据"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymilvus import MilvusClient
c = MilvusClient(uri="http://localhost:19530")
for col in ["symptom_collection", "disease_collection", "medical_knowledge"]:
    c.flush(col)
    stats = c.get_collection_stats(col)
    print(f"{col}: {stats.get('row_count', 0)} 条数据")

# 测试搜索
print("\n--- 测试症状搜索: '妊娠' ---")
from utils.embedding_client import embedding_client
import asyncio
async def test():
    vec = await embedding_client.encode_single("妊娠")
    results = c.search(collection_name="symptom_collection", data=[vec], limit=5, output_fields=["symptom_name", "category"])
    for r in results[0]:
        print(f"  {r['entity']['symptom_name']} ({r['entity']['category']}) score={r['distance']:.4f}")
    
    print("\n--- 测试知识搜索: '怀孕了吃什么药' ---")
    vec2 = await embedding_client.encode_single("怀孕了吃什么药")
    results2 = c.search(collection_name="medical_knowledge", data=[vec2], limit=3, output_fields=["title", "category"])
    for r in results2[0]:
        print(f"  {r['entity']['title']} ({r['entity']['category']}) score={r['distance']:.4f}")

asyncio.run(test())
