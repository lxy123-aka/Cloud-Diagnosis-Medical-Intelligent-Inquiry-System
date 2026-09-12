import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymilvus import MilvusClient
from configs.settings import settings

uri = f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
c = MilvusClient(uri=uri)

cols = c.list_collections()
print(f"Collections: {cols}")

for col in cols:
    stats = c.get_collection_stats(col)
    print(f"  {col}: {stats}")
