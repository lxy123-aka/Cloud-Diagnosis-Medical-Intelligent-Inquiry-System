"""检查关键依赖包是否已安装"""
import sys
print(f"Python: {sys.executable}")
print(f"Version: {sys.version}")

packages = [
    "dashscope",
    "pymilvus",
    "neo4j",
    "redis",
    "asyncpg",
    "langgraph",
    "langchain_openai",
    "loguru",
    "numpy",
    "duckduckgo_search",
    "pydantic_settings",
]

for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "?")
        print(f"  OK  {pkg} ({ver})")
    except ImportError:
        print(f"  MISS {pkg}")
