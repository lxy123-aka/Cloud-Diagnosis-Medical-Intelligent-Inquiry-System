"""检查 Python 包安装位置"""
import sys
import site

print(f"Python 可执行文件: {sys.executable}")
print(f"Python prefix:     {sys.prefix}")
print(f"site-packages 路径:")
for p in site.getsitepackages():
    print(f"  {p}")
print(f"用户 site-packages: {site.getusersitepackages()}")

# 检查关键包位置
import pymilvus
import langgraph
import langchain_openai
import dashscope

print(f"\n关键包位置:")
print(f"  pymilvus:        {pymilvus.__file__}")
print(f"  langgraph:       {langgraph.__file__}")
print(f"  langchain_openai: {langchain_openai.__file__}")
print(f"  dashscope:       {dashscope.__file__}")
