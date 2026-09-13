# 云诊医疗智能问诊系统（CDMIIS）

> Cloud Diagnosis Medical Intelligent Inquiry System

## 1. 项目简介

云诊医疗智能问诊系统是一套基于 **大语言模型 + 多智能体协作** 的 AI 辅助问诊平台。系统采用 **Supervisor + 5 Worker** 多智能体架构，支持以下五大业务场景：

| 业务 | 说明 |
|------|------|
| **在线问诊** | 多轮追问 → 症状标准化 → 置信度驱动诊断收敛 |
| **报告解读** | 文本报告 / CT·B超·化验单图片多模态分析 |
| **药物咨询** | 药品信息查询、用法用量、禁忌·副作用·相互作用 |
| **知识问答** | 医学科普、疾病知识检索（RAG 向量检索增强） |
| **处方审核** | 药物相互作用、禁忌症、剂量安全性自动校验 |

**核心技术亮点：**

- **三层症状标准化流水线**：LLM 语义提取 → Neo4j 知识图谱精确匹配 → Milvus 向量语义召回，实测召回率从基线 83.8% 提升至 97.3%（+13.5%）
- **置信度驱动诊断收敛引擎**：双条件收敛（Top1 ≥ 70% 或 Top1-Top2 差值 ≥ 30%）+ 最小 3 轮门控 + 10 轮强制兜底，RAG 鉴别要点辅助追问
- **双层记忆系统**：RedisStack 短期会话记忆 + PostgreSQL 长期病历存储
- **Qwen-VL 多模态**：支持 CT / B超 / 化验单等医学影像图片分析
- **RAG 向量检索增强**：统一 RAG 检索服务，三集合（知识/疾病/症状）并行混合检索，全链路 Worker 接入
- **LoRA 微调 + 模型蒸馏**：LlamaFactory + SwanLab 微调 Qwen3.5-4B 医疗 NER（r=16, 全线性层注入），单张 RTX4090 约 6 小时完成训练，微调后医疗实体识别命中 90-95% 以上；蒸馏至 1.5B 实现整体推理管线平均耗时 ≤300ms

---

## 2. 目录结构

```
Cloud Diagnosis Medical Intelligent Inquiry System/
├── main.py                          # 项目启动入口（命令行交互 + FastAPI 服务）
├── requirements.txt                 # Python 依赖清单
├── .env.example                     # 环境变量模板
├── README.md                        # 本文件
│
├── configs/                         # 全局配置 & 数据库初始化脚本
│   ├── __init__.py
│   ├── settings.py                  # Pydantic Settings 全局配置（.env 自动加载）
│   ├── neo4j_init.cypher            # Neo4j 知识图谱初始化（节点/关系/示例数据）
│   ├── milvus_init.py               # Milvus 向量库建表 + 示例症状向量
│   └── postgres_init.sql            # PostgreSQL 建表 SQL（users / medical_records / consultations）
│
├── state/                           # LangGraph 状态定义
│   ├── __init__.py
│   └── state_schema.py              # InquiryState（全局共享状态）+ 辅助 TypedDict
│
├── agent/                           # 多智能体层
│   ├── __init__.py
│   ├── graph_builder.py             # LangGraph StateGraph 构建（Supervisor + Worker 拓扑）
│   ├── supervisor_agent.py          # Supervisor 调度 Agent（意图识别 + 收敛判断）
│   └── worker/                      # 5 个 Worker Agent
│       ├── __init__.py
│       ├── collect_agent.py         # 问诊采集 Worker（症状标准化 + 诊断引擎）
│       ├── report_agent.py          # 报告解读 Worker（文本 + Qwen-VL 图片分析）
│       ├── drug_agent.py            # 药物咨询 Worker（药品知识库 + LLM 用药建议）
│       ├── qa_agent.py              # 知识问答 Worker（RAG 检索 + LLM 科普）
│       └── prescription_agent.py    # 处方审核 Worker（规则校验 + LLM 审核）
│
├── pipeline/                        # 核心流水线
│   ├── __init__.py
│   ├── symptom_normalize.py         # 三层症状标准化流水线
│   └── diagnosis_engine.py          # 置信度驱动诊断收敛引擎
│
├── memory/                          # 双层记忆
│   ├── __init__.py
│   ├── short_memory.py              # RedisStack 短期会话记忆（LangGraph Checkpointer）
│   └── long_memory.py               # PostgreSQL 长期病历记忆（asyncpg 连接池）
│
├── multimodal/                      # 多模态模块
│   ├── __init__.py
│   ├── vl_infer.py                  # Qwen-VL 多模态推理（CT/B超/化验单分析）
│   └── vl_sft_train.py              # Qwen-VL LoRA 微调脚本
│
├── train/                           # 模型训练
│   ├── __init__.py
│   ├── sft_lora_train.py            # LlamaFactory LoRA SFT 微调（Qwen3.5-4B 医疗 NER）
│   ├── data_processor.py            # 训练数据集预处理
│   └── model_distill.py             # 模型蒸馏（教师 4B → 学生 1.5B）
│
├── utils/                           # 工具层
│   ├── __init__.py
│   ├── llm_client.py                # 统一 LLM 客户端封装（文本 + 多模态）
│   ├── embedding_client.py          # 统一 Embedding 客户端（Qwen API + 本地模型降级）
│   ├── rag_retrieval.py             # 统一 RAG 检索服务（三集合混合检索）
│   └── tools.py                     # Function Calling 工具注册（图谱/向量/药物/RAG）
│
├── tests/                           # 测试与评估
│   ├── test_conversation.py         # 端到端测试（8 个场景）
│   ├── eval_symptom_recall.py       # 症状标准化召回率评估
│   └── eval_ner.py                  # 医疗 NER 实体识别评估
│
├── docs/                            # 文档
│   └── EVAL.md                      # 评估报告（目标/实测对比）
│
└── logs/                            # 运行日志（自动生成）
```

---

## 3. 技术栈

| 类别 | 技术 |
|------|------|
| **编程语言** | Python 3.10+ |
| **多智能体框架** | LangGraph（StateGraph + 条件路由） |
| **大语言模型** | Qwen3.5（4B / 1.5B）、Qwen-VL（多模态） |
| **知识图谱** | Neo4j（疾病-症状-药品-科室） |
| **向量数据库** | Milvus（IVF_FLAT 索引，1024 维） |
| **短期记忆** | RedisStack（RedisJSON，LangGraph Checkpointer） |
| **长期记忆** | PostgreSQL + asyncpg 异步连接池 |
| **微调框架** | LlamaFactory（LoRA SFT） |
| **训练监控** | SwanLab |
| **嵌入模型** | BAAI/bge-large-zh-v1.5 本地模型（1024 维，降级至 Qwen text-embedding-v3 API） |
| **Web 框架** | FastAPI + Uvicorn |
| **配置管理** | Pydantic Settings（.env 自动加载） |
| **日志** | Loguru |

---

## 4. 环境要求

| 项目 | 要求 |
|------|------|
| **Python** | 3.10 及以上 |
| **GPU** | NVIDIA RTX 4090（可选，模型微调/蒸馏时需要） |
| **Neo4j** | 5.x（知识图谱） |
| **Milvus** | 2.3+（向量检索） |
| **RedisStack** | 7.x（需 RedisJSON 模块） |
| **PostgreSQL** | 14+（长期病历存储） |

---

## 5. 部署步骤

### 5.1 Docker 启动四个数据库

```bash
# Neo4j 知识图谱
docker run -d --name cdmiis_neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/cdmiis_password \
  -e NEO4J_PLUGINS='["apoc"]' \
  neo4j:5

# Milvus 向量库（Standalone 模式）
docker run -d --name cdmiis_milvus \
  -p 19530:19530 -p 9091:9091 \
  -e ETCD_ENDPOINTS=localhost:2379 \
  milvusdb/milvus:v2.3.0

# RedisStack 短期记忆
docker run -d --name cdmiis_redis \
  -p 6379:6379 \
  redis/redis-stack:latest

# PostgreSQL 长期记忆
docker run -d --name cdmiis_postgres \
  -p 5432:5432 \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=cdmiis_password \
  -e POSTGRES_DB=cloud_diagnosis \
  postgres:16
```

### 5.2 安装 Python 依赖

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 5.3 配置环境变量

```bash
# 复制模板
cp .env.example .env

# 编辑 .env 文件，填入实际配置
# 必填项：QWEN_API_KEY、NEO4J_PASSWORD、POSTGRES_PASSWORD
```

---

## 6. 数据库初始化

```bash
# 1. Neo4j 知识图谱初始化
#    方式一：Neo4j Browser 打开 http://localhost:7474，粘贴 neo4j_init.cypher 内容执行
#    方式二：neo4j-shell
cat configs/neo4j_init.cypher | neo4j-shell -u neo4j -p cdmiis_password

# 2. Milvus 向量库初始化
python configs/milvus_init.py

# 3. PostgreSQL 建表
psql -U postgres -d cloud_diagnosis -f configs/postgres_init.sql
```

---

## 7. 启动方式

### 命令行交互模式

```bash
python main.py
```

启动后可直接输入问题开始问诊：
- 输入文本进行问诊 / 药物咨询 / 知识问答
- 输入 `image:图片路径` 上传医学图片
- 输入 `quit` 或 `exit` 退出

### API 服务模式

```bash
python main.py --api
```

启动 FastAPI 服务后，可通过 HTTP 调用：

```bash
# 对话接口
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "我头疼三天了", "user_id": "u001", "session_id": "s001"}'

# 健康检查
curl http://localhost:8000/health
```

---

## 8. 测试

```bash
# 运行全部端到端测试（8 个场景）
python tests/test_conversation.py
```

测试覆盖：
1. 在线问诊完整流程（多轮追问 → 诊断收敛）
2. 报告解读（文本报告）
3. 药物咨询
4. 知识问答
5. 处方审核
6. 症状标准化流水线
7. 诊断收敛引擎
8. Function Calling 工具注册

### 评估脚本

```bash
# 症状标准化流水线召回率评估（基线 vs 完整流水线）
python tests/eval_symptom_recall.py

# 医疗 NER 实体识别评测（F1 / 命中率）
python tests/eval_ner.py

# 整体推理管线耗时基准测试
python scripts/benchmark_pipeline.py
```

评估报告详见 `docs/EVAL.md`。

每个 Agent 文件底部均有 `if __name__ == "__main__"` 独立调试入口，可单独运行测试：

```bash
python -m agent.supervisor_agent
python -m agent.worker.collect_agent
python -m pipeline.symptom_normalize
python -m pipeline.diagnosis_engine
python -m memory.short_memory
python -m memory.long_memory
```

---

## 9. 模型微调

### 9.1 Qwen3.5-4B 医疗 NER LoRA 微调

```bash
# 使用 LlamaFactory 进行 LoRA SFT
python train/sft_lora_train.py

# 训练参数在 configs/settings.py 中配置
# 训练日志通过 SwanLab 监控
```

训练数据格式见 `train/data_processor.py`，支持将标注数据转换为 LlamaFactory 所需的 JSON 格式。训练数据加载国家标准医疗数据集，由 `data_processor.py` 将 BIO 标注转换为 Alpaca 格式后灌入训练。单张 RTX4090 24G 显卡约 6 小时完成训练。

### 9.2 Qwen-VL 多模态微调

```bash
python multimodal/vl_sft_train.py
```

用于微调 Qwen-VL 模型在医学影像分析任务上的表现。

### 9.3 模型蒸馏

```bash
# 教师 Qwen3.5-4B → 学生 Qwen3.5-1.5B
python train/model_distill.py
```

蒸馏至 1.5B 实现单次 NER 推理 ≤300ms，整体推理管线平均耗时 ≤300ms（本地链路），面向私有化部署。运行 `python scripts/benchmark_pipeline.py` 可实测管线耗时。

---

## 10. 常见问题

### Q: 数据库连不上怎么办？

系统对四个外部数据库均实现了 **优雅降级**，任何一个数据库不可用时系统不会崩溃：

| 数据库 | 降级行为 |
|--------|----------|
| **Neo4j** | 知识图谱查询失败时，症状标准化退化为仅使用 LLM 提取 + Milvus 向量召回；诊断引擎退化为 RAG + LLM 直接推理候选疾病 |
| **Milvus** | 向量检索失败时，症状标准化退化为 LLM 提取 + Neo4j 精确匹配；向量权重归零 |
| **RedisStack** | 短期记忆不可用时，对话历史保存跳过，不影响当前会话处理 |
| **PostgreSQL** | 长期记忆不可用时，历史病历加载跳过，不影响当前问诊流程 |

启动时如果数据库连接失败，会看到类似日志：
```
⚠️ Redis 连接失败（可选）: Connection refused
⚠️ PostgreSQL 连接失败（可选）: Connection refused
```
这是正常行为，系统仍可正常运行，只是对应的记忆功能暂不可用。

### Q: 没有 GPU 能运行吗？

可以。GPU 仅在模型微调（`train/`）和蒸馏（`model_distill.py`）时需要。在线问诊、报告解读等核心功能通过 API 调用 Qwen3.5 云端模型，不需要本地 GPU。

### Q: RAG 知识库如何工作？

系统已构建统一 RAG 检索服务（`utils/rag_retrieval.py`），封装 Milvus 三个集合：
- **medical_knowledge**：医学知识（症状鉴别、常用药物、疾病科普、检查指标、健康常识）
- **disease_collection**：疾病信息库（疾病名、ICD 编码、描述）
- **symptom_collection**：标准症状库（症状名、分类）

全链路 Worker（问诊追问、知识问答、药物咨询、诊断引擎 LLM 降级）均已接入 RAG，混合检索通过 `asyncio.gather` 并行查询三集合以最小化延迟。运行 `python configs/milvus_init.py` 初始化知识库数据。

### Q: 如何修改诊断收敛阈值？

在 `.env` 文件中配置：
```
CONFIDENCE_THRESHOLD=0.70    # Top1 置信度阈值（双条件之一：Top1 ≥ 70% 或 Top1-Top2 差值 ≥ 30%）
CONFIDENCE_GAP=0.30          # Top1-Top2 差值阈值
MIN_INQUIRY_ROUNDS=3         # 最小追问轮数（至少追问 3 轮才允许收敛）
MAX_INQUIRY_ROUNDS=10        # 最大追问轮数
```
