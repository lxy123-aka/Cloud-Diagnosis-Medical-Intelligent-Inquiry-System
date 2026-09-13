"""
pipeline/symptom_normalize.py
=============================
三层症状标准化流水线。

第一层：LLM 语义提取 —— 从用户口语中提取症状关键词
第二层：Neo4j 知识图谱精确匹配 —— 通过名称/同义词精确匹配标准症状
第三层：Milvus 向量语义召回 —— 语义相似度召回候选症状

最终融合三层结果，输出标准化症状列表。
输入：用户口语描述（如"肚子左边一直疼"）
输出：标准化症状列表（如[{"standard_name": "腹痛", "confidence": 0.95}]）
"""

from __future__ import annotations
import json
from loguru import logger
from langchain_openai import ChatOpenAI

from configs.settings import settings


def _is_rate_limit_error(e: Exception) -> bool:
    """判断是否为额度/限流类错误（可降级到其他模型）"""
    status = getattr(e, "status_code", None)
    # langchain 异常可能包装了 status_code；也检查字符串匹配
    err_str = str(e).lower()
    return (
        status in (429, 402)
        or "rate limit" in err_str
        or "quota" in err_str
    )


def _build_chat_openai(model_name: str) -> ChatOpenAI:
    """构建指定模型的 ChatOpenAI 实例"""
    return ChatOpenAI(
        model=model_name,
        api_key=settings.QWEN_API_KEY,
        base_url=settings.QWEN_BASE_URL,
        temperature=0.1,
    )


class SymptomNormalizePipeline:
    """三层症状标准化流水线"""

    # 类级别 Neo4j driver 单例，所有实例共享
    _neo4j_driver = None
    _milvus_client = None
    _symptom_whitelist: set[str] = None  # 标准症状名称白名单（懒加载）

    def __init__(self):
        # 主模型 + 降级模型链
        self._model_chain = [settings.QWEN_MODEL_NAME]
        for m in settings.QWEN_FALLBACK_MODELS.split(","):
            m = m.strip()
            if m and m not in self._model_chain:
                self._model_chain.append(m)
        # 懒加载：按需创建 ChatOpenAI 实例（避免启动时初始化所有模型）
        self._llm_instances: dict[str, ChatOpenAI] = {}

    def _get_llm(self, model_name: str) -> ChatOpenAI:
        """获取或创建指定模型的 ChatOpenAI 实例（懒加载缓存）"""
        if model_name not in self._llm_instances:
            self._llm_instances[model_name] = _build_chat_openai(model_name)
        return self._llm_instances[model_name]

    @property
    def _llm(self) -> ChatOpenAI:
        """兼容旧代码：返回主模型的 ChatOpenAI 实例"""
        return self._get_llm(self._model_chain[0])

    @classmethod
    def _get_whitelist(cls) -> set[str]:
        """懒加载标准症状名称白名单（从 neo4j_symptoms 数据 + 常见别名）"""
        if cls._symptom_whitelist is None:
            try:
                from configs.neo4j_symptoms import SYMPTOMS
                names = set()
                for s in SYMPTOMS:
                    names.add(s[0])  # 标准名称
                    # 同义词/别名也加入白名单
                    if len(s) > 3 and s[3]:
                        names.update(s[3])
                cls._symptom_whitelist = names
                logger.debug(f"[症状流水线] 白名单加载完成: {len(names)} 个标准症状名")
            except Exception as e:
                logger.warning(f"[症状流水线] 白名单加载失败: {e}，跳过过滤")
                cls._symptom_whitelist = set()
        return cls._symptom_whitelist

    @classmethod
    def _get_driver(cls):
        """获取或创建 Neo4j 异步 driver（类级别单例，避免重复建连）"""
        if cls._neo4j_driver is None:
            from neo4j import AsyncGraphDatabase
            cls._neo4j_driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            )
        return cls._neo4j_driver

    @classmethod
    def _get_milvus_client(cls):
        """获取或创建 Milvus 客户端（类级别单例，避免重复建连）"""
        if cls._milvus_client is None:
            from pymilvus import MilvusClient
            cls._milvus_client = MilvusClient(
                uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
            )
        return cls._milvus_client

    @classmethod
    async def close(cls):
        """关闭 Neo4j driver 连接（程序退出时调用）"""
        if cls._neo4j_driver is not None:
            await cls._neo4j_driver.close()
            cls._neo4j_driver = None

    async def run(self, user_text: str, is_short_answer: bool = False) -> list[dict]:
        """
        执行完整的三层标准化流水线。

        :param user_text: 用户原始输入（如"肚子左边一直疼"）
        :param is_short_answer: 是否为追问的简短回答（如"没有"、"有"、"有点"）
        :return: 标准化后的症状列表
        """
        logger.info(f"[症状流水线] 开始处理: {user_text} (short_answer={is_short_answer})")

        # 简短回答（"没有"、"有"、"有点"等）不包含新症状，直接返回空
        if is_short_answer:
            logger.info("[症状流水线] 检测到简短回答，跳过症状提取")
            return []

        # === 第一层：LLM 语义提取 ===
        llm_symptoms = await self._layer1_llm_extract(user_text)
        logger.info(
            f"[症状流水线] LLM提取结果: "
            f"{[s.get('standard_name', '') for s in llm_symptoms]}"
        )

        # === 第一层半：白名单诊断日志（不再硬过滤，让图谱/向量层做映射） ===
        whitelist = self._get_whitelist()
        standard_count = sum(
            1 for s in llm_symptoms
            if s.get("standard_name", "") in whitelist
        )
        logger.info(
            f"[症状流水线] LLM提取 {len(llm_symptoms)} 个症状，"
            f"其中 {standard_count} 个为标准术语"
        )

        # === 第二层 + 第三层：对每个 LLM 提取的症状做图谱匹配 + 向量召回 ===
        final_symptoms: list[dict] = []
        for symptom in llm_symptoms:
            name = symptom.get("standard_name", "")

            # 第二层：Neo4j 精确匹配
            graph_matches = await self._layer2_graph_match(name)

            # 第三层：Milvus 向量语义召回
            vector_matches = await self._layer3_vector_recall(name)

            # 三层融合
            merged = self._merge_results(symptom, graph_matches, vector_matches)
            final_symptoms.extend(merged)

        # 去重并保留最高置信度
        final_symptoms = self._deduplicate(final_symptoms)
        # 限制最多返回 3 个症状，避免噪声污染诊断
        final_symptoms = final_symptoms[:3]
        logger.info(
            f"[症状流水线] 最终标准化结果: "
            f"{[(s.get('standard_name', ''), round(s.get('confidence', 0), 2)) for s in final_symptoms]}"
        )
        return final_symptoms

    # ============================================================
    # 第一层：LLM 语义提取
    # ============================================================

    async def _layer1_llm_extract(self, text: str) -> list[dict]:
        """
        第一层：使用 LLM 从口语化描述中提取结构化症状。
        通过 Function Calling 强制输出格式，支持模型自动降级。
        """
        # 定义 Function Calling 工具
        tools = [{
            "type": "function",
            "function": {
                "name": "extract_symptoms",
                "description": "从用户描述中提取症状信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symptoms": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "raw_text": {"type": "string", "description": "原始描述"},
                                    "standard_name": {"type": "string", "description": "标准医学术语"},
                                    "category": {"type": "string", "description": "症状分类"},
                                    "severity": {"type": "string", "enum": ["轻度", "中度", "重度"]},
                                    "confidence": {"type": "number", "description": "置信度 0-1"}
                                },
                                "required": ["raw_text", "standard_name", "confidence"]
                            }
                        }
                    },
                    "required": ["symptoms"]
                }
            }
        }]
        
        system_content = (
            "你是一个医学症状提取专家。从用户的口语化描述中提取症状信息。\n\n"
            "⚠️ 严格规则（必须遵守）：\n"
            "1. 只提取用户明确表述为【正在经历的身体不适】的症状\n"
            "2. 忽略否定表达：用户说'没有X'、'不痛'、'无X'、'不X'时，"
            "不要提取该症状（否定≠存在）\n"
            "3. 忽略非症状实体：人名、药名、食物、生活场景、情绪状态不是症状\n"
            "4. 忽略回答性短语：'是的'、'没有'、'不清楚'、'还好'等不包含症状\n"
            "5. 标准医学术语：将口语、方言转换为标准症状术语\n"
            "   口语映射：拉肚子→腹泻，头疼→头痛，老想上厕所→尿频，鼻子堵了→鼻塞\n"
            "   炎症/不适描述映射为症状而非疾病：\n"
            "     嗓子发炎→咽痛（非咽炎），胃发炎→上腹痛（非胃炎）\n"
            "   口语优先映射到标准症状名：\n"
            "     胃酸→反酸，打嗝→嗳气，老想上厕所→尿频\n"
            "   方言映射：脑壳疼→头痛，肚肚痛→腹痛，刺挠→皮肤瘙痒\n"
            "   身体部位口语：心口窝儿疼→上腹痛，腰杆子疼→腰痛\n"
            "6. 不要推测或脑补：只提取文本中明确提到的症状\n"
            "7. 如果文本中没有明确的症状，返回空数组\n"
            "8. 使用 extract_symptoms 工具输出"
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": text},
        ]

        # 模型降级链：主模型 → 备用模型1 → 备用模型2 → ...
        last_error = None
        for model_name in self._model_chain:
            try:
                llm = self._get_llm(model_name)
                llm_with_tools = llm.bind_tools(
                    tools,
                    tool_choice={"type": "function", "function": {"name": "extract_symptoms"}}
                )
                response = await llm_with_tools.ainvoke(messages)
                if model_name != self._model_chain[0]:
                    logger.info(f"[症状流水线] LLM 已降级到备用模型: {model_name}")
                break
            except Exception as e:
                if _is_rate_limit_error(e) and model_name != self._model_chain[-1]:
                    logger.warning(f"[症状流水线] 模型 {model_name} 额度用尽，尝试下一个备用模型...")
                    last_error = e
                    continue
                raise
        else:
            # 所有模型都失败
            raise last_error

        # 解析 Function Calling 结果
        try:
            tool_call = response.tool_calls[0]
            args = tool_call["args"] if isinstance(tool_call["args"], dict) else json.loads(tool_call["args"])
            symptoms = []
            for s in args.get("symptoms", []):
                symptoms.append({
                    "raw_text": s.get("raw_text", ""),
                    "standard_name": s.get("standard_name", ""),
                    "category": s.get("category", ""),
                    "severity": s.get("severity", "中度"),
                    "confidence": float(s.get("confidence", 0.5)),
                    "source": "llm_extract",
                })
            return symptoms
        except (AttributeError, IndexError, KeyError, json.JSONDecodeError) as e:
            logger.warning(f"LLM 症状提取解析失败: {e}，降级处理")
            # 降级：直接返回原始文本作为症状
            return [{
                "raw_text": text,
                "standard_name": text,
                "confidence": 0.3,
                "source": "llm_extract",
            }]

    # ============================================================
    # 第一层半：白名单过滤
    # ============================================================

    def _filter_by_whitelist(self, symptoms: list[dict]) -> list[dict]:
        """
        用标准症状白名单过滤 LLM 提取结果。
        过滤规则：
        1. 完全匹配白名单 → 保留
        2. 白名单中某症状名包含提取名（如提取“头痛”被“头痛”包含）→ 保留
        3. 提取名包含白名单中某症状名（如提取“左侧头痛”包含“头痛”）→ 保留
        4. 都不匹配 → 过滤掉（如“第二颗牙齿”、“存在症状”）
        """
        whitelist = self._get_whitelist()
        if not whitelist:
            return symptoms  # 白名单未加载，跳过过滤

        filtered = []
        for s in symptoms:
            name = s.get("standard_name", "").strip()
            if not name:
                continue

            # 规则 1：完全匹配
            if name in whitelist:
                filtered.append(s)
                continue

            # 规则 2/3：包含关系匹配
            matched = False
            for wn in whitelist:
                if len(name) >= 2 and len(wn) >= 2:
                    if name in wn or wn in name:
                        filtered.append(s)
                        matched = True
                        break

            if not matched:
                logger.info(f"[症状流水线] 白名单过滤: 丢弃非标准症状 '{name}'")

        return filtered

    # ============================================================
    # 第二层：Neo4j 知识图谱精确匹配
    # ============================================================

    async def _layer2_graph_match(self, symptom_name: str) -> list[dict]:
        """
        第二层：Neo4j 知识图谱精确匹配。
        通过症状名称或同义词在图谱中精确查找。
        """
        try:
            driver = self._get_driver()
            async with driver.session() as session:
                result = await session.run(
                    """
                    MATCH (s:Symptom)
                    WHERE s.name = $name OR $name IN s.aliases
                    RETURN s.name AS name, s.snomed_code AS snomed_code,
                           s.category AS category, s.aliases AS synonyms,
                           s.questions AS questions
                    """,
                    {"name": symptom_name},
                )
                records = await result.data()

            results = []
            for m in records:
                results.append({
                    "symptom_name": m["name"],
                    "snomed_code": m.get("snomed_code"),
                    "category": m.get("category", ""),
                    "questions": m.get("questions", []),
                    "confidence": 1.0,  # 精确匹配置信度最高
                    "source": "graph_match",
                })
            logger.debug(f"[图谱匹配] '{symptom_name}' 匹配到 {len(results)} 个结果")
            return results

        except Exception as e:
            logger.warning(f"[图谱匹配] 查询失败（Neo4j可能未启动）: {e}")
            return []

    # ============================================================
    # 第三层：Milvus 向量语义召回
    # ============================================================

    async def _layer3_vector_recall(self, symptom_name: str) -> list[dict]:
        """
        第三层：Milvus 向量语义召回。
        将症状文本向量化后做近似最近邻检索。
        严格过滤：高阈值 + 字符重叠检查，避免召回无关症状。
        """
        try:
            from utils.embedding_client import embedding_client

            # 使用统一的 Embedding 客户端（优先 Qwen API）
            query_vector = await embedding_client.encode_single(symptom_name, normalize=True)

            # 连接 Milvus 检索（复用类级别单例）
            client = self._get_milvus_client()
            results = client.search(
                collection_name="symptom_collection",
                data=[query_vector],
                limit=5,
                output_fields=["symptom_name", "category"],
            )

            matches = []
            for hit in results[0]:
                matched_name = hit["entity"]["symptom_name"]
                score = hit["distance"]

                # 严格过滤1：向量相似度阈值提高到 0.80
                if score < 0.80:
                    continue

                # 严格过滤2：字符重叠检查
                # 如果两个症状名称共享大量字符但并非同义（如"腹泻"vs"腹水"），跳过
                if not _is_valid_symptom_match(symptom_name, matched_name):
                    logger.debug(
                        f"[向量召回] 过滤噪声匹配: '{symptom_name}' → '{matched_name}' (score={score:.2f})"
                    )
                    continue

                matches.append({
                    "symptom_name": matched_name,
                    "category": hit["entity"]["category"],
                    "score": score,
                    "source": "vector_recall",
                })
            logger.debug(f"[向量召回] '{symptom_name}' 召回 {len(matches)} 个有效结果")
            return matches

        except Exception as e:
            logger.warning(f"[向量召回] 检索失败（Milvus可能未启动）: {e}")
            return []

    # ============================================================
    # 三层融合
    # ============================================================

    def _merge_results(
        self,
        original: dict,
        graph_matches: list[dict],
        vector_matches: list[dict],
    ) -> list[dict]:
        """
        三层结果融合策略：
        - 图谱精确匹配优先（置信度最高）
        - 向量语义召回次之（需通过严格过滤）
        - LLM 原始提取作为兜底
        - 宁缺毋滥：不确定的匹配不如不匹配
        """
        merged: list[dict] = []

        # 如果有图谱精确匹配，优先使用（最多取 1 个最佳匹配）
        if graph_matches:
            gm = graph_matches[0]  # 图谱精确匹配只有 1 个结果
            merged.append({
                "raw_text": original.get("raw_text", ""),
                "standard_name": gm["symptom_name"],
                "snomed_code": gm.get("snomed_code", ""),
                "category": gm.get("category", ""),
                "confidence": (
                    settings.SYMPTOM_GRAPH_WEIGHT * gm["confidence"]
                    + settings.SYMPTOM_LLM_WEIGHT * original.get("confidence", 0.5)
                ),
                "source": "graph_match",
                "questions": gm.get("questions", []),
            })
        elif vector_matches:
            # 没有图谱匹配，使用向量召回结果（最多取 1 个最高分）
            vm = vector_matches[0]
            merged.append({
                "raw_text": original.get("raw_text", ""),
                "standard_name": vm["symptom_name"],
                "category": vm.get("category", ""),
                "confidence": (
                    settings.SYMPTOM_VECTOR_WEIGHT * vm["score"]
                    + settings.SYMPTOM_LLM_WEIGHT * original.get("confidence", 0.5)
                ),
                "source": "vector_recall",
            })

        # 如果三层都没有结果，保留 LLM 原始提取
        if not merged:
            merged.append(original)

        return merged

    def _deduplicate(self, symptoms: list[dict]) -> list[dict]:
        """去重 - 同一标准症状保留最高置信度"""
        seen: dict[str, dict] = {}
        for s in symptoms:
            key = s.get("standard_name", "")
            if key:
                if key not in seen or s.get("confidence", 0) > seen[key].get("confidence", 0):
                    seen[key] = s
        return sorted(seen.values(), key=lambda x: x.get("confidence", 0), reverse=True)


# ============================================================
# 模块级辅助函数
# ============================================================

def _is_valid_symptom_match(query: str, candidate: str) -> bool:
    """
    判断向量召回的候选症状是否为有效匹配（而非字符重叠导致的噪声）。

    规则：
    1. 完全相同 → 有效
    2. 候选是 query 的同义词/别名（如"拉肚子"→"腹泻"）→ 有效
    3. 仅共享部分汉字但含义不同（如"腹泻"vs"腹水"）→ 无效
    """
    if query == candidate:
        return True

    # 计算字符重叠度：共享字符数 / 较短字符串长度
    query_chars = set(query)
    candidate_chars = set(candidate)
    overlap = query_chars & candidate_chars
    shorter_len = min(len(query), len(candidate))

    if shorter_len == 0:
        return False

    overlap_ratio = len(overlap) / shorter_len

    # 如果重叠率 > 80% 且长度相近，认为是有效匹配（同义词）
    # 否则认为是噪声（如"腹泻"和"腹水"共享"腹"字但含义完全不同）
    len_ratio = min(len(query), len(candidate)) / max(len(query), len(candidate))

    # 长度相近 + 高重叠 → 有效；否则 → 无效
    if len_ratio >= 0.5 and overlap_ratio >= 0.8:
        return True

    # 特殊情况：如果候选包含 query 的所有字符（如"腹泻"是"腹泻"的子串），有效
    if query in candidate or candidate in query:
        return True

    return False


# ============================================================
# 单独调试入口
# ============================================================

if __name__ == "__main__":
    """单独测试症状标准化流水线"""
    import asyncio

    async def test_pipeline():
        pipeline = SymptomNormalizePipeline()

        # 测试用例
        test_cases = [
            "肚子左边一直疼，吃完饭更疼",
            "头疼三天了，还有点发烧",
            "嗓子疼，吞咽的时候特别疼",
            "胸口闷，喘不上气",
        ]

        for text in test_cases:
            print(f"\n输入: {text}")
            result = await pipeline.run(text)
            for s in result:
                print(f"  → {s.get('standard_name', '')} (置信度={s.get('confidence', 0):.2f}, 来源={s.get('source', '')})")

    asyncio.run(test_pipeline())
