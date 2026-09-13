"""
pipeline/diagnosis_engine.py
============================
置信度驱动诊断收敛引擎。

职责：
  1. 读取标准化症状列表
  2. 查询 Neo4j 知识图谱获取候选疾病集合
  3. 多维症状加权计算疾病置信度
  4. 实现收敛判断：
     - 双条件收敛：Top1 置信度 ≥ 70% 或 Top1 - Top2 差值 ≥ 30%
     - 最小追问轮数门控（至少 3 轮）+ 最大轮数强制收敛（10 轮兜底）
  5. 返回 top 候选疾病、置信度差值，判断问诊是否结束
"""

from __future__ import annotations
import json
import re
from loguru import logger
from utils.fallback_llm import FallbackChatOpenAI as ChatOpenAI

from configs.settings import settings


class DiagnosisEngine:
    """置信度驱动的诊断收敛引擎"""

    # 类级别 Neo4j driver 单例，所有实例共享
    _neo4j_driver = None

    def __init__(self):
        self._llm = ChatOpenAI(
            temperature=0.2,
        )

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
    async def close(cls):
        """关闭 Neo4j driver 连接（程序退出时调用）"""
        if cls._neo4j_driver is not None:
            await cls._neo4j_driver.close()
            cls._neo4j_driver = None

    async def evaluate(
        self,
        symptoms: list[dict],
        current_round: int,
    ) -> tuple[list[dict], bool, str]:
        """
        评估当前诊断状态。

        :param symptoms: 当前已收集的标准症状列表
        :param current_round: 当前追问轮数
        :return: (候选疾病列表, 是否收敛, 最终诊断名称)
        """
        symptom_names = [s.get("standard_name", "") for s in symptoms]
        logger.info(f"[收敛引擎] 第{current_round}轮评估，症状: {symptom_names}")

        # 1. 从知识图谱获取候选疾病
        diseases = await self._query_diseases_from_graph(symptom_names)
        if not diseases:
            logger.warning("[收敛引擎] 知识图谱未找到匹配疾病，使用LLM推理")
            diseases = await self._llm_fallback_diagnosis(symptom_names)

        if not diseases:
            return [], False, ""

        # 2. 获取症状共现关系用于加权
        co_occurrences = await self._get_co_occurrences(symptom_names)

        # 3. 多维加权计算置信度
        candidates = self._calculate_confidence(diseases, symptoms, co_occurrences)

        # 4. 检查收敛条件
        converged, diagnosis = self._check_convergence(candidates, current_round)

        if converged:
            logger.info(f"[收敛引擎] 诊断收敛: {diagnosis}")
        else:
            top2_conf = candidates[1].get("confidence", 0) if len(candidates) > 1 else 0
            logger.info(
                f"[收敛引擎] 未收敛，Top1={candidates[0].get('confidence', 0):.2f} "
                f"Top2={top2_conf:.2f} 轮数={current_round}/{settings.MAX_INQUIRY_ROUNDS}"
            )

        return candidates, converged, diagnosis

    # ============================================================
    # Neo4j 知识图谱查询
    # ============================================================

    async def _query_diseases_from_graph(self, symptom_names: list[str]) -> list[dict]:
        """从 Neo4j 知识图谱查询关联疾病"""
        try:
            driver = self._get_driver()
            async with driver.session() as session:
                result = await session.run(
                    """
                    MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
                    WHERE s.name IN $symptom_names
                    WITH d, COLLECT(s.name) AS matched_symptoms, COUNT(s) AS match_count
                    OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(all_s:Symptom)
                    WITH d, matched_symptoms, match_count, COUNT(all_s) AS total_symptoms
                    RETURN d.name AS disease_name,
                           d.icd_code AS icd_code,
                           d.department AS department,
                           matched_symptoms,
                           match_count,
                           total_symptoms,
                           toFloat(match_count) / total_symptoms AS symptom_coverage
                    ORDER BY match_count DESC, symptom_coverage DESC
                    LIMIT 20
                    """,
                    {"symptom_names": symptom_names},
                )
                records = await result.data()
                logger.debug(f"[收敛引擎] 图谱查询: 输入{symptom_names}，返回{len(records)}条")

            return records

        except Exception as e:
            logger.warning(f"[收敛引擎] Neo4j查询失败: {e}")
            return []

    async def _get_co_occurrences(self, symptom_names: list[str]) -> list[dict]:
        """获取症状共现关系"""
        try:
            driver = self._get_driver()
            async with driver.session() as session:
                result = await session.run(
                    """
                    MATCH (s1:Symptom)-[r:CO_OCCURS_WITH]->(s2:Symptom)
                    WHERE s1.name IN $symptom_names AND s2.name IN $symptom_names
                    RETURN s1.name AS symptom1, s2.name AS symptom2,
                           r.weight AS weight
                    """,
                    {"symptom_names": symptom_names},
                )
                records = await result.data()

            return records

        except Exception as e:
            logger.warning(f"[收敛引擎] 共现关系查询失败: {e}")
            return []

    # ============================================================
    # LLM 降级推理（图谱不可用时）
    # ============================================================

    async def _llm_fallback_diagnosis(self, symptom_names: list[str]) -> list[dict]:
        """当 Neo4j 不可用时，使用 RAG + LLM 推理候选疾病"""
        try:
            # 先从 Milvus 检索相关疾病知识作为上下文
            rag_context = ""
            try:
                from utils.rag_retrieval import rag_retriever
                rag_context = await rag_retriever.hybrid_retrieve(
                    " ".join(symptom_names), top_k=5
                )
                if rag_context:
                    logger.info(f"[收敛引擎] RAG检索到 {len(rag_context)} 字符的医学知识")
            except Exception as e:
                logger.debug(f"[收敛引擎] RAG检索跳过: {e}")

            # 构造 prompt（包含 RAG 上下文 + 常见疾病先验）
            system_content = (
                "你是一位全科医生。根据症状列表"
            )
            if rag_context:
                system_content += "和以下参考资料"
            system_content += (
                "，给出最可能的候选疾病。\n\n"
                "⚠️ 重要原则（必须遵守）：\n"
                "1. 优先考虑常见病、多发病：如急性胃肠炎、上呼吸道感染、过敏性鼻炎、"
                "胃食管反流、功能性消化不良等，这些疾病的发病率远高于罕见病\n"
                "2. 不要过度诊断罕见重症：除非症状高度特异，否则不要将胰腺癌、"
                "小肠淋巴管扩张症等罕见病排在前面\n"
                "3. 遵循「常见优先」原则：同样症状匹配度下，常见病应排在罕见病前面\n\n"
                "输出 JSON 数组，每个元素包含：\n"
                '{"disease_name": "疾病名", "icd_code": "ICD编码", '
                '"department": "科室", "matched_symptoms": ["症状"], '
                '"match_count": 数字, "total_symptoms": 数字, '
                '"symptom_coverage": 0-1的浮点数}\n'
                "按可能性从高到低排列，最多5个。只输出JSON。"
            )
            if rag_context:
                system_content += "\n请优先参考以下资料中的疾病信息：\n" + rag_context

            response = await self._llm.ainvoke([
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"症状列表：{symptom_names}"},
            ])

            text = response.content
            # 提取 JSON
            try:
                json_match = re.search(r'\[.*\]', text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                return []
            except json.JSONDecodeError as e:
                logger.warning(f"[收敛引擎] LLM返回JSON解析失败: {e}")
                return []
        except Exception as e:
            logger.error(f"[收敛引擎] LLM降级推理失败: {e}")
            return []

    # ============================================================
    # 多维加权置信度计算
    # ============================================================

    def _calculate_confidence(
        self,
        diseases: list[dict],
        symptoms: list[dict],
        co_occurrences: list[dict],
    ) -> list[dict]:
        """
        多维加权置信度计算。
        维度1：症状覆盖率（0-0.5）
        维度2：症状置信度加权（0-0.3）
        维度3：共现加分（0-0.2）
        """
        # 构建共现权重映射
        co_weight_map: dict[tuple[str, str], float] = {}
        for co in co_occurrences:
            key = (co["symptom1"], co["symptom2"])
            co_weight_map[key] = co.get("weight", 0.5)
            co_weight_map[(co["symptom2"], co["symptom1"])] = co.get("weight", 0.5)

        candidates = []
        for d in diseases:
            matched = d.get("matched_symptoms", [])
            symptom_coverage = d.get("symptom_coverage", 0.0)

            # 维度1：症状覆盖率得分（0-0.5）
            coverage_score = symptom_coverage * 0.5

            # 维度2：症状置信度加权（0-0.3）
            symptom_conf_sum = 0.0
            for ms in matched:
                for s in symptoms:
                    if s.get("standard_name", "") == ms:
                        symptom_conf_sum += s.get("confidence", 0.5)
                        break
            avg_sym_conf = symptom_conf_sum / max(len(matched), 1)
            conf_score = avg_sym_conf * 0.3

            # 维度3：共现加分（0-0.2）
            co_score = 0.0
            for i, s1 in enumerate(matched):
                for s2 in matched[i + 1:]:
                    co_score += co_weight_map.get((s1, s2), 0.0)
            max_possible_co = len(matched) * (len(matched) - 1) / 2 if len(matched) > 1 else 1
            co_score = min(co_score / max(max_possible_co, 1), 1.0) * 0.2

            total_confidence = min(coverage_score + conf_score + co_score, 1.0)

            candidates.append({
                "disease_name": d.get("disease_name", ""),
                "icd_code": d.get("icd_code", ""),
                "department": d.get("department", ""),
                "confidence": round(total_confidence, 4),
                "matched_symptoms": matched,
                "total_symptoms": d.get("total_symptoms", 0),
                "symptom_coverage": symptom_coverage,
                "evidence_paths": [f"{s} -> {d.get('disease_name', '')}" for s in matched],
            })

        # 按置信度降序排列
        candidates.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return candidates

    # ============================================================
    # 收敛判断
    # ============================================================

    def _check_convergence(
        self,
        candidates: list[dict],
        current_round: int,
    ) -> tuple[bool, str]:
        """
        检查是否满足收敛条件。

        收敛策略：双条件收敛 + 最小轮数门控 + 最大轮数兜底
          - 双条件（满足任一即收敛）：
            条件1：Top1 置信度 ≥ 70%
            条件2：Top1 - Top2 差值 ≥ 30%
          - 过程控制：
            最小追问轮数门控：未达到 MIN_INQUIRY_ROUNDS 不允许收敛
            最大追问轮数兜底：达到 MAX_INQUIRY_ROUNDS 强制收敛
        """
        if not candidates:
            return False, ""

        top1 = candidates[0]
        top1_conf = top1.get("confidence", 0)

        # 条件1：未达到最小轮数，不允许收敛（强制继续追问）
        if current_round < settings.MIN_INQUIRY_ROUNDS:
            logger.info(
                f"[收敛引擎] 当前第{current_round}轮，未达到最小轮数{settings.MIN_INQUIRY_ROUNDS}，继续追问"
            )
            return False, ""

        # 条件2：Top1 置信度 ≥ 阈值
        if top1_conf >= settings.CONFIDENCE_THRESHOLD:
            return True, top1.get("disease_name", "")

        # 条件3：Top1 - Top2 差值 ≥ 阈值
        if len(candidates) >= 2:
            gap = top1_conf - candidates[1].get("confidence", 0)
            if gap >= settings.CONFIDENCE_GAP:
                return True, top1.get("disease_name", "")

        # 条件4：达到最大追问轮数（强制收敛）
        if current_round >= settings.MAX_INQUIRY_ROUNDS:
            logger.warning(
                f"[收敛引擎] 达到最大轮数{settings.MAX_INQUIRY_ROUNDS}，强制收敛"
            )
            return True, top1.get("disease_name", "")

        return False, ""


# ============================================================
# 单独调试入口
# ============================================================

if __name__ == "__main__":
    """单独测试诊断收敛引擎"""
    import asyncio

    async def test_engine():
        engine = DiagnosisEngine()

        # 模拟已标准化的症状
        symptoms = [
            {"standard_name": "头痛", "confidence": 0.9, "source": "graph_match"},
            {"standard_name": "发热", "confidence": 0.85, "source": "graph_match"},
            {"standard_name": "咽痛", "confidence": 0.8, "source": "llm_extract"},
        ]

        candidates, converged, diagnosis = await engine.evaluate(symptoms, current_round=3)

        print(f"\n候选疾病:")
        for c in candidates[:5]:
            print(f"  {c['disease_name']}: {c['confidence']:.2f} (匹配: {c['matched_symptoms']})")
        print(f"\n收敛: {converged}")
        print(f"最终诊断: {diagnosis}")

    asyncio.run(test_engine())
