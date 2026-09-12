"""
configs/neo4j_init.py
=====================
Neo4j 知识图谱初始化脚本。

功能：
  1. 创建 200 个症状节点
  2. 创建 200 个疾病节点
  3. 创建 200 个药品节点
  4. 创建 200+ 条关系（HAS_SYMPTOM / TREATS / BELONGS_TO / CO_OCCURS_WITH）

运行方式：
  python configs/neo4j_init.py
"""

from __future__ import annotations
import sys
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import settings
from configs.neo4j_symptoms import SYMPTOMS
from configs.neo4j_diseases import DISEASES
from configs.neo4j_drugs import DRUGS


def init_neo4j():
    """初始化 Neo4j 知识图谱"""
    from neo4j import GraphDatabase

    uri = settings.NEO4J_URI
    user = settings.NEO4J_USER
    password = settings.NEO4J_PASSWORD

    logger.info(f"连接 Neo4j: {uri}")
    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        # ============================================================
        # 1. 清空旧数据
        # ============================================================
        logger.info("清空旧数据...")
        session.run("MATCH (n) DETACH DELETE n").consume()
        logger.info("旧数据已清空")

        # ============================================================
        # 2. 创建症状节点 (200 个)
        # ============================================================
        logger.info(f"创建 {len(SYMPTOMS)} 个症状节点...")
        for i, (name, snomed, category, *rest) in enumerate(SYMPTOMS):
            aliases = rest[0] if len(rest) > 0 else []
            questions = rest[1] if len(rest) > 1 else []
            session.run(
                """MERGE (s:Symptom {name: $name})
                   SET s.snomed_code = $snomed,
                       s.category = $category,
                       s.aliases = $aliases,
                       s.questions = $questions""",
                name=name, snomed=snomed, category=category,
                aliases=aliases, questions=questions,
            ).consume()
        logger.info(f"✅ {len(SYMPTOMS)} 个症状节点创建完成")

        # ============================================================
        # 3. 创建疾病节点 (200 个)
        # ============================================================
        logger.info(f"创建 {len(DISEASES)} 个疾病节点...")
        for i, (name, icd, dept, desc, symptoms, tests, treatments) in enumerate(DISEASES):
            session.run(
                """MERGE (d:Disease {name: $name})
                   SET d.icd_code = $icd,
                       d.department = $dept,
                       d.description = $desc,
                       d.typical_symptoms = $symptoms,
                       d.recommended_tests = $tests,
                       d.treatments = $treatments""",
                name=name, icd=icd, dept=dept, desc=desc,
                symptoms=symptoms, tests=tests, treatments=treatments,
            ).consume()
        logger.info(f"✅ {len(DISEASES)} 个疾病节点创建完成")

        # ============================================================
        # 4. 创建药品节点 (200 个)
        # ============================================================
        logger.info(f"创建 {len(DRUGS)} 个药品节点...")
        for i, (name, category, indications, dosage, contraindications, side_effects, pregnancy) in enumerate(DRUGS):
            session.run(
                """MERGE (dr:Drug {name: $name})
                   SET dr.category = $category,
                       dr.indications = $indications,
                       dr.dosage = $dosage,
                       dr.contraindications = $contraindications,
                       dr.side_effects = $side_effects,
                       dr.pregnancy_category = $pregnancy""",
                name=name, category=category, indications=indications,
                dosage=dosage, contraindications=contraindications,
                side_effects=side_effects, pregnancy=pregnancy,
            ).consume()
        logger.info(f"✅ {len(DRUGS)} 个药品节点创建完成")

        # ============================================================
        # 5. 创建关系
        # ============================================================
        # 5a. 疾病-症状关系 (HAS_SYMPTOM)
        symptom_set = {s[0] for s in SYMPTOMS}
        has_symptom_count = 0
        logger.info("创建疾病-症状关系...")
        for name, icd, dept, desc, symptoms, tests, treatments in DISEASES:
            for sym in symptoms:
                if sym in symptom_set:
                    session.run(
                        """MATCH (d:Disease {name: $d_name}), (s:Symptom {name: $s_name})
                           MERGE (d)-[r:HAS_SYMPTOM]->(s)""",
                        d_name=name, s_name=sym,
                    ).consume()
                    has_symptom_count += 1
        logger.info(f"✅ {has_symptom_count} 条 HAS_SYMPTOM 关系创建完成")

        # 5b. 疾病-药品关系 (TREATS) — 从疾病的治疗方案匹配药品
        drug_set = {d[0] for d in DRUGS}
        treats_count = 0
        logger.info("创建疾病-药品关系...")
        for name, icd, dept, desc, symptoms, tests, treatments in DISEASES:
            for treat in treatments:
                if treat in drug_set:
                    session.run(
                        """MATCH (d:Disease {name: $d_name}), (dr:Drug {name: $dr_name})
                           MERGE (d)-[r:TREATS]->(dr)""",
                        d_name=name, dr_name=treat,
                    ).consume()
                    treats_count += 1
        logger.info(f"✅ {treats_count} 条 TREATS 关系创建完成")

        # 5c. 药品-适应症关系 (INDICATES) — 从药品的适应症匹配疾病
        indicates_count = 0
        logger.info("创建药品-适应症关系...")
        for drug_name, category, indications, dosage, contraindications, side_effects, pregnancy in DRUGS:
            # 从药品适应症中提取疾病关键词进行匹配
            for dis_name, dis_icd, dis_dept, dis_desc, dis_symptoms, dis_tests, dis_treatments in DISEASES:
                # 如果药品适应症包含疾病名称
                if dis_name in indications:
                    session.run(
                        """MATCH (dr:Drug {name: $dr_name}), (d:Disease {name: $d_name})
                           MERGE (dr)-[r:INDICATES]->(d)""",
                        dr_name=drug_name, d_name=dis_name,
                    ).consume()
                    indicates_count += 1
        logger.info(f"✅ {indicates_count} 条 INDICATES 关系创建完成")

        # 5d. 疾病-科室关系 (BELONGS_TO) — 自动从疾病数据创建
        dept_count = 0
        logger.info("创建疾病-科室关系...")
        for name, icd, dept, desc, symptoms, tests, treatments in DISEASES:
            session.run(
                """MERGE (dp:Department {name: $dept})
                   WITH dp
                   MATCH (d:Disease {name: $name})
                   MERGE (d)-[r:BELONGS_TO]->(dp)""",
                dept=dept, name=name,
            ).consume()
            dept_count += 1
        logger.info(f"✅ {dept_count} 条 BELONGS_TO 关系创建完成")

        # 5e. 症状共现关系 (CO_OCCURS_WITH) — 同一种疾病的多个症状之间
        co_occur_count = 0
        logger.info("创建症状共现关系...")
        for name, icd, dept, desc, symptoms, tests, treatments in DISEASES:
            valid_syms = [s for s in symptoms if s in symptom_set]
            for i in range(len(valid_syms)):
                for j in range(i + 1, min(i + 3, len(valid_syms))):
                    session.run(
                        """MATCH (s1:Symptom {name: $s1}), (s2:Symptom {name: $s2})
                           MERGE (s1)-[r:CO_OCCURS_WITH]-(s2)
                           SET r.weight = 0.5""",
                        s1=valid_syms[i], s2=valid_syms[j],
                    ).consume()
                    co_occur_count += 1
        logger.info(f"✅ {co_occur_count} 条 CO_OCCURS_WITH 关系创建完成")

        # ============================================================
        # 6. 统计
        # ============================================================
        result = session.run("MATCH (s:Symptom) RETURN count(s) as cnt").single()
        symptom_cnt = result["cnt"]
        result = session.run("MATCH (d:Disease) RETURN count(d) as cnt").single()
        disease_cnt = result["cnt"]
        result = session.run("MATCH (dr:Drug) RETURN count(dr) as cnt").single()
        drug_cnt = result["cnt"]
        result = session.run("MATCH ()-[r]->() RETURN count(r) as cnt").single()
        rel_cnt = result["cnt"]
        result = session.run("MATCH (dp:Department) RETURN count(dp) as cnt").single()
        dept_cnt = result["cnt"]

        logger.info(f"\n{'='*50}")
        logger.info(f"  Neo4j 知识图谱统计")
        logger.info(f"{'='*50}")
        logger.info(f"  症状节点: {symptom_cnt}")
        logger.info(f"  疾病节点: {disease_cnt}")
        logger.info(f"  药品节点: {drug_cnt}")
        logger.info(f"  科室节点: {dept_cnt}")
        logger.info(f"  关系总数: {rel_cnt}")
        logger.info(f"{'='*50}")
        logger.info("✅ Neo4j 知识图谱初始化完成")

    driver.close()


if __name__ == "__main__":
    init_neo4j()
