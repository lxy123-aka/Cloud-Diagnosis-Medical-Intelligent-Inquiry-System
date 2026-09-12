// ============================================================
// configs/neo4j_init.cypher
// 云诊医疗智能问诊系统 —— 医学知识图谱初始化脚本
// 节点：Disease(疾病)、Symptom(症状)、Drug(药品)、Department(科室)
// 关系：HAS_SYMPTOM(疾病-症状)、TREATS(药品-疾病)、BELONGS_TO(疾病-科室)
// 执行方式：neo4j browser 粘贴执行，或 neo4j-shell < neo4j_init.cypher
// ============================================================

// ============================================================
// 1. 创建约束和索引（加速查询）
// ============================================================

CREATE CONSTRAINT disease_name_unique IF NOT EXISTS
FOR (d:Disease) REQUIRE d.name IS UNIQUE;

CREATE CONSTRAINT symptom_name_unique IF NOT EXISTS
FOR (s:Symptom) REQUIRE s.name IS UNIQUE;

CREATE CONSTRAINT drug_name_unique IF NOT EXISTS
FOR (drug:Drug) REQUIRE drug.name IS UNIQUE;

CREATE CONSTRAINT department_name_unique IF NOT EXISTS
FOR (dept:Department) REQUIRE dept.name IS UNIQUE;

// ============================================================
// 2. 创建科室节点 (Department)
// ============================================================

MERGE (dept1:Department {name: "呼吸内科"})
  ON CREATE SET dept1.description = "呼吸系统疾病诊治";

MERGE (dept2:Department {name: "消化内科"})
  ON CREATE SET dept2.description = "消化系统疾病诊治";

MERGE (dept3:Department {name: "心血管内科"})
  ON CREATE SET dept3.description = "心血管系统疾病诊治";

MERGE (dept4:Department {name: "内分泌科"})
  ON CREATE SET dept4.description = "内分泌及代谢疾病诊治";

MERGE (dept5:Department {name: "神经内科"})
  ON CREATE SET dept5.description = "神经系统疾病诊治";

MERGE (dept6:Department {name: "全科"})
  ON CREATE SET dept6.description = "常见病初步诊断与分诊";

// ============================================================
// 3. 创建症状节点 (Symptom)
// ============================================================

// 呼吸系统症状
MERGE (s1:Symptom {name: "头痛"})
  ON CREATE SET s1.snomed_code = "25064002",
                s1.category = "全身症状",
                s1.synonyms = ["头疼", "头部疼痛", "头胀痛"],
                s1.associated_questions = ["头痛持续多久了？", "是持续性还是间歇性？", "哪个部位最痛？"];

MERGE (s2:Symptom {name: "发热"})
  ON CREATE SET s2.snomed_code = "386661006",
                s2.category = "全身症状",
                s2.synonyms = ["发烧", "体温升高", "高热"],
                s2.associated_questions = ["最高体温多少度？", "发热几天了？", "有没有畏寒？"];

MERGE (s3:Symptom {name: "咳嗽"})
  ON CREATE SET s3.snomed_code = "49727002",
                s3.category = "呼吸系统",
                s3.synonyms = ["咳", "咳嗽不止"],
                s3.associated_questions = ["干咳还是有痰？", "痰是什么颜色？", "夜间咳嗽更严重吗？"];

MERGE (s4:Symptom {name: "咽痛"})
  ON CREATE SET s4.snomed_code = "162397003",
                s4.category = "呼吸系统",
                s4.synonyms = ["嗓子疼", "咽喉痛", "喉咙痛"],
                s4.associated_questions = ["吞咽时疼痛加重吗？", "有没有声音嘶哑？"];

MERGE (s5:Symptom {name: "胸闷"})
  ON CREATE SET s5.snomed_code = "29857009",
                s5.category = "呼吸系统",
                s5.synonyms = ["胸口闷", "胸闷气短"],
                s5.associated_questions = ["活动后加重吗？", "有没有伴随胸痛？"];

// 消化系统症状
MERGE (s6:Symptom {name: "腹痛"})
  ON CREATE SET s6.snomed_code = "21522001",
                s6.category = "消化系统",
                s6.synonyms = ["肚子疼", "胃痛", "腹部疼痛"],
                s6.associated_questions = ["哪个位置最痛？", "与进食有关吗？", "疼痛性质是胀痛还是绞痛？"];

MERGE (s7:Symptom {name: "恶心"})
  ON CREATE SET s7.snomed_code = "422587007",
                s7.category = "消化系统",
                s7.synonyms = ["想吐", "反胃"],
                s7.associated_questions = ["有没有呕吐？", "与进食有关吗？"];

MERGE (s8:Symptom {name: "腹泻"})
  ON CREATE SET s8.snomed_code = "62315008",
                s8.category = "消化系统",
                s8.synonyms = ["拉肚子", "大便稀"],
                s8.associated_questions = ["每天几次？", "有没有脓血？", "持续几天了？"];

// 心血管/全身症状
MERGE (s9:Symptom {name: "心悸"})
  ON CREATE SET s9.snomed_code = "80313002",
                s9.category = "心血管系统",
                s9.synonyms = ["心跳快", "心慌", "心跳加速"],
                s9.associated_questions = ["什么情况下出现？", "持续多久？", "有没有伴随胸痛？"];

MERGE (s10:Symptom {name: "头晕"})
  ON CREATE SET s10.snomed_code = "404640003",
                s10.category = "神经系统",
                s10.synonyms = ["头昏", "眩晕"],
                s10.associated_questions = ["是旋转感还是昏沉感？", "改变体位时加重吗？"];

// ============================================================
// 4. 创建疾病节点 (Disease)
// ============================================================

MERGE (d1:Disease {name: "上呼吸道感染"})
  ON CREATE SET d1.icd_code = "J06.9",
                d1.department = "呼吸内科",
                d1.description = "由病毒或细菌引起的鼻、咽、喉部急性炎症",
                d1.typical_symptoms = ["头痛", "发热", "咳嗽", "咽痛"],
                d1.recommended_examinations = ["血常规", "CRP", "胸部X线"],
                d1.treatment_options = ["对症治疗", "抗病毒药物", "抗生素（细菌感染时）"];

MERGE (d2:Disease {name: "急性支气管炎"})
  ON CREATE SET d2.icd_code = "J20.9",
                d2.department = "呼吸内科",
                d2.description = "支气管黏膜的急性炎症",
                d2.typical_symptoms = ["咳嗽", "发热", "胸闷"],
                d2.recommended_examinations = ["血常规", "胸部X线", "痰培养"],
                d2.treatment_options = ["止咳化痰", "抗感染", "雾化吸入"];

MERGE (d3:Disease {name: "急性胃肠炎"})
  ON CREATE SET d3.icd_code = "K52.9",
                d3.department = "消化内科",
                d3.description = "胃黏膜和肠黏膜的急性炎症",
                d3.typical_symptoms = ["腹痛", "腹泻", "恶心"],
                d3.recommended_examinations = ["血常规", "大便常规", "电解质"],
                d3.treatment_options = ["补液", "止泻", "抗感染"];

MERGE (d4:Disease {name: "高血压"})
  ON CREATE SET d4.icd_code = "I10",
                d4.department = "心血管内科",
                d4.description = "以体循环动脉压升高为主要特征的临床综合征",
                d4.typical_symptoms = ["头痛", "头晕", "心悸"],
                d4.recommended_examinations = ["血压监测", "心电图", "肾功能", "眼底检查"],
                d4.treatment_options = ["生活方式干预", "降压药物治疗"];

MERGE (d5:Disease {name: "2型糖尿病"})
  ON CREATE SET d5.icd_code = "E11",
                d5.department = "内分泌科",
                d5.description = "胰岛素抵抗和相对胰岛素分泌不足导致的代谢性疾病",
                d5.typical_symptoms = ["头晕", "心悸"],
                d5.recommended_examinations = ["空腹血糖", "糖化血红蛋白", "OGTT", "胰岛素释放试验"],
                d5.treatment_options = ["饮食控制", "运动", "口服降糖药", "胰岛素"];

MERGE (d6:Disease {name: "偏头痛"})
  ON CREATE SET d6.icd_code = "G43",
                d6.department = "神经内科",
                d6.description = "反复发作的一侧搏动性头痛",
                d6.typical_symptoms = ["头痛", "恶心", "头晕"],
                d6.recommended_examinations = ["头颅MRI", "脑电图"],
                d6.treatment_options = ["急性期止痛", "预防性用药", "避免诱因"];

// ============================================================
// 5. 创建药品节点 (Drug)
// ============================================================

MERGE (drug1:Drug {name: "阿莫西林"})
  ON CREATE SET drug1.category = "青霉素类抗生素",
                drug1.indications = ["上呼吸道感染", "中耳炎", "泌尿道感染"],
                drug1.dosage = "0.5g/次，每8小时一次",
                drug1.contraindications = ["青霉素过敏者禁用"],
                drug1.side_effects = ["皮疹", "腹泻", "恶心"],
                drug1.pregnancy_category = "B";

MERGE (drug2:Drug {name: "布洛芬"})
  ON CREATE SET drug2.category = "非甾体抗炎药",
                drug2.indications = ["发热", "头痛", "关节痛", "痛经"],
                drug2.dosage = "0.2-0.4g/次，每4-6小时一次",
                drug2.contraindications = ["活动性消化道溃疡", "严重肝肾功能不全"],
                drug2.side_effects = ["胃肠道不适", "头晕", "皮疹"],
                drug2.pregnancy_category = "B/D";

MERGE (drug3:Drug {name: "头孢克洛"})
  ON CREATE SET drug3.category = "二代头孢菌素",
                drug3.indications = ["呼吸道感染", "尿路感染", "中耳炎"],
                drug3.dosage = "0.25g/次，每8小时一次",
                drug3.contraindications = ["头孢类过敏者禁用"],
                drug3.side_effects = ["腹泻", "恶心", "皮疹"],
                drug3.pregnancy_category = "B";

MERGE (drug4:Drug {name: "奥美拉唑"})
  ON CREATE SET drug4.category = "质子泵抑制剂",
                drug4.indications = ["胃溃疡", "胃食管反流病"],
                drug4.dosage = "20mg/次，每日1-2次",
                drug4.contraindications = ["对本品过敏者禁用"],
                drug4.side_effects = ["头痛", "腹泻", "恶心"],
                drug4.pregnancy_category = "C";

MERGE (drug5:Drug {name: "二甲双胍"})
  ON CREATE SET drug5.category = "双胍类降糖药",
                drug5.indications = ["2型糖尿病"],
                drug5.dosage = "0.5g/次，每日2-3次",
                drug5.contraindications = ["严重肾功能不全", "代谢性酸中毒"],
                drug5.side_effects = ["胃肠道反应", "乳酸酸中毒（罕见）"],
                drug5.pregnancy_category = "B";

MERGE (drug6:Drug {name: "氨氯地平"})
  ON CREATE SET drug6.category = "钙通道阻滞剂",
                drug6.indications = ["高血压", "心绞痛"],
                drug6.dosage = "5mg/次，每日1次",
                drug6.contraindications = ["严重低血压", "主动脉瓣狭窄"],
                drug6.side_effects = ["水肿", "头晕", "面部潮红"],
                drug6.pregnancy_category = "C";

// ============================================================
// 6. 创建关系：疾病 -[HAS_SYMPTOM]-> 症状（带权重）
// ============================================================

// 上呼吸道感染
MERGE (d1)-[r:HAS_SYMPTOM]->(s1) ON CREATE SET r.weight = 0.8, r.frequency = 0.7;
MERGE (d1)-[r:HAS_SYMPTOM]->(s2) ON CREATE SET r.weight = 0.9, r.frequency = 0.8;
MERGE (d1)-[r:HAS_SYMPTOM]->(s3) ON CREATE SET r.weight = 0.7, r.frequency = 0.6;
MERGE (d1)-[r:HAS_SYMPTOM]->(s4) ON CREATE SET r.weight = 0.85, r.frequency = 0.75;

// 急性支气管炎
MERGE (d2)-[r:HAS_SYMPTOM]->(s3) ON CREATE SET r.weight = 0.95, r.frequency = 0.9;
MERGE (d2)-[r:HAS_SYMPTOM]->(s2) ON CREATE SET r.weight = 0.7, r.frequency = 0.6;
MERGE (d2)-[r:HAS_SYMPTOM]->(s5) ON CREATE SET r.weight = 0.6, r.frequency = 0.5;

// 急性胃肠炎
MERGE (d3)-[r:HAS_SYMPTOM]->(s6) ON CREATE SET r.weight = 0.9, r.frequency = 0.85;
MERGE (d3)-[r:HAS_SYMPTOM]->(s8) ON CREATE SET r.weight = 0.95, r.frequency = 0.9;
MERGE (d3)-[r:HAS_SYMPTOM]->(s7) ON CREATE SET r.weight = 0.8, r.frequency = 0.7;

// 高血压
MERGE (d4)-[r:HAS_SYMPTOM]->(s1) ON CREATE SET r.weight = 0.6, r.frequency = 0.5;
MERGE (d4)-[r:HAS_SYMPTOM]->(s10) ON CREATE SET r.weight = 0.7, r.frequency = 0.6;
MERGE (d4)-[r:HAS_SYMPTOM]->(s9) ON CREATE SET r.weight = 0.65, r.frequency = 0.55;

// 2型糖尿病
MERGE (d5)-[r:HAS_SYMPTOM]->(s10) ON CREATE SET r.weight = 0.5, r.frequency = 0.4;
MERGE (d5)-[r:HAS_SYMPTOM]->(s9) ON CREATE SET r.weight = 0.55, r.frequency = 0.45;

// 偏头痛
MERGE (d6)-[r:HAS_SYMPTOM]->(s1) ON CREATE SET r.weight = 0.95, r.frequency = 0.95;
MERGE (d6)-[r:HAS_SYMPTOM]->(s7) ON CREATE SET r.weight = 0.6, r.frequency = 0.5;
MERGE (d6)-[r:HAS_SYMPTOM]->(s10) ON CREATE SET r.weight = 0.5, r.frequency = 0.4;

// ============================================================
// 7. 创建关系：疾病 -[BELONGS_TO]-> 科室
// ============================================================

MERGE (d1)-[:BELONGS_TO]->(dept1);  // 上呼吸道感染 → 呼吸内科
MERGE (d2)-[:BELONGS_TO]->(dept1);  // 急性支气管炎 → 呼吸内科
MERGE (d3)-[:BELONGS_TO]->(dept2);  // 急性胃肠炎 → 消化内科
MERGE (d4)-[:BELONGS_TO]->(dept3);  // 高血压 → 心血管内科
MERGE (d5)-[:BELONGS_TO]->(dept4);  // 2型糖尿病 → 内分泌科
MERGE (d6)-[:BELONGS_TO]->(dept5);  // 偏头痛 → 神经内科

// ============================================================
// 8. 创建关系：药品 -[TREATS]-> 疾病（带用法说明）
// ============================================================

MERGE (drug1)-[r:TREATS]->(d1) ON CREATE SET r.dosage = "0.5g/次，每8小时一次", r.route = "口服";
MERGE (drug1)-[r:TREATS]->(d2) ON CREATE SET r.dosage = "0.5g/次，每8小时一次", r.route = "口服";

MERGE (drug2)-[r:TREATS]->(d1) ON CREATE SET r.dosage = "0.2-0.4g/次，退热时用", r.route = "口服";
MERGE (drug2)-[r:TREATS]->(d6) ON CREATE SET r.dosage = "急性期止痛用", r.route = "口服";

MERGE (drug3)-[r:TREATS]->(d1) ON CREATE SET r.dosage = "0.25g/次，每8小时一次", r.route = "口服";
MERGE (drug3)-[r:TREATS]->(d2) ON CREATE SET r.dosage = "0.25g/次，每8小时一次", r.route = "口服";

MERGE (drug4)-[r:TREATS]->(d3) ON CREATE SET r.dosage = "20mg/次，每日1次", r.route = "口服";

MERGE (drug5)-[r:TREATS]->(d5) ON CREATE SET r.dosage = "0.5g/次，每日2-3次", r.route = "口服";

MERGE (drug6)-[r:TREATS]->(d4) ON CREATE SET r.dosage = "5mg/次，每日1次", r.route = "口服";

// ============================================================
// 9. 创建症状共现关系（用于诊断加权）
// ============================================================

MERGE (s1)-[r:CO_OCCURS_WITH]->(s2) ON CREATE SET r.weight = 0.7, r.frequency = 120;
MERGE (s1)-[r:CO_OCCURS_WITH]->(s4) ON CREATE SET r.weight = 0.65, r.frequency = 95;
MERGE (s2)-[r:CO_OCCURS_WITH]->(s3) ON CREATE SET r.weight = 0.6, r.frequency = 88;
MERGE (s3)-[r:CO_OCCURS_WITH]->(s5) ON CREATE SET r.weight = 0.55, r.frequency = 72;
MERGE (s6)-[r:CO_OCCURS_WITH]->(s8) ON CREATE SET r.weight = 0.8, r.frequency = 150;
MERGE (s6)-[r:CO_OCCURS_WITH]->(s7) ON CREATE SET r.weight = 0.7, r.frequency = 110;
MERGE (s1)-[r:CO_OCCURS_WITH]->(s10) ON CREATE SET r.weight = 0.6, r.frequency = 85;
MERGE (s9)-[r:CO_OCCURS_WITH]->(s10) ON CREATE SET r.weight = 0.5, r.frequency = 65;

// ============================================================
// 10. 验证脚本（可选执行）
// ============================================================

// 统计节点数量
MATCH (n) RETURN labels(n) AS node_type, count(n) AS count ORDER BY count DESC;

// 统计关系数量
MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS count ORDER BY count DESC;

// 示例查询：根据症状查找疾病
MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
WHERE s.name IN ["头痛", "发热", "咽痛"]
RETURN d.name AS disease, collect(s.name) AS matched_symptoms, count(s) AS match_count
ORDER BY match_count DESC;
