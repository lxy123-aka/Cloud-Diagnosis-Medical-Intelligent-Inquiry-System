# 训练数据目录

> ⚠️ **声明**：本目录下所有数据均为**模拟生成**，不包含任何真实病历或患者信息。
> 仅用于系统流程验证和开发调试，不可用于临床决策。

## 目录结构

```
data/
├── README.md                    # 本文件
├── generate_sample_data.py      # 示例数据生成脚本
├── sample_ner.json              # 示例 NER 标注数据（症状-疾病）
└── sample_dialogue.json         # 示例问诊对话数据
```

## 数据说明

### sample_ner.json — 症状-疾病 NER 标注

用于训练/评估症状识别和疾病分类模型。

格式：
```json
{
  "text": "患者头痛发热三天，伴有咳嗽",
  "entities": [
    {"text": "头痛", "label": "SYMPTOM", "standard_name": "头痛"},
    {"text": "发热", "label": "SYMPTOM", "standard_name": "发热"},
    {"text": "咳嗽", "label": "SYMPTOM", "standard_name": "咳嗽"}
  ],
  "diagnosis": "上呼吸道感染"
}
```

### sample_dialogue.json — 问诊对话数据

用于训练/评估多轮问诊对话模型。

格式：
```json
{
  "session_id": "sim_001",
  "chief_complaint": "头痛发热三天",
  "dialogue": [
    {"role": "patient", "content": "我头痛发热三天了"},
    {"role": "doctor", "content": "请问最高体温多少？"},
    ...
  ],
  "final_diagnosis": "上呼吸道感染"
}
```

## 生成方式

```bash
python data/generate_sample_data.py
```

## 数据来源

- 症状名称：来自 `configs/neo4j_symptoms.py` 的 200 个标准症状
- 疾病名称：来自 `configs/neo4j_diseases.py` 的 200 个常见疾病
- 对话模板：基于常见问诊流程人工设计规则生成
- **所有数据均为模拟，不对应任何真实患者**
