# 训练数据目录

> ⚠️ **声明**：本目录下 `sample_*.json` 均为**模拟生成**，不包含任何真实病历或患者信息。
> 仅用于系统流程验证和开发调试，不可用于临床决策。
> 正式训练数据见下方“正式训练数据来源”章节。

## 目录结构

```
data/
├── README.md                    # 本文件
├── generate_sample_data.py      # 示例数据生成脚本
├── sample_ner.json              # 示例 NER 标注数据（症状-疾病，12 条）
└── sample_dialogue.json         # 示例问诊对话数据
```

## 数据说明

### sample_ner.json — 症状-疾病 NER 标注

用于训练/评估症状识别和疾病分类模型。共 12 条样本（6 条标准 + 6 条口语化），涵盖 6 种常见疾病。

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

### 示例数据（sample_*.json）

- 症状名称：来自 `configs/neo4j_symptoms.py` 的 200 个标准症状
- 疾病名称：来自 `configs/neo4j_diseases.py` 的 200 个常见疾病
- 对话模板：基于常见问诊流程人工设计规则生成
- **所有示例数据均为模拟，不对应任何真实患者**

### 正式训练数据来源

正式训练加载**国家标准医疗数据集**，具体包括：

| 数据类型 | 来源 | 说明 |
|----------|------|------|
| 医疗 NER 语料 | 公开医疗 NER 数据集 / 国家标准数据 | 包含症状、疾病、药物、检查、指标等实体标注 |
| 医学知识问答 | 公开医学知识库 | 疾病科普、用药指南、检查解读等 |
| 问诊对话数据 | 标准问诊流程模板 | 多轮问诊对话，覆盖五大业务场景 |

#### 数据处理流程

1. **原始格式**：BIO 标注格式（实体边界 + 类型）
2. **格式转换**：由 `train/data_processor.py` 将 BIO 标注转换为 Alpaca 格式（instruction/input/output）
3. **数据清洗**：去除空值、格式错误、超长文本（`DataCleaner` 类）
4. **数据集划分**：训练集 90% / 验证集 10%（`split_dataset` 函数）
5. **灌入方式**：通过 LlamaFactory 的 `dataset_info.json` 注册数据集，训练时自动加载

#### 训练配置

- **基座模型**：Qwen3.5-4B
- **微调方法**：LoRA SFT（r=16, α=32, 全线性层注入）
- **训练时长**：单张 RTX4090 24G 显卡约 6 小时完成训练
- **训练监控**：SwanLab 实时记录训练指标
- **训练命令**：`python train/sft_lora_train.py`
