"""
data/generate_sample_data.py
=============================
生成模拟训练数据（NER 标注 + 问诊对话对）。

⚠️ 所有数据均为模拟生成，不包含任何真实病历或患者信息。
仅用于系统流程验证和开发调试。

运行方式：
  python data/generate_sample_data.py
"""

from __future__ import annotations
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 模拟数据模板（人工设计，非真实病历）
# ============================================================

# 常见疾病 → 典型症状 → 对话模板
DISEASE_TEMPLATES = [
    {
        "disease": "上呼吸道感染",
        "chief_complaint": "头痛发热三天",
        "symptoms": ["头痛", "发热", "咳嗽", "咽痛", "鼻塞"],
        "dialogue": [
            ("patient", "医生，我头痛发热三天了"),
            ("doctor", "请问最高体温多少？"),
            ("patient", "大概38度左右"),
            ("doctor", "有没有咳嗽、嗓子疼？"),
            ("patient", "有点咳嗽，嗓子也疼"),
            ("doctor", "有没有流鼻涕？"),
            ("patient", "有的，鼻子不通气"),
            ("doctor", "有没有做过血常规检查？"),
            ("patient", "还没有"),
            ("doctor", "建议查个血常规，目前看考虑上呼吸道感染，先吃点退烧药和多喝水"),
        ],
    },
    {
        "disease": "急性胃肠炎",
        "chief_complaint": "腹痛腹泻两天",
        "symptoms": ["腹痛", "腹泻", "恶心", "呕吐", "食欲不振"],
        "dialogue": [
            ("patient", "医生，我肚子疼，拉了两天了"),
            ("doctor", "大便是什么性状的？水样还是糊状？"),
            ("patient", "水样的，一天拉了五六次"),
            ("doctor", "有没有恶心呕吐？"),
            ("patient", "有，吐了两次"),
            ("doctor", "最近吃了什么不干净的东西吗？"),
            ("patient", "前天晚上吃了顿火锅，第二天就开始不舒服了"),
            ("doctor", "有没有发热？"),
            ("patient", "没有发烧，就是肚子一直疼"),
            ("doctor", "考虑急性胃肠炎，注意补充水分，吃点蒙脱石散和益生菌"),
        ],
    },
    {
        "disease": "胃食管反流病",
        "chief_complaint": "反酸烧心一个月",
        "symptoms": ["反酸", "上腹痛", "嗳气", "口干"],
        "dialogue": [
            ("patient", "医生，我最近一个月老是反酸，烧心"),
            ("doctor", "什么时侯最明显？饭后还是空腹？"),
            ("patient", "吃完饭以后特别明显，尤其是吃辣的"),
            ("doctor", "晚上睡觉的时候有没有加重？"),
            ("patient", "有的，平躺的时候更难受"),
            ("doctor", "有没有打嗝？"),
            ("patient", "经常打嗝"),
            ("doctor", "做过胃镜吗？"),
            ("patient", "没有"),
            ("doctor", "建议做个胃镜，目前考虑胃食管反流，先吃奥美拉唑，少吃辛辣油腻"),
        ],
    },
    {
        "disease": "过敏性鼻炎",
        "chief_complaint": "打喷嚏流鼻涕一个月",
        "symptoms": ["打喷嚏", "流涕", "鼻塞", "嗅觉减退"],
        "dialogue": [
            ("patient", "医生，我最近老是打喷嚏，流清鼻涕"),
            ("doctor", "持续多长时间了？"),
            ("patient", "差不多一个月了"),
            ("doctor", "什么情况下会加重？"),
            ("patient", "早上起床和遇到灰尘的时候最厉害"),
            ("doctor", "鼻子痒不痒？"),
            ("patient", "特别痒"),
            ("doctor", "有没有眼睛痒？"),
            ("patient", "也有点"),
            ("doctor", "考虑过敏性鼻炎，可以用氯雷他定和鼻喷激素"),
        ],
    },
    {
        "disease": "高血压",
        "chief_complaint": "头晕头痛一周",
        "symptoms": ["头晕", "头痛", "心悸", "失眠"],
        "dialogue": [
            ("patient", "医生，我最近总是头晕头疼"),
            ("doctor", "血压量过吗？"),
            ("patient", "在家里量过，好像偏高"),
            ("doctor", "最高多少？"),
            ("patient", "高压160左右"),
            ("doctor", "有没有心慌、睡眠不好？"),
            ("patient", "有的，晚上睡不着，心跳也快"),
            ("doctor", "之前有高血压病史吗？"),
            ("patient", "以前体检说过偏高，但没吃药"),
            ("doctor", "需要开始规律服药了，先用氨氯地平，每天监测血压"),
        ],
    },
    {
        "disease": "2型糖尿病",
        "chief_complaint": "口渴多尿三个月",
        "symptoms": ["多饮", "多尿", "多食", "消瘦", "乏力"],
        "dialogue": [
            ("patient", "医生，我最近特别容易口渴，喝水多，尿也多"),
            ("doctor", "持续多长时间了？"),
            ("patient", "大概三个月了"),
            ("doctor", "饭量有没有变化？"),
            ("patient", "吃得比以前多了，但体重反而下降了"),
            ("doctor", "瘦了多少斤？"),
            ("patient", "大概十斤左右"),
            ("doctor", "有没有查过血糖？"),
            ("patient", "没有查过"),
            ("doctor", "需要查空腹血糖和糖化血红蛋白，怀疑糖尿病"),
        ],
    },
]

# 口语化变体映射（用于生成更自然的 NER 数据）
COLLOQUIAL_VARIANTS = {
    "头痛": ["头疼", "脑袋疼", "头胀"],
    "发热": ["发烧", "体温高", "烧"],
    "咳嗽": ["咳", "咳咳的"],
    "腹泻": ["拉肚子", "拉稀", "大便稀"],
    "腹痛": ["肚子疼", "肚肚痛"],
    "恶心": ["想吐", "反胃", "恶心心"],
    "咽痛": ["嗓子疼", "喉咙痛"],
    "鼻塞": ["鼻子不通", "鼻子堵了"],
    "反酸": ["胃酸", "烧心"],
    "头晕": ["头昏", "晕乎乎"],
}


def generate_ner_data() -> list[dict]:
    """生成 NER 标注数据"""
    ner_samples = []

    for template in DISEASE_TEMPLATES:
        # 标准版本
        symptoms_text = "、".join(template["symptoms"][:3])
        text = f"患者{symptoms_text}{random.choice(['两天', '三天', '一周'])}"

        entities = []
        for symptom in template["symptoms"]:
            start = text.find(symptom)
            if start >= 0:
                entities.append({
                    "text": symptom,
                    "label": "SYMPTOM",
                    "standard_name": symptom,
                    "start": start,
                    "end": start + len(symptom),
                })

        ner_samples.append({
            "text": text,
            "entities": entities,
            "diagnosis": template["disease"],
            "source": "模拟数据",
        })

        # 口语化版本
        colloquial_symptoms = []
        for s in template["symptoms"][:3]:
            if s in COLLOQUIAL_VARIANTS:
                colloquial_symptoms.append(random.choice(COLLOQUIAL_VARIANTS[s]))
            else:
                colloquial_symptoms.append(s)

        colloquial_text = f"我{'，'.join(colloquial_symptoms)}，有{random.choice(['两三天', '几天', '一周多了'])}"
        colloquial_entities = []
        for i, (colloquial, standard) in enumerate(
            zip(colloquial_symptoms, template["symptoms"][:3])
        ):
            start = colloquial_text.find(colloquial)
            if start >= 0:
                colloquial_entities.append({
                    "text": colloquial,
                    "label": "SYMPTOM",
                    "standard_name": standard,
                    "start": start,
                    "end": start + len(colloquial),
                })

        ner_samples.append({
            "text": colloquial_text,
            "entities": colloquial_entities,
            "diagnosis": template["disease"],
            "source": "模拟数据（口语化）",
        })

    return ner_samples


def generate_dialogue_data() -> list[dict]:
    """生成问诊对话数据"""
    dialogue_samples = []

    for i, template in enumerate(DISEASE_TEMPLATES):
        dialogue = [
            {"role": role, "content": content}
            for role, content in template["dialogue"]
        ]

        dialogue_samples.append({
            "session_id": f"sim_{i+1:03d}",
            "chief_complaint": template["chief_complaint"],
            "dialogue": dialogue,
            "symptoms": template["symptoms"],
            "final_diagnosis": template["disease"],
            "source": "模拟数据",
            "note": "此数据为人工设计的模拟问诊对话，不对应任何真实患者",
        })

    return dialogue_samples


def main():
    """生成所有示例数据"""
    data_dir = Path(__file__).resolve().parent

    # 生成 NER 数据
    ner_data = generate_ner_data()
    ner_path = data_dir / "sample_ner.json"
    with open(ner_path, "w", encoding="utf-8") as f:
        json.dump(ner_data, f, ensure_ascii=False, indent=2)
    print(f"✅ NER 数据已生成: {ner_path} ({len(ner_data)} 条)")

    # 生成对话数据
    dialogue_data = generate_dialogue_data()
    dialogue_path = data_dir / "sample_dialogue.json"
    with open(dialogue_path, "w", encoding="utf-8") as f:
        json.dump(dialogue_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 对话数据已生成: {dialogue_path} ({len(dialogue_data)} 条)")

    # 打印预览
    print("\n--- NER 数据预览 ---")
    for sample in ner_data[:3]:
        print(f"  文本: {sample['text']}")
        print(f"  实体: {[(e['text'], e['standard_name']) for e in sample['entities']]}")
        print(f"  诊断: {sample['diagnosis']}")
        print()

    print("--- 对话数据预览 ---")
    for sample in dialogue_data[:2]:
        print(f"  主诉: {sample['chief_complaint']}")
        print(f"  对话轮数: {len(sample['dialogue'])}")
        print(f"  诊断: {sample['final_diagnosis']}")
        print()


if __name__ == "__main__":
    main()
