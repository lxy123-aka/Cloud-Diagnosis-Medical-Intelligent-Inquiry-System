"""
train/data_processor.py
=======================
医疗数据集预处理脚本。

功能：
  1. 处理医疗 NER 标注数据（BIO/实体标注格式）
  2. 转换成 LlamaFactory 训练格式（Alpaca 格式）
  3. 清洗脏数据（空值、格式错误、超长文本）
  4. 划分训练集 / 验证集
  5. 生成数据统计报告
"""

from __future__ import annotations
import os
import json
import re
from pathlib import Path
from collections import Counter
from loguru import logger


# ============================================================
# 数据清洗
# ============================================================

class DataCleaner:
    """医疗数据清洗器"""

    @staticmethod
    def clean_text(text: str) -> str:
        """清洗文本：去除多余空白、特殊字符"""
        if not text:
            return ""
        # 去除多余空白
        text = re.sub(r'\s+', ' ', text).strip()
        # 去除不可见字符
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        return text

    @staticmethod
    def is_valid_ner_sample(sample: dict) -> bool:
        """验证 NER 样本是否有效"""
        # 必须包含 input 和 output
        if not sample.get("input") or not sample.get("output"):
            return False
        # input 不能太短
        if len(sample["input"]) < 5:
            return False
        # output 必须是合法 JSON 或至少包含实体
        try:
            entities = json.loads(sample["output"])
            if not isinstance(entities, dict):
                return False
        except json.JSONDecodeError:
            # 允许非 JSON 格式，但必须有内容
            if len(sample["output"]) < 3:
                return False
        return True

    @staticmethod
    def truncate_text(text: str, max_length: int = 1500) -> str:
        """截断过长文本"""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."


# ============================================================
# NER 数据格式转换
# ============================================================

class NERDataConverter:
    """NER 数据格式转换器"""

    # NER 实体类型映射
    ENTITY_TYPES = {
        "symptom": "症状",
        "disease": "疾病",
        "drug": "药物",
        "body_part": "身体部位",
        "examination": "检查项目",
        "indicator": "检验指标",
        "treatment": "治疗方案",
        "department": "科室",
    }

    @staticmethod
    def bio_to_alpaca(bio_samples: list[dict]) -> list[dict]:
        """
        将 BIO 标注格式转换为 Alpaca 训练格式。

        BIO 格式输入：
        {"text": "患者头痛发热三天", "entities": [{"start": 2, "end": 4, "type": "symptom", "value": "头痛"}, ...]}

        Alpaca 格式输出：
        {"instruction": "...", "input": "患者头痛发热三天", "output": '{"症状": ["头痛"], ...}'}
        """
        alpaca_data = []

        for sample in bio_samples:
            text = sample.get("text", "")
            entities = sample.get("entities", [])

            if not text or not entities:
                continue

            # 按实体类型分组
            grouped = {}
            for ent in entities:
                ent_type = ent.get("type", "unknown")
                ent_value = ent.get("value", "")
                cn_type = NERDataConverter.ENTITY_TYPES.get(ent_type, ent_type)
                if cn_type not in grouped:
                    grouped[cn_type] = []
                if ent_value and ent_value not in grouped[cn_type]:
                    grouped[cn_type].append(ent_value)

            # 构造 Alpaca 格式
            alpaca_sample = {
                "instruction": (
                    "你是一个医疗命名实体识别（NER）专家。请从以下医疗文本中提取所有医学实体，"
                    "包括症状、疾病、药物、身体部位、检查项目、检验指标、治疗方案、科室等。"
                    "以JSON格式输出，key为实体类型，value为实体列表。"
                ),
                "input": DataCleaner.clean_text(text),
                "output": json.dumps(grouped, ensure_ascii=False),
            }

            if DataCleaner.is_valid_ner_sample(alpaca_sample):
                alpaca_data.append(alpaca_sample)

        return alpaca_data

    @staticmethod
    def conversation_to_alpaca(conversations: list[dict]) -> list[dict]:
        """
        将对话格式转换为 Alpaca 格式。

        对话格式输入：
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        """
        alpaca_data = []

        for conv in conversations:
            if len(conv) < 2:
                continue

            user_msg = None
            assistant_msg = None
            for msg in conv:
                if msg.get("role") == "user":
                    user_msg = msg.get("content", "")
                elif msg.get("role") == "assistant":
                    assistant_msg = msg.get("content", "")

            if user_msg and assistant_msg:
                alpaca_data.append({
                    "instruction": "作为医学AI助手，请专业准确地回答以下医学问题。",
                    "input": DataCleaner.clean_text(user_msg),
                    "output": DataCleaner.clean_text(assistant_msg),
                })

        return alpaca_data


# ============================================================
# 数据集划分与统计
# ============================================================

def split_dataset(data: list[dict], eval_ratio: float = 0.1) -> tuple[list, list]:
    """
    划分训练集和验证集。

    :param data: 全量数据
    :param eval_ratio: 验证集比例
    :return: (训练集, 验证集)
    """
    import random
    random.shuffle(data)
    split_idx = int(len(data) * (1 - eval_ratio))
    return data[:split_idx], data[split_idx:]


def generate_statistics(data: list[dict]) -> dict:
    """生成数据统计报告"""
    stats = {
        "total_samples": len(data),
        "avg_input_length": 0,
        "avg_output_length": 0,
        "max_input_length": 0,
        "max_output_length": 0,
        "entity_type_counts": Counter(),
    }

    input_lengths = []
    output_lengths = []

    for sample in data:
        inp_len = len(sample.get("input", ""))
        out_len = len(sample.get("output", ""))
        input_lengths.append(inp_len)
        output_lengths.append(out_len)

        # 统计实体类型
        try:
            entities = json.loads(sample.get("output", "{}"))
            if isinstance(entities, dict):
                for k, v in entities.items():
                    if isinstance(v, list):
                        stats["entity_type_counts"][k] += len(v)
        except json.JSONDecodeError:
            pass

    if input_lengths:
        stats["avg_input_length"] = sum(input_lengths) / len(input_lengths)
        stats["max_input_length"] = max(input_lengths)
    if output_lengths:
        stats["avg_output_length"] = sum(output_lengths) / len(output_lengths)
        stats["max_output_length"] = max(output_lengths)

    stats["entity_type_counts"] = dict(stats["entity_type_counts"])
    return stats


# ============================================================
# 生成示例数据（用于测试流程）
# ============================================================

def generate_sample_data(output_dir: str = "./data"):
    """生成示例 NER 训练数据（用于测试流程）"""
    os.makedirs(output_dir, exist_ok=True)

    # 示例医疗 NER 数据
    samples = [
        {
            "text": "患者男，45岁，因反复头痛伴发热3天入院。查体：体温38.5℃，咽部充血，双侧扁桃体II度肿大。血常规示白细胞升高。初步诊断为上呼吸道感染。",
            "entities": [
                {"start": 10, "end": 12, "type": "symptom", "value": "头痛"},
                {"start": 13, "end": 15, "type": "symptom", "value": "发热"},
                {"start": 30, "end": 34, "type": "indicator", "value": "体温38.5℃"},
                {"start": 36, "end": 40, "type": "symptom", "value": "咽部充血"},
                {"start": 53, "end": 58, "type": "examination", "value": "血常规"},
                {"start": 59, "end": 64, "type": "indicator", "value": "白细胞升高"},
                {"start": 69, "end": 76, "type": "disease", "value": "上呼吸道感染"},
            ],
        },
        {
            "text": "患者女，32岁，咳嗽一周，咳黄色粘痰，伴胸闷气促。既往有支气管哮喘病史5年。给予阿莫西林0.5g tid口服，布洛芬0.4g prn退热。",
            "entities": [
                {"start": 8, "end": 10, "type": "symptom", "value": "咳嗽"},
                {"start": 14, "end": 20, "type": "symptom", "value": "咳黄色粘痰"},
                {"start": 22, "end": 24, "type": "symptom", "value": "胸闷"},
                {"start": 24, "end": 26, "type": "symptom", "value": "气促"},
                {"start": 30, "end": 38, "type": "disease", "value": "支气管哮喘"},
                {"start": 44, "end": 48, "type": "drug", "value": "阿莫西林"},
                {"start": 55, "end": 59, "type": "drug", "value": "布洛芬"},
            ],
        },
        {
            "text": "患者老年男性，68岁，高血压病史10年，糖尿病史5年。近日出现头晕、视物模糊、下肢水肿。查血压160/100mmHg，空腹血糖8.2mmol/L。调整降压方案为氨氯地平5mg qd，加用二甲双胍0.5g bid。",
            "entities": [
                {"start": 12, "end": 15, "type": "disease", "value": "高血压"},
                {"start": 19, "end": 22, "type": "disease", "value": "糖尿病"},
                {"start": 28, "end": 30, "type": "symptom", "value": "头晕"},
                {"start": 31, "end": 35, "type": "symptom", "value": "视物模糊"},
                {"start": 36, "end": 40, "type": "symptom", "value": "下肢水肿"},
                {"start": 55, "end": 63, "type": "indicator", "value": "空腹血糖8.2mmol/L"},
                {"start": 70, "end": 74, "type": "drug", "value": "氨氯地平"},
                {"start": 84, "end": 88, "type": "drug", "value": "二甲双胍"},
            ],
        },
    ]

    # 转换为 Alpaca 格式
    converter = NERDataConverter()
    alpaca_data = converter.bio_to_alpaca(samples)

    # 扩充数据（重复 + 轻微变化，仅用于演示）
    expanded_data = alpaca_data * 10  # 扩充到30条

    # 划分训练集/验证集
    train_data, eval_data = split_dataset(expanded_data, eval_ratio=0.1)

    # 保存
    train_path = os.path.join(output_dir, "medical_ner_train.json")
    eval_path = os.path.join(output_dir, "medical_ner_eval.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)

    # 统计
    stats = generate_statistics(expanded_data)
    logger.info(f"示例数据已生成:")
    logger.info(f"  训练集: {len(train_data)} 条 → {train_path}")
    logger.info(f"  验证集: {len(eval_data)} 条 → {eval_path}")
    logger.info(f"  统计: {json.dumps(stats, ensure_ascii=False, indent=2)}")


# ============================================================
# 主流程
# ============================================================

def process_medical_data(
    input_path: str,
    output_dir: str = "./data",
    eval_ratio: float = 0.1,
    data_format: str = "bio",
):
    """
    处理医疗数据集主流程。

    :param input_path: 输入数据文件路径
    :param output_dir: 输出目录
    :param eval_ratio: 验证集比例
    :param data_format: 输入数据格式（bio / conversation）
    """
    logger.info(f"[数据处理] 输入文件: {input_path}")
    logger.info(f"[数据处理] 数据格式: {data_format}")

    # 加载原始数据
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 格式转换
    converter = NERDataConverter()
    cleaner = DataCleaner()

    if data_format == "bio":
        alpaca_data = converter.bio_to_alpaca(raw_data)
    elif data_format == "conversation":
        alpaca_data = converter.conversation_to_alpaca(raw_data)
    else:
        raise ValueError(f"不支持的数据格式: {data_format}")

    # 数据清洗
    cleaned_data = [s for s in alpaca_data if cleaner.is_valid_ner_sample(s)]
    logger.info(f"[数据处理] 清洗前: {len(alpaca_data)} 条, 清洗后: {len(cleaned_data)} 条")

    # 划分数据集
    train_data, eval_data = split_dataset(cleaned_data, eval_ratio)

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "medical_ner_train.json")
    eval_path = os.path.join(output_dir, "medical_ner_eval.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)

    # 统计报告
    stats = generate_statistics(cleaned_data)
    logger.info(f"[数据处理] 处理完成:")
    logger.info(f"  训练集: {len(train_data)} 条")
    logger.info(f"  验证集: {len(eval_data)} 条")
    logger.info(f"  统计: {json.dumps(stats, ensure_ascii=False, indent=2)}")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    """运行数据预处理"""
    import argparse

    parser = argparse.ArgumentParser(description="医疗数据集预处理")
    parser.add_argument("--input", type=str, default="", help="输入数据文件路径")
    parser.add_argument("--output_dir", type=str, default="./data", help="输出目录")
    parser.add_argument("--format", type=str, default="bio", choices=["bio", "conversation"])
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--generate_sample", action="store_true", help="生成示例数据")
    args = parser.parse_args()

    if args.generate_sample:
        generate_sample_data(args.output_dir)
    elif args.input:
        process_medical_data(args.input, args.output_dir, args.eval_ratio, args.format)
    else:
        print("请指定 --input 输入文件或使用 --generate_sample 生成示例数据")
