"""
tests/eval_ner.py
=================
医疗 NER 实体识别评测。

评估目标：
  验证医疗实体识别命中率。简历口径："微调后医疗实体识别命中 90-95% 以上"。

评估方法：
  1. 以 data/sample_ner.json 全部样本为评估集（含标准样本和口语化样本）
  2. 优先使用本地蒸馏模型（checkpoints/distilled）或 LoRA 模型（checkpoints/ner_lora）
  3. 若本地无模型，则评估 LLM API 抽取的实体命中率
  4. 输出 Precision / Recall / F1

运行方式：
  python tests/eval_ner.py

注意：
  - 所有数字均为实测值，脚本中不硬编码任何目标值
  - 需要 Qwen API 可用（当本地模型不存在时）
"""

from __future__ import annotations
import sys
import os
import json
import asyncio
from pathlib import Path
from collections import Counter

# 确保项目根目录在路径中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger


# ============================================================
# 评估集加载
# ============================================================

def load_eval_set() -> list[dict]:
    """
    加载 data/sample_ner.json 全部样本作为评估集。

    每条记录格式：
      {
        "text": str,
        "entities": [{"text": str, "label": str, "standard_name": str, ...}],
        "diagnosis": str
      }
    """
    ner_path = PROJECT_ROOT / "data" / "sample_ner.json"
    if not ner_path.exists():
        logger.error(f"评估数据不存在: {ner_path}")
        return []

    with open(ner_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"加载评估集: {len(data)} 条样本")
    return data


# ============================================================
# 本地模型推理
# ============================================================

def try_local_model_infer(text: str) -> list[dict] | None:
    """
    尝试使用本地模型进行 NER 推理。
    优先蒸馏模型，其次 LoRA 模型。

    Returns:
        实体列表 [{"text": str, "label": str, "standard_name": str}] 或 None（模型不存在）
    """
    import torch

    # 检查蒸馏模型
    distilled_path = PROJECT_ROOT / "checkpoints" / "distilled"
    lora_path = PROJECT_ROOT / "checkpoints" / "ner_lora"

    model_path = None
    if distilled_path.exists():
        model_path = str(distilled_path)
    elif lora_path.exists():
        model_path = str(lora_path)

    if model_path is None:
        return None

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()

        prompt = (
            "### Instruction:\n"
            "你是一个医疗命名实体识别（NER）专家。请从以下医疗文本中提取所有医学实体，"
            "包括症状、疾病、药物、身体部位、检查项目、检验指标、治疗方案、科室等。"
            "以JSON格式输出，key为实体类型，value为实体列表。\n\n"
            f"### Input:\n{text}\n\n"
            "### Output:\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200)

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # 解析 JSON 输出
        entities = _parse_ner_output(response)
        return entities

    except Exception as e:
        logger.warning(f"本地模型推理失败: {e}")
        return None


def _parse_ner_output(text: str) -> list[dict]:
    """解析模型输出的 NER JSON"""
    import re
    entities = []

    # 尝试提取 JSON
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if isinstance(data, dict):
                for entity_type, values in data.items():
                    if isinstance(values, list):
                        for v in values:
                            entities.append({
                                "text": v,
                                "label": "SYMPTOM" if "症状" in entity_type else "ENTITY",
                                "standard_name": v,
                            })
        except json.JSONDecodeError:
            pass

    return entities


# ============================================================
# LLM API 推理
# ============================================================

async def llm_api_infer(text: str) -> list[dict]:
    """
    使用 LLM API 进行 NER 实体抽取。

    Returns:
        实体列表 [{"text": str, "label": str, "standard_name": str}]
    """
    from langchain_openai import ChatOpenAI
    from configs.settings import settings

    llm = ChatOpenAI(
        model=settings.QWEN_MODEL_NAME,
        api_key=settings.QWEN_API_KEY,
        base_url=settings.QWEN_BASE_URL,
        temperature=0.1,
    )

    tools = [{
        "type": "function",
        "function": {
            "name": "extract_medical_entities",
            "description": "从医疗文本中提取医学实体",
            "parameters": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string", "description": "实体原文"},
                                "label": {"type": "string", "description": "实体类型：SYMPTOM/DISEASE/DRUG/EXAMINATION/INDICATOR/BODY_PART"},
                                "standard_name": {"type": "string", "description": "标准医学术语名称"},
                            },
                            "required": ["text", "label", "standard_name"],
                        },
                    },
                },
                "required": ["entities"],
            },
        },
    }]

    llm_with_tools = llm.bind_tools(
        tools, tool_choice={"type": "function", "function": {"name": "extract_medical_entities"}}
    )

    response = await llm_with_tools.ainvoke([
        {
            "role": "system",
            "content": (
                "你是一个医疗命名实体识别（NER）专家。从文本中提取所有医学实体。"
                "实体类型包括：症状(SYMPTOM)、疾病(DISEASE)、药物(DRUG)、"
                "检查项目(EXAMINATION)、检验指标(INDICATOR)、身体部位(BODY_PART)。"
                "将口语化表达转换为标准医学术语。"
            ),
        },
        {"role": "user", "content": text},
    ])

    try:
        tool_call = response.tool_calls[0]
        args = tool_call["args"] if isinstance(tool_call["args"], dict) else json.loads(tool_call["args"])
        return args.get("entities", [])
    except (IndexError, KeyError, json.JSONDecodeError, TypeError) as e:
        logger.warning(f"LLM API NER 解析失败: {e}")
        return []


# ============================================================
# 评测指标计算
# ============================================================

def compute_metrics(
    all_predicted: list[list[dict]],
    all_ground_truth: list[list[dict]],
) -> dict:
    """
    计算 NER 评测指标。

    匹配策略：
      - 以 standard_name 为匹配键（标准术语匹配）
      - 预测的 standard_name 出现在 ground truth 的 standard_name 集合中即为命中

    Returns:
        {"precision": float, "recall": float, "f1": float,
         "total_predicted": int, "total_gt": int, "total_hit": int}
    """
    total_predicted = 0
    total_gt = 0
    total_hit = 0

    for predicted, gt in zip(all_predicted, all_ground_truth):
        pred_names = set(
            e.get("standard_name", "") or e.get("text", "")
            for e in predicted
            if e.get("standard_name") or e.get("text")
        )
        gt_names = set(
            e.get("standard_name", "") or e.get("text", "")
            for e in gt
            if e.get("standard_name") or e.get("text")
        )

        total_predicted += len(pred_names)
        total_gt += len(gt_names)
        total_hit += len(pred_names & gt_names)

    precision = total_hit / max(total_predicted, 1)
    recall = total_hit / max(total_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_predicted": total_predicted,
        "total_gt": total_gt,
        "total_hit": total_hit,
    }


# ============================================================
# 评估主流程
# ============================================================

async def evaluate_ner(eval_set: list[dict]) -> dict:
    """
    执行 NER 评估。

    Returns:
        {"metrics": dict, "details": list[dict], "mode": str}
    """
    # 判断推理模式
    mode = "llm_api"
    distilled_path = PROJECT_ROOT / "checkpoints" / "distilled"
    lora_path = PROJECT_ROOT / "checkpoints" / "ner_lora"

    if distilled_path.exists():
        mode = "local_distilled"
    elif lora_path.exists():
        mode = "local_lora"

    print(f"\n推理模式: {mode}")
    if mode == "llm_api":
        print("  （本地无蒸馏/LoRA 模型，使用 LLM API 抽取实体）")

    all_predicted = []
    all_ground_truth = []
    details = []

    for i, sample in enumerate(eval_set):
        text = sample["text"]
        gt_entities = sample.get("entities", [])
        all_ground_truth.append(gt_entities)

        # 推理
        if mode.startswith("local"):
            predicted = try_local_model_infer(text)
            if predicted is None:
                # 降级到 API
                predicted = await llm_api_infer(text)
        else:
            predicted = await llm_api_infer(text)

        all_predicted.append(predicted)

        # 详细记录
        pred_names = set(
            e.get("standard_name", "") or e.get("text", "")
            for e in predicted
        )
        gt_names = set(
            e.get("standard_name", "") or e.get("text", "")
            for e in gt_entities
        )
        hits = pred_names & gt_names

        details.append({
            "text": text,
            "expected": sorted(gt_names),
            "predicted": sorted(pred_names),
            "hits": sorted(hits),
            "missed": sorted(gt_names - pred_names),
        })

        if (i + 1) % 5 == 0:
            print(f"  已处理 {i+1}/{len(eval_set)} 条...")

    metrics = compute_metrics(all_predicted, all_ground_truth)

    return {
        "metrics": metrics,
        "details": details,
        "mode": mode,
    }


# ============================================================
# 报告输出
# ============================================================

def print_report(result: dict, eval_set_size: int):
    """打印 NER 评测报告"""
    metrics = result["metrics"]
    mode = result["mode"]

    print("\n" + "=" * 60)
    print("  医疗 NER 实体识别 —— 评测报告")
    print("=" * 60)

    print(f"\n评估集规模: {eval_set_size} 条样本")
    print(f"推理模式: {mode}")

    print(f"\n【评测指标】")
    print(f"  Precision: {metrics['precision']:.1%}")
    print(f"  Recall:    {metrics['recall']:.1%}")
    print(f"  F1:        {metrics['f1']:.1%}")
    print(f"  预测实体总数: {metrics['total_predicted']}")
    print(f"  真实实体总数: {metrics['total_gt']}")
    print(f"  命中实体数: {metrics['total_hit']}")

    # 详细结果
    print(f"\n【详细结果】")
    for i, d in enumerate(result["details"]):
        print(f"\n  [{i+1}] 输入: {d['text']}")
        print(f"      期望: {d['expected']}")
        print(f"      预测: {d['predicted']}")
        print(f"      命中: {d['hits']}  遗漏: {d['missed']}")

    # 口径说明
    print(f"\n【口径说明】")
    print(f"  - 匹配键 = standard_name（标准医学术语名称）")
    print(f"  - Precision = 命中数 / 预测实体总数")
    print(f"  - Recall = 命中数 / 真实实体总数")
    print(f"  - F1 = 2 × Precision × Recall / (Precision + Recall)")
    print(f"  - 以上所有数字均为实测值，未硬编码任何目标值")
    print("=" * 60)


# ============================================================
# 主函数
# ============================================================

async def main():
    """运行 NER 评测"""
    print("医疗 NER 实体识别 —— 评测")

    eval_set = load_eval_set()
    if not eval_set:
        print("❌ 评估集为空，请检查 data/sample_ner.json 是否存在")
        return

    print(f"评估集: {len(eval_set)} 条样本")

    # 执行评估
    print("\n正在评估...")
    result = await evaluate_ner(eval_set)

    # 输出报告
    print_report(result, len(eval_set))


if __name__ == "__main__":
    asyncio.run(main())
