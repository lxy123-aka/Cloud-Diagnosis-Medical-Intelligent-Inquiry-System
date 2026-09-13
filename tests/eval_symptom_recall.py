"""
tests/eval_symptom_recall.py
=============================
症状标准化流水线召回率评估。

评估目标：
  验证三层流水线（LLM → Neo4j → Milvus）相比仅 LLM 提取，
  标准术语命中率的提升幅度。简历口径："召回率从 60% 提升至 95% 以上"。

评估方法：
  1. 以 data/sample_ner.json 口语化样本为基础，扩充口语化用例构建评估集
  2. 基线 = 仅 LLM 提取结果直接作为症状名（不经过图谱/向量/白名单）
  3. 完整 = 三层流水线（LLM → Neo4j 图谱 → Milvus 向量）
  4. 对比两者"标准术语命中率"
  5. 每条样本跑 2 次取并集，消除 LLM 随机性影响

运行方式：
  python tests/eval_symptom_recall.py

注意：
  - 所有数字均为实测值，脚本中不硬编码任何目标值
  - 需要 Qwen API 可用；Neo4j/Milvus 不可用时对应层跳过（降级模式）
"""

from __future__ import annotations
import sys
import os
import json
import asyncio
from pathlib import Path

# 确保项目根目录在路径中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger


# ============================================================
# 评估集定义
# ============================================================

# 从 sample_ner.json 加载基础评估样本
def load_base_eval_set() -> list[dict]:
    """加载 data/sample_ner.json 中的口语化样本"""
    ner_path = PROJECT_ROOT / "data" / "sample_ner.json"
    if not ner_path.exists():
        logger.warning(f"评估数据不存在: {ner_path}")
        return []

    with open(ner_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 只保留口语化样本（source 包含"口语化"）
    oral_samples = [s for s in data if "口语化" in s.get("source", "")]
    return oral_samples


# 额外扩充的口语化评估用例
# 格式：{"text": 口语描述, "expected_standard_names": [标准症状名列表]}
EXPANDED_EVAL_CASES = [
    {
        "text": "我肚肚痛，大便稀，想吐，有几天",
        "expected_standard_names": ["腹痛", "腹泻", "恶心"],
    },
    {
        "text": "我头胀，发烧，咳，有几天",
        "expected_standard_names": ["头痛", "发热", "咳嗽"],
    },
    {
        "text": "我胃酸，上腹痛，嗳气，有一周多了",
        "expected_standard_names": ["反酸", "上腹痛", "嗳气"],
    },
    {
        "text": "我打喷嚏，流涕，鼻子堵了，有几天",
        "expected_standard_names": ["打喷嚏", "流涕", "鼻塞"],
    },
    {
        "text": "我头昏，头疼，心悸，有几天",
        "expected_standard_names": ["头晕", "头痛", "心悸"],
    },
    {
        "text": "我多饮，多尿，多食，有两三天",
        "expected_standard_names": ["多饮", "多尿", "多食"],
    },
    # 额外扩充用例——更口语化、方言、描述性表达
    {
        "text": "脑壳疼得厉害，还有点发烧",
        "expected_standard_names": ["头痛", "发热"],
    },
    {
        "text": "肚子胀得慌，老放屁",
        "expected_standard_names": ["腹胀"],
    },
    {
        "text": "昨儿个吃坏东西了，上吐下泻",
        "expected_standard_names": ["呕吐", "腹泻"],
    },
    {
        "text": "心慌慌的，睡不着觉",
        "expected_standard_names": ["心悸", "失眠"],
    },
    {
        "text": "眼睛花，看东西模模糊糊的",
        "expected_standard_names": ["视力下降"],
    },
    {
        "text": "心口窝儿疼，吃凉的就加重",
        "expected_standard_names": ["上腹痛"],
    },
    {
        "text": "身上刺挠，起了一片红疙瘩",
        "expected_standard_names": ["皮肤瘙痒", "皮疹"],
    },
    {
        "text": "老打嗝停不下来",
        "expected_standard_names": ["嗳气"],
    },
    {
        "text": "腰杆子酸痛，使不上劲",
        "expected_standard_names": ["腰痛"],
    },
    {
        "text": "老做噩梦，睡不踏实",
        "expected_standard_names": ["失眠", "多梦"],
    },
    {
        "text": "眼睛干涩，看东西费劲",
        "expected_standard_names": ["眼干"],
    },
    {
        "text": "特别怕冷，手脚冰凉",
        "expected_standard_names": ["畏寒"],
    },
    {
        "text": "浑身不得劲儿，关节酸",
        "expected_standard_names": ["乏力", "关节痛"],
    },
]


def build_eval_set() -> list[dict]:
    """
    构建完整评估集。
    每条记录格式：
      {"text": str, "expected_standard_names": list[str]}
    """
    eval_set = []

    # 1. 从 sample_ner.json 加载口语化样本
    base_samples = load_base_eval_set()
    for s in base_samples:
        expected = list(set(
            e["standard_name"] for e in s.get("entities", [])
            if e.get("standard_name")
        ))
        if expected:
            eval_set.append({
                "text": s["text"],
                "expected_standard_names": expected,
                "source": "sample_ner.json",
            })

    # 2. 合入扩充用例（去重）
    existing_texts = {item["text"] for item in eval_set}
    for case in EXPANDED_EVAL_CASES:
        if case["text"] not in existing_texts:
            eval_set.append({
                "text": case["text"],
                "expected_standard_names": case["expected_standard_names"],
                "source": "expanded",
            })

    return eval_set


# ============================================================
# 基线评估：仅 LLM 提取
# ============================================================

async def evaluate_baseline(eval_set: list[dict]) -> dict:
    """
    基线：仅 LLM 提取结果直接作为症状名（不经过图谱/向量/白名单）。

    Returns:
        {
            "total_expected": int,     # 评估集中标准症状总数
            "total_hit": int,          # 命中数
            "hit_rate": float,         # 命中率
            "details": list[dict],     # 每条样本的详细结果
        }
    """
    from pipeline.symptom_normalize import SymptomNormalizePipeline

    pipeline = SymptomNormalizePipeline()

    total_expected = 0
    total_hit = 0
    details = []

    RUNS = 2  # 每条样本跑 2 次取并集，消除 LLM 随机性

    for item in eval_set:
        text = item["text"]
        expected_names = set(item["expected_standard_names"])
        total_expected += len(expected_names)

        # 仅 LLM 提取（第一层），跑 2 次取并集
        extracted_names = set()
        for _ in range(RUNS):
            try:
                llm_symptoms = await pipeline._layer1_llm_extract(text)
                extracted_names.update(
                    s.get("standard_name", "") for s in llm_symptoms
                    if s.get("standard_name")
                )
            except Exception as e:
                logger.warning(f"基线 LLM 提取失败: {text} → {e}")

        # 计算命中
        hits = expected_names & extracted_names
        total_hit += len(hits)

        details.append({
            "text": text,
            "expected": sorted(expected_names),
            "extracted": sorted(extracted_names),
            "hits": sorted(hits),
            "missed": sorted(expected_names - extracted_names),
        })

    hit_rate = total_hit / max(total_expected, 1)

    return {
        "total_expected": total_expected,
        "total_hit": total_hit,
        "hit_rate": hit_rate,
        "details": details,
    }


# ============================================================
# 完整评估：三层流水线
# ============================================================

async def evaluate_full_pipeline(eval_set: list[dict]) -> dict:
    """
    完整评估：三层流水线（LLM → 白名单 → Neo4j → Milvus）。

    Returns:
        同基线评估格式
    """
    from pipeline.symptom_normalize import SymptomNormalizePipeline

    pipeline = SymptomNormalizePipeline()

    total_expected = 0
    total_hit = 0
    details = []

    RUNS = 2  # 每条样本跑 2 次取并集，消除 LLM 随机性

    for item in eval_set:
        text = item["text"]
        expected_names = set(item["expected_standard_names"])
        total_expected += len(expected_names)

        # 完整三层流水线，跑 2 次取并集
        extracted_names = set()
        for _ in range(RUNS):
            try:
                result = await pipeline.run(text)
                extracted_names.update(
                    s.get("standard_name", "") for s in result
                    if s.get("standard_name")
                )
            except Exception as e:
                logger.warning(f"完整流水线处理失败: {text} → {e}")

        # 计算命中
        hits = expected_names & extracted_names
        total_hit += len(hits)

        details.append({
            "text": text,
            "expected": sorted(expected_names),
            "extracted": sorted(extracted_names),
            "hits": sorted(hits),
            "missed": sorted(expected_names - extracted_names),
        })

    hit_rate = total_hit / max(total_expected, 1)

    return {
        "total_expected": total_expected,
        "total_hit": total_hit,
        "hit_rate": hit_rate,
        "details": details,
    }


# ============================================================
# 报告输出
# ============================================================

def print_report(baseline: dict, full_pipeline: dict, eval_set_size: int):
    """打印评估报告"""
    print("\n" + "=" * 60)
    print("  症状标准化流水线 —— 标准术语命中率评估报告")
    print("=" * 60)

    print(f"\n评估集规模: {eval_set_size} 条样本")
    print(f"标准症状总数: {baseline['total_expected']} 个")

    print(f"\n【基线：仅 LLM 提取】")
    print(f"  命中: {baseline['total_hit']}/{baseline['total_expected']}")
    print(f"  命中率: {baseline['hit_rate']:.1%}")

    print(f"\n【完整：三层流水线（LLM → Neo4j → Milvus）】")
    print(f"  命中: {full_pipeline['total_hit']}/{full_pipeline['total_expected']}")
    print(f"  命中率: {full_pipeline['hit_rate']:.1%}")

    improvement = full_pipeline['hit_rate'] - baseline['hit_rate']
    print(f"\n【提升幅度】")
    print(f"  命中率提升: {improvement:+.1%}")

    # 详细对比
    print(f"\n【详细对比】")
    for i, (b, f) in enumerate(zip(baseline['details'], full_pipeline['details'])):
        print(f"\n  [{i+1}] 输入: {b['text']}")
        print(f"      期望: {b['expected']}")
        print(f"      基线提取: {b['extracted']}  命中: {b['hits']}  遗漏: {b['missed']}")
        print(f"      流水线提取: {f['extracted']}  命中: {f['hits']}  遗漏: {f['missed']}")

    # 口径说明
    print(f"\n【口径说明】")
    print(f"  - 基线 = 仅 LLM 语义提取结果直接作为标准症状名（不经过图谱/向量/白名单）")
    print(f"  - 完整 = 三层流水线（LLM 提取 → Neo4j 图谱精确匹配 → Milvus 向量语义召回）")
    print(f"  - 命中率 = 提取的标准症状名与期望标准名的交集 / 期望标准名总数")
    print(f"  - 每条样本跑 2 次取并集，消除 LLM 随机性")
    print(f"  - 以上所有数字均为实测值，未硬编码任何目标值")
    print("=" * 60)


# ============================================================
# 主函数
# ============================================================

async def main():
    """运行症状召回率评估"""
    print("症状标准化流水线 —— 标准术语命中率评估")

    # 构建评估集
    eval_set = build_eval_set()
    if not eval_set:
        print("❌ 评估集为空，请检查 data/sample_ner.json 是否存在")
        return

    print(f"评估集: {len(eval_set)} 条样本")

    # 基线评估
    print("\n正在评估基线（仅 LLM 提取）...")
    baseline = await evaluate_baseline(eval_set)

    # 完整流水线评估
    print("正在评估完整三层流水线...")
    full_pipeline = await evaluate_full_pipeline(eval_set)

    # 输出报告
    print_report(baseline, full_pipeline, len(eval_set))


if __name__ == "__main__":
    asyncio.run(main())
