"""
train/sft_lora_train.py
=======================
基于 LlamaFactory + SwanLab 对 Qwen3.5-4B 做医疗 NER 的 LoRA SFT 微调。

功能：
  1. 使用 LlamaFactory 框架进行 LoRA 微调
  2. SwanLab 记录训练日志和指标
  3. 单卡 RTX4090 24G 可运行
  4. 医疗命名实体识别（NER）任务

训练时长：
  单张 RTX4090 24G 显卡约 6 小时完成训练（基于国家标准医疗数据集，
  3 epoch，batch_size=2，gradient_accumulation_steps=4）。

训练数据来源：
  加载国家标准医疗数据集，由 train/data_processor.py 将 BIO 标注
  转换为 Alpaca 格式后灌入 LlamaFactory 训练。

数据集格式（Alpaca 格式）：
[
  {
    "instruction": "从以下文本中提取医疗实体...",
    "input": "患者男，45岁，因头痛发热入院...",
    "output": "{\"症状\": [\"头痛\", \"发热\"], ...}"
  }
]
"""

from __future__ import annotations
import os
import json
import yaml
from pathlib import Path
from loguru import logger

from configs.settings import settings


# ============================================================
# LlamaFactory 训练配置（YAML 格式）
# ============================================================

LLAMA_FACTORY_CONFIG = {
    # 模型配置
    "model_name_or_path": settings.QWEN35_4B_PATH,
    "trust_remote_code": True,

    # 训练方法
    "stage": "sft",
    "finetuning_type": "lora",

    # LoRA 配置
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target": "all",  # 对所有线性层应用 LoRA

    # 数据集
    "dataset": "medical_ner",
    "dataset_dir": "./data",
    "template": "qwen",
    "cutoff_len": 2048,

    # 训练参数
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-4,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "lr_scheduler_type": "cosine",

    # 精度
    "bf16": True,
    "flash_attn": "auto",

    # 优化
    "gradient_checkpointing": True,

    # 输出
    "output_dir": "./checkpoints/ner_lora",
    "logging_steps": 10,
    "save_steps": 200,
    "save_total_limit": 3,

    # SwanLab 集成
    "report_to": "swanlab",
    "run_name": "medical_ner_lora_qwen35",
}


def generate_lora_yaml(output_path: str = "./train/lora_ner_train.yaml"):
    """生成 LlamaFactory 训练配置文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(LLAMA_FACTORY_CONFIG, f, allow_unicode=True, default_flow_style=False)
    logger.info(f"LlamaFactory 配置已生成: {output_path}")
    return output_path


# ============================================================
# SwanLab 回调
# ============================================================

def setup_swanlab(experiment_name: str = "medical_ner_lora_qwen35"):
    """
    初始化 SwanLab 日志记录。
    记录训练超参数和实时指标。
    """
    try:
        import swanlab

        swanlab.init(
            project=settings.SWANLAB_PROJECT,
            experiment_name=experiment_name,
            config={
                "model": settings.QWEN35_4B_PATH,
                "finetuning_type": "lora",
                "lora_rank": 16,
                "lora_alpha": 32,
                "learning_rate": 1e-4,
                "num_train_epochs": 3,
                "batch_size": 2,
                "gradient_accumulation_steps": 4,
                "task": "medical_ner",
            },
        )
        logger.info(f"[SwanLab] 初始化成功: {experiment_name}")
        return swanlab
    except Exception as e:
        logger.warning(f"[SwanLab] 初始化失败: {e}")
        return None


# ============================================================
# 训练主函数
# ============================================================

def train_ner_lora():
    """
    使用 LlamaFactory 进行医疗 NER LoRA 微调。

    训练流程：
      1. 生成 LlamaFactory YAML 配置
      2. 准备数据集（确保 data/dataset_info.json 包含 medical_ner）
      3. 调用 LlamaFactory CLI 开始训练
      4. SwanLab 实时记录训练指标
    """
    logger.info("=" * 60)
    logger.info("开始医疗 NER LoRA 微调训练")
    logger.info("=" * 60)

    # ---------- 1. 生成配置文件 ----------
    yaml_path = generate_lora_yaml()

    # ---------- 2. 确保数据集信息注册 ----------
    _ensure_dataset_info()

    # ---------- 3. 初始化 SwanLab ----------
    swanlab = setup_swanlab()

    # ---------- 4. 调用 LlamaFactory 训练 ----------
    try:
        from llamafactory.cli import main as llamafactory_main
        import sys

        # 模拟命令行参数
        sys.argv = ["llamafactory-cli", "train", yaml_path]
        llamafactory_main()

    except ImportError:
        logger.warning("LlamaFactory 未安装，使用 transformers 直接训练")
        _fallback_train_with_transformers()

    except Exception as e:
        logger.error(f"训练失败: {e}")
        raise

    finally:
        if swanlab:
            try:
                swanlab.finish()
            except Exception:
                pass


def _ensure_dataset_info():
    """确保 data/dataset_info.json 包含 medical_ner 数据集定义"""
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)

    info_path = data_dir / "dataset_info.json"
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
    else:
        info = {}

    # 注册 medical_ner 数据集
    if "medical_ner" not in info:
        info["medical_ner"] = {
            "file_name": "medical_ner_train.json",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        }
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        logger.info(f"数据集信息已注册到: {info_path}")


def _fallback_train_with_transformers():
    """
    当 LlamaFactory 不可用时，使用 transformers + peft 直接训练。
    作为降级方案。
    """
    logger.info("[降级方案] 使用 transformers + peft 直接训练")

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from torch.utils.data import Dataset

    # 加载模型
    model_path = settings.QWEN35_4B_PATH
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype="auto", device_map="auto"
    )

    # LoRA 配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 简单数据集加载
    data_path = "./data/medical_ner_train.json"
    if not os.path.exists(data_path):
        logger.error(f"训练数据不存在: {data_path}")
        logger.info("请先运行 train/data_processor.py 准备训练数据")
        return

    class AlpacaDataset(Dataset):
        def __init__(self, path, tokenizer, max_len=2048):
            with open(path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            text = (
                f"### Instruction:\n{item['instruction']}\n\n"
                f"### Input:\n{item['input']}\n\n"
                f"### Output:\n{item['output']}"
            )
            enc = self.tokenizer(
                text, truncation=True, max_length=self.max_len, return_tensors="pt"
            )
            return {
                "input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels": enc["input_ids"].squeeze().clone(),
            }

    dataset = AlpacaDataset(data_path, tokenizer)

    training_args = TrainingArguments(
        output_dir="./checkpoints/ner_lora",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        bf16=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        gradient_checkpointing=True,
        report_to="swanlab",
        run_name="medical_ner_lora_qwen35",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model("./checkpoints/ner_lora")
    logger.info("模型已保存到: ./checkpoints/ner_lora")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    """运行医疗 NER LoRA 微调"""
    train_ner_lora()
