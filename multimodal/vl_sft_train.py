"""
multimodal/vl_sft_train.py
==========================
Qwen-VL 多模态 LoRA 微调代码。

功能：
  1. 支持医学影像图文数据集（图片+描述对）
  2. 使用 LoRA 微调 Qwen-VL 模型
  3. SwanLab 记录训练指标
  4. 单卡 RTX4090 24G 可运行

数据集格式（JSON）：
[
  {
    "image": "path/to/image.jpg",
    "conversations": [
      {"role": "user", "content": "<image>\n请分析这张CT影像"},
      {"role": "assistant", "content": "影像表现：..."}
    ]
  }
]
"""

from __future__ import annotations
import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from loguru import logger

from configs.settings import settings


# ============================================================
# 训练配置
# ============================================================

@dataclass
class VLTrainConfig:
    """Qwen-VL 微调训练配置"""
    # 模型路径
    model_path: str = settings.QWEN_VL_PATH
    output_dir: str = "./checkpoints/vl_lora"

    # 数据集
    train_data_path: str = "./data/vl_train.json"
    eval_data_path: str = "./data/vl_eval.json"

    # LoRA 参数
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # 训练参数
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 2048

    # 精度与优化
    bf16: bool = True
    gradient_checkpointing: bool = True

    # SwanLab
    use_swanlab: bool = True
    swanlab_project: str = settings.SWANLAB_PROJECT
    swanlab_experiment: str = "qwen_vl_medical_lora"

    # 日志
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200


# ============================================================
# 数据集构建
# ============================================================

def build_vl_dataset(data_path: str, processor, max_seq_length: int = 2048):
    """
    构建 Qwen-VL 多模态数据集。

    :param data_path: JSON 数据文件路径
    :param processor: Qwen-VL 处理器
    :param max_seq_length: 最大序列长度
    :return: PyTorch Dataset
    """
    from torch.utils.data import Dataset
    from PIL import Image

    class MedicalVLDataset(Dataset):
        """医学影像图文数据集"""

        def __init__(self, data_path, processor, max_seq_length):
            self.processor = processor
            self.max_seq_length = max_seq_length
            self.data = self._load_data(data_path)

        def _load_data(self, path):
            """加载 JSON 数据"""
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            image_path = item.get("image", "")
            conversations = item.get("conversations", [])

            # 加载图片
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                # 如果图片加载失败，创建空白图片
                image = Image.new("RGB", (224, 224), color="white")

            # 构造对话文本
            text = self.processor.apply_chat_template(
                conversations, tokenize=False, add_generation_prompt=False
            )

            # 处理输入
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )

            return {
                "input_ids": inputs["input_ids"].squeeze(),
                "attention_mask": inputs["attention_mask"].squeeze(),
                "pixel_values": inputs.get("pixel_values", None),
                "labels": inputs["input_ids"].squeeze().clone(),
            }

    return MedicalVLDataset(data_path, processor, max_seq_length)


# ============================================================
# 训练主函数
# ============================================================

def train_vl_lora(config: VLTrainConfig = None):
    """
    Qwen-VL LoRA 微调主函数。

    训练流程：
      1. 加载 Qwen-VL 基座模型和处理器
      2. 注入 LoRA 适配器
      3. 构建数据集
      4. 配置 Trainer（集成 SwanLab）
      5. 开始训练
    """
    if config is None:
        config = VLTrainConfig()

    logger.info(f"[VL训练] 开始 Qwen-VL LoRA 微调")
    logger.info(f"[VL训练] 模型路径: {config.model_path}")
    logger.info(f"[VL训练] 输出目录: {config.output_dir}")

    # ---------- 1. 加载模型和处理器 ----------
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    logger.info("[VL训练] 加载 Qwen-VL 模型...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        config.model_path, trust_remote_code=True
    )

    # ---------- 2. 注入 LoRA ----------
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---------- 3. 构建数据集 ----------
    logger.info("[VL训练] 构建数据集...")
    train_dataset = build_vl_dataset(
        config.train_data_path, processor, config.max_seq_length
    )
    eval_dataset = None
    if os.path.exists(config.eval_data_path):
        eval_dataset = build_vl_dataset(
            config.eval_data_path, processor, config.max_seq_length
        )

    # ---------- 4. 配置训练参数 ----------
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=config.eval_steps if eval_dataset else None,
        save_total_limit=3,
        report_to="swanlab" if config.use_swanlab else "none",
        run_name=config.swanlab_experiment,
    )

    # ---------- 5. 初始化 SwanLab ----------
    if config.use_swanlab:
        try:
            import swanlab
            swanlab.init(
                project=config.swanlab_project,
                experiment_name=config.swanlab_experiment,
                config={
                    "model": config.model_path,
                    "lora_rank": config.lora_rank,
                    "lora_alpha": config.lora_alpha,
                    "learning_rate": config.learning_rate,
                    "epochs": config.num_train_epochs,
                    "batch_size": config.per_device_train_batch_size,
                },
            )
            logger.info("[VL训练] SwanLab 初始化成功")
        except Exception as e:
            logger.warning(f"[VL训练] SwanLab 初始化失败: {e}")

    # ---------- 6. 开始训练 ----------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
    )

    logger.info("[VL训练] 开始训练...")
    trainer.train()

    # ---------- 7. 保存模型 ----------
    trainer.save_model(config.output_dir)
    processor.save_pretrained(config.output_dir)
    logger.info(f"[VL训练] 模型已保存到: {config.output_dir}")

    if config.use_swanlab:
        try:
            import swanlab
            swanlab.finish()
        except Exception:
            pass


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    """运行 Qwen-VL LoRA 微调"""
    import argparse

    parser = argparse.ArgumentParser(description="Qwen-VL LoRA 微调")
    parser.add_argument("--model_path", type=str, default=settings.QWEN_VL_PATH)
    parser.add_argument("--train_data", type=str, default="./data/vl_train.json")
    parser.add_argument("--eval_data", type=str, default="./data/vl_eval.json")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/vl_lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--swanlab", action="store_true", default=True)
    args = parser.parse_args()

    config = VLTrainConfig(
        model_path=args.model_path,
        train_data_path=args.train_data,
        eval_data_path=args.eval_data,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        lora_rank=args.lora_rank,
        learning_rate=args.lr,
        use_swanlab=args.swanlab,
    )

    train_vl_lora(config)
