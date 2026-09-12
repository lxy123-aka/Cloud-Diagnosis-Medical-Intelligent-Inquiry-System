"""
train/model_distill.py
======================
模型蒸馏脚本 —— 将微调后的大模型知识蒸馏至小模型。

功能：
  1. 教师模型：LoRA 微调后的 Qwen3.5-4B
  2. 学生模型：Qwen3.5-1.5B 或更小的模型
  3. 知识蒸馏：使用教师模型的输出分布指导学生模型训练
  4. 目标：管线推理耗时 ≤ 300ms
  5. SwanLab 记录蒸馏训练指标
"""

from __future__ import annotations
import os
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from loguru import logger

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from configs.settings import settings


# ============================================================
# 蒸馏配置
# ============================================================

@dataclass
class DistillConfig:
    """模型蒸馏配置"""
    # 教师模型
    teacher_model_path: str = settings.NER_LORA_PATH  # LoRA 微调后的模型
    teacher_base_path: str = settings.QWEN35_4B_PATH  # 教师基座模型

    # 学生模型
    student_model_path: str = "./models_cache/Qwen3.5-1.5B"
    student_output_dir: str = "./checkpoints/distilled"

    # 蒸馏参数
    temperature: float = 2.0        # 蒸馏温度（越高分布越平滑）
    alpha: float = 0.7              # KD loss 权重（1-alpha 为硬标签权重）

    # 训练参数
    num_epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 5e-5
    max_seq_length: int = 1024
    gradient_accumulation_steps: int = 4

    # 数据
    train_data_path: str = "./data/medical_ner_train.json"

    # SwanLab
    use_swanlab: bool = True
    swanlab_project: str = settings.SWANLAB_PROJECT

    # 性能目标
    target_inference_ms: int = 300  # 目标推理耗时


# ============================================================
# 蒸馏数据集
# ============================================================

class DistillDataset(Dataset):
    """蒸馏训练数据集"""

    def __init__(self, data_path: str, tokenizer, max_seq_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 构造完整文本
        text = (
            f"### Instruction:\n{item['instruction']}\n\n"
            f"### Input:\n{item['input']}\n\n"
            f"### Output:\n{item['output']}"
        )
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": enc["input_ids"].squeeze().clone(),
        }


# ============================================================
# 知识蒸馏训练器
# ============================================================

class KnowledgeDistiller:
    """知识蒸馏训练器"""

    def __init__(self, config: DistillConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher_model = None
        self.student_model = None
        self.tokenizer = None

    def load_models(self):
        """加载教师模型和学生模型"""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("[蒸馏] 加载教师模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.teacher_base_path, trust_remote_code=True
        )

        # 教师模型（冻结参数）
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            self.config.teacher_base_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        # 加载 LoRA 权重（如果有）
        if os.path.exists(self.config.teacher_model_path):
            try:
                from peft import PeftModel
                self.teacher_model = PeftModel.from_pretrained(
                    self.teacher_model, self.config.teacher_model_path
                )
                logger.info("[蒸馏] 教师模型 LoRA 权重加载成功")
            except Exception as e:
                logger.warning(f"[蒸馏] LoRA 加载失败，使用基座模型: {e}")

        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # 学生模型
        logger.info("[蒸馏] 加载学生模型...")
        self.student_model = AutoModelForCausalLM.from_pretrained(
            self.config.student_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.student_model.train()

        # 打印参数量
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters()) / 1e6
        student_params = sum(p.numel() for p in self.student_model.parameters()) / 1e6
        logger.info(f"[蒸馏] 教师模型: {teacher_params:.1f}M 参数")
        logger.info(f"[蒸馏] 学生模型: {student_params:.1f}M 参数")
        logger.info(f"[蒸馏] 压缩比: {teacher_params / student_params:.1f}x")

    def compute_distill_loss(self, teacher_logits, student_logits, labels, temperature):
        """
        计算知识蒸馏损失（KL 散度）。

        :param teacher_logits: 教师模型输出 logits
        :param student_logits: 学生模型输出 logits
        :param labels: 真实标签
        :param temperature: 蒸馏温度
        :return: 蒸馏损失
        """
        # 只对非 padding 位置计算损失
        shift_labels = labels[..., 1:].contiguous()
        shift_teacher = teacher_logits[..., :-1, :].contiguous()
        shift_student = student_logits[..., :-1, :].contiguous()

        # 温度缩放
        teacher_probs = F.softmax(shift_teacher / temperature, dim=-1)
        student_log_probs = F.log_softmax(shift_student / temperature, dim=-1)

        # KL 散度
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        kl_loss *= temperature ** 2

        return kl_loss

    def train(self):
        """执行蒸馏训练"""
        self.load_models()

        # 构建数据集
        dataset = DistillDataset(
            self.config.train_data_path,
            self.tokenizer,
            self.config.max_seq_length,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
        )

        # 优化器
        optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=self.config.learning_rate,
        )

        # SwanLab
        swanlab = None
        if self.config.use_swanlab:
            try:
                import swanlab
                swanlab = swanlab
                swanlab.init(
                    project=self.config.swanlab_project,
                    experiment_name="model_distill",
                    config={
                        "teacher": self.config.teacher_base_path,
                        "student": self.config.student_model_path,
                        "temperature": self.config.temperature,
                        "alpha": self.config.alpha,
                        "epochs": self.config.num_epochs,
                    },
                )
            except Exception as e:
                logger.warning(f"[蒸馏] SwanLab 初始化失败: {e}")

        # 训练循环
        global_step = 0
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # 教师前向传播（不计算梯度）
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(
                        input_ids=input_ids, attention_mask=attention_mask
                    )

                # 学生前向传播
                student_outputs = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                # 蒸馏损失
                distill_loss = self.compute_distill_loss(
                    teacher_outputs.logits,
                    student_outputs.logits,
                    labels,
                    self.config.temperature,
                )

                # 硬标签损失（交叉熵）
                ce_loss = student_outputs.loss

                # 总损失
                loss = self.config.alpha * distill_loss + (1 - self.config.alpha) * ce_loss

                # 反向传播
                loss.backward()

                if (global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1

                if global_step % 10 == 0:
                    avg_loss = epoch_loss / num_batches
                    logger.info(
                        f"[蒸馏] Epoch {epoch+1}, Step {global_step}, "
                        f"Loss: {loss.item():.4f}, Avg: {avg_loss:.4f}"
                    )
                    if swanlab:
                        try:
                            swanlab.log({
                                "train/loss": loss.item(),
                                "train/avg_loss": avg_loss,
                                "train/distill_loss": distill_loss.item(),
                                "train/ce_loss": ce_loss.item(),
                            }, step=global_step)
                        except Exception:
                            pass

            # 每个 epoch 保存
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            logger.info(f"[蒸馏] Epoch {epoch+1} 完成, 平均 Loss: {avg_epoch_loss:.4f}")

        # 保存学生模型
        self.student_model.save_pretrained(self.config.student_output_dir)
        self.tokenizer.save_pretrained(self.config.student_output_dir)
        logger.info(f"[蒸馏] 蒸馏模型已保存到: {self.config.student_output_dir}")

        # 性能测试
        self._benchmark_inference()

        if swanlab:
            try:
                swanlab.finish()
            except Exception:
                pass

    def _benchmark_inference(self):
        """测试蒸馏后模型的推理耗时"""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("[蒸馏] 开始推理性能测试...")

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.student_output_dir, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.config.student_output_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()

        test_texts = [
            "### Instruction:\n提取医疗实体\n\n### Input:\n患者头痛发热三天\n\n### Output:\n",
            "### Instruction:\n提取医疗实体\n\n### Input:\n患者咳嗽咳痰伴胸闷\n\n### Output:\n",
        ]

        times = []
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 预热
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=50)

            # 计时
            start = time.time()
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=100)
            elapsed = (time.time() - start) * 1000  # ms
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        logger.info(f"[蒸馏] 推理耗时测试: {times}")
        logger.info(f"[蒸馏] 平均推理耗时: {avg_time:.1f}ms")
        if avg_time <= self.config.target_inference_ms:
            logger.info(f"[蒸馏] ✅ 达标！目标 ≤{self.config.target_inference_ms}ms")
        else:
            logger.warning(
                f"[蒸馏] ⚠️ 未达标，目标 ≤{self.config.target_inference_ms}ms，"
                f"实际 {avg_time:.1f}ms"
            )


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    """运行模型蒸馏"""
    import argparse

    parser = argparse.ArgumentParser(description="模型知识蒸馏")
    parser.add_argument("--teacher_path", type=str, default=settings.NER_LORA_PATH)
    parser.add_argument("--teacher_base", type=str, default=settings.QWEN35_4B_PATH)
    parser.add_argument("--student_path", type=str, default="./models_cache/Qwen3.5-1.5B")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/distilled")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    args = parser.parse_args()

    config = DistillConfig(
        teacher_model_path=args.teacher_path,
        teacher_base_path=args.teacher_base,
        student_model_path=args.student_path,
        student_output_dir=args.output_dir,
        num_epochs=args.epochs,
        temperature=args.temperature,
        alpha=args.alpha,
    )

    distiller = KnowledgeDistiller(config)
    distiller.train()
