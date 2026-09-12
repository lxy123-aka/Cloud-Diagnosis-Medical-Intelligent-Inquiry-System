"""
multimodal/vl_infer.py
======================
Qwen-VL 多模态推理模块。

功能：
  1. 接收图片二进制数据
  2. 调用 Qwen-VL 模型解析 CT/B超影像、检验报告图片
  3. 提取医学实体并返回解析结果
  4. 支持 base64 编码和本地文件路径两种输入方式
"""

from __future__ import annotations
import base64
import json
import re
from pathlib import Path
from loguru import logger
from openai import AsyncOpenAI

from configs.settings import settings


class QwenVLInfer:
    """Qwen-VL 多模态推理服务"""

    def __init__(self):
        self._client = AsyncOpenAI(
            api_key=settings.QWEN_API_KEY,
            base_url=settings.QWEN_BASE_URL,
        )
        self._model = settings.QWEN_VL_MODEL_NAME

    async def analyze_medical_image(
        self,
        image_data: bytes = None,
        image_path: str = None,
        prompt: str = "请详细分析这张医学图片的内容。",
    ) -> str:
        """
        分析医学影像图片。

        支持两种输入方式：
          1. image_data: 图片二进制数据
          2. image_path: 图片文件路径

        :param image_data: 图片二进制数据
        :param image_path: 图片文件路径
        :param prompt: 分析提示语
        :return: 分析结果文本
        """
        # 构造图片 URL（base64 或文件路径）
        image_url = self._prepare_image_url(image_data, image_path)

        logger.info(f"[QwenVL] 开始分析医学图片，模型: {self._model}")

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一位资深医学影像诊断专家，擅长分析CT、B超、X光、"
                            "化验单等医学图片。请仔细观察图片，提取关键医学信息。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
                max_tokens=2048,
                temperature=0.2,
            )

            result = response.choices[0].message.content
            logger.info(f"[QwenVL] 图片分析完成，结果长度: {len(result)}")
            return result

        except Exception as e:
            logger.error(f"[QwenVL] 图片分析失败: {e}")
            return f"图片分析失败: {str(e)}"

    async def extract_medical_entities(self, image_data: bytes = None, image_path: str = None) -> dict:
        """
        从医学图片中提取结构化医学实体。

        :return: 包含提取到的实体的字典
        """
        prompt = (
            "请分析这张医学图片，提取以下信息并以JSON格式输出：\n"
            "{\n"
            '  "image_type": "图片类型（CT/B超/X光/化验单/心电图等）",\n'
            '  "key_findings": ["关键发现列表"],\n'
            '  "abnormal_indicators": [\n'
            '    {"name": "指标名称", "value": "数值", "reference": "参考范围", "status": "偏高/偏低/正常"}\n'
            "  ],\n"
            '  "possible_conditions": ["可能的疾病/状况"],\n'
            '  "urgency_level": "紧急程度（正常/需关注/需尽快就医/紧急）",\n'
            '  "recommendations": ["建议"]\n'
            "}\n"
            "只输出JSON，不要其他文字。"
        )

        result = await self.analyze_medical_image(
            image_data=image_data,
            image_path=image_path,
            prompt=prompt,
        )

        # 尝试解析 JSON
        try:
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        # 解析失败，返回原始文本
        return {"raw_analysis": result}

    async def analyze_report_image(self, image_data: bytes = None, image_path: str = None) -> str:
        """
        专门用于检验报告图片的解读。

        :return: 结构化解读文本
        """
        prompt = (
            "这是一张医学检验报告图片，请按以下格式进行解读：\n\n"
            "【报告类型】识别报告类型\n"
            "【关键指标】列出关键检测指标及其数值\n"
            "【异常项目】标注超出参考范围的异常项目\n"
            "【临床意义】解释异常指标的临床意义\n"
            "【综合评估】给出综合评估意见\n"
            "【就医建议】给出就医建议"
        )

        return await self.analyze_medical_image(
            image_data=image_data,
            image_path=image_path,
            prompt=prompt,
        )

    async def analyze_ct_scan(self, image_data: bytes = None, image_path: str = None) -> str:
        """
        专门用于 CT 影像的分析。
        """
        prompt = (
            "这是一张CT影像图片，请按以下格式进行分析：\n\n"
            "【扫描部位】识别CT扫描部位\n"
            "【影像表现】描述主要影像表现\n"
            "【异常发现】标注异常区域\n"
            "【影像诊断】给出影像诊断意见\n"
            "【建议】给出进一步检查或就医建议\n\n"
            "注意：如无法确定，请如实说明。"
        )

        return await self.analyze_medical_image(
            image_data=image_data,
            image_path=image_path,
            prompt=prompt,
        )

    async def analyze_ultrasound(self, image_data: bytes = None, image_path: str = None) -> str:
        """
        专门用于 B超影像的分析。
        """
        prompt = (
            "这是一张B超影像图片，请按以下格式进行分析：\n\n"
            "【检查部位】识别B超检查部位\n"
            "【脏器形态】描述脏器形态、大小\n"
            "【回声特征】描述回声特征\n"
            "【异常发现】标注异常区域\n"
            "【超声提示】给出超声提示\n"
            "【建议】给出进一步建议\n\n"
            "注意：如无法确定，请如实说明。"
        )

        return await self.analyze_medical_image(
            image_data=image_data,
            image_path=image_path,
            prompt=prompt,
        )

    # ============================================================
    # 内部辅助方法
    # ============================================================

    def _prepare_image_url(self, image_data: bytes = None, image_path: str = None) -> str:
        """
        将图片转为 Qwen-VL 可接受的 URL 格式。

        支持：
          1. 二进制数据 → base64 data URL
          2. 文件路径 → file:// URL
        """
        if image_data:
            # 二进制数据 → base64
            b64 = base64.b64encode(image_data).decode("utf-8")
            return f"data:image/png;base64,{b64}"

        elif image_path:
            path = Path(image_path)
            if path.exists():
                # 本地文件 → base64
                with open(path, "rb") as f:
                    data = f.read()
                b64 = base64.b64encode(data).decode("utf-8")
                # 根据扩展名判断 MIME
                suffix = path.suffix.lower()
                mime_map = {
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".png": "image/png",
                    ".gif": "image/gif",
                    ".bmp": "image/bmp",
                    ".webp": "image/webp",
                }
                mime = mime_map.get(suffix, "image/png")
                return f"data:{mime};base64,{b64}"
            else:
                raise FileNotFoundError(f"图片文件不存在: {image_path}")
        else:
            raise ValueError("必须提供 image_data 或 image_path 之一")


# ============================================================
# 单独调试入口
# ============================================================

if __name__ == "__main__":
    """单独测试 Qwen-VL 多模态推理"""
    import asyncio

    async def test_vl():
        vl = QwenVLInfer()

        # 测试1：从文件路径分析（如果有测试图片）
        test_image = "test_report.jpg"
        if Path(test_image).exists():
            result = await vl.analyze_report_image(image_path=test_image)
            print(f"报告解读:\n{result}")
        else:
            print(f"测试图片 {test_image} 不存在，跳过文件测试")

        # 测试2：实体提取
        print("\n请提供医学图片路径进行测试：")
        print("(调试模式下跳过)")

    asyncio.run(test_vl())
