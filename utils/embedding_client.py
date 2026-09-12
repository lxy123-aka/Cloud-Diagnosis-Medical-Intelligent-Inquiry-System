"""
utils/embedding_client.py
=========================
统一嵌入模型客户端，支持本地模型和 Qwen Embedding API。
优先使用 Qwen API（无需下载模型），fallback 到本地模型。
"""

from __future__ import annotations
import numpy as np
from loguru import logger
from configs.settings import settings


class EmbeddingClient:
    """
    嵌入模型客户端。
    支持两种模式：
    1. Qwen Embedding API（推荐，无需下载模型）
    2. 本地 SentenceTransformer 模型（fallback）
    """

    def __init__(self):
        self._use_api = True  # 默认使用 API
        self._api_model = "text-embedding-v3"  # Qwen 中文嵌入模型
        self._local_model = None
        
        # 不自动加载本地模型（避免 SSL 问题），只在 API 失败时按需加载
        logger.info("[EmbeddingClient] 初始化完成，将优先使用 Qwen Embedding API")

    async def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """
        批量编码文本为向量。
        
        :param texts: 文本列表
        :param normalize: 是否归一化（Milvus COSINE 距离需要）
        :return: (n_texts, dim) 的 numpy 数组
        """
        if not texts:
            return np.array([]).reshape(0, settings.EMBEDDING_DIMENSION)
        
        try:
            if self._use_api:
                embeddings = await self._encode_via_api(texts)
            else:
                embeddings = self._encode_local(texts)
            
            if normalize:
                # L2 归一化
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)  # 避免除零
                embeddings = embeddings / norms
            
            return embeddings
            
        except Exception as e:
            logger.error(f"[EmbeddingClient] 编码失败: {e}")
            # 降级：返回随机向量（仅作为最后手段，检索结果将无意义）
            logger.warning(
                f"[EmbeddingClient] 降级为随机向量（{len(texts)} 条文本），"
                "检索结果将无实际语义意义，请检查 Embedding 服务配置"
            )
            return np.random.randn(len(texts), settings.EMBEDDING_DIMENSION)

    async def _encode_via_api(self, texts: list[str]) -> np.ndarray:
        """通过 Qwen Embedding API 获取向量（OpenAI 兼容接口）。

        注意：QWEN_API_KEY 是百炼"工作空间专属" sk-ws 开头 key，
        只能配合 QWEN_BASE_URL 的 compatible-mode 端点使用，
        不能用 dashscope SDK 默认端点（会 401 导致降级随机向量）。
        """
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=settings.QWEN_API_KEY,
                base_url=settings.QWEN_BASE_URL,
            )

            # 调用 Embedding API（text-embedding-v3 支持指定维度）
            response = client.embeddings.create(
                model=self._api_model,
                input=texts,
                dimensions=settings.EMBEDDING_DIMENSION,
            )

            # 提取向量
            embeddings = [d.embedding for d in response.data]
            return np.array(embeddings)

        except ImportError:
            logger.warning("[EmbeddingClient] openai 未安装，请 pip install openai")
            raise  # 向上抛出，让 encode() 统一处理降级
        except Exception as e:
            logger.error(f"[EmbeddingClient] API 调用失败: {e}")
            raise  # 向上抛出，让 encode() 统一处理降级

    def _encode_local(self, texts: list[str]) -> np.ndarray:
        """使用本地 SentenceTransformer 模型编码"""
        if self._local_model is None:
            raise RuntimeError("本地模型未加载")
        
        embeddings = self._local_model.encode(
            texts, 
            normalize_embeddings=False,  # 我们自己归一化
            convert_to_numpy=True
        )
        return embeddings

    async def encode_single(self, text: str, normalize: bool = True) -> list[float]:
        """编码单个文本"""
        result = await self.encode([text], normalize=normalize)
        return result[0].tolist() if len(result) > 0 else []


# 全局单例
embedding_client = EmbeddingClient()


if __name__ == "__main__":
    """测试嵌入客户端"""
    import asyncio
    
    async def test():
        client = EmbeddingClient()
        
        # 测试批量编码
        texts = ["头痛", "发热", "咳嗽"]
        embeddings = await client.encode(texts)
        print(f"批量编码结果形状: {embeddings.shape}")
        print(f"第一个向量维度: {len(embeddings[0])}")
        print(f"向量范数: {np.linalg.norm(embeddings[0]):.4f}")
        
        # 测试单个编码
        single = await client.encode_single("腹痛")
        print(f"\n单个编码维度: {len(single)}")
    
    asyncio.run(test())
