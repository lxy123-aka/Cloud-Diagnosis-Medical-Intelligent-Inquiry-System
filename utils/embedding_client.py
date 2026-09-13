"""
utils/embedding_client.py
=========================
统一嵌入模型客户端，优先使用本地模型，Qwen Embedding API 作为降级。

本地模型：BAAI/bge-large-zh-v1.5（1024 维）
模型路径：./models_cache/bge-large-zh-v1.5
通过 scripts/download_embedding_model.py 从镜像源下载。
"""

from __future__ import annotations
import numpy as np
from loguru import logger
from configs.settings import settings


class EmbeddingClient:
    """
    嵌入模型客户端。
    优先使用本地 bge-large-zh-v1.5 模型，Qwen API 作为降级方案。
    """

    def __init__(self):
        self._use_api = False  # 优先使用本地模型
        self._api_model = "text-embedding-v3"  # Qwen 中文嵌入模型（降级用）
        self._local_model = None
        self._local_model_path = str(settings.BASE_DIR / "models_cache" / "bge-large-zh-v1.5")
        
        # 启动时自动加载本地模型
        if not self._load_local_model():
            logger.warning("[EmbeddingClient] 本地模型加载失败，将使用 Qwen Embedding API")
            self._use_api = True
        else:
            logger.info("[EmbeddingClient] 本地模型加载完成，使用 bge-large-zh-v1.5")

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
            # 本地模型失败时尝试 API 降级
            if not self._use_api:
                logger.warning("[EmbeddingClient] 本地模型编码失败，尝试 Qwen API...")
                try:
                    embeddings = await self._encode_via_api(texts)
                    if normalize:
                        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                        norms = np.where(norms == 0, 1, norms)
                        embeddings = embeddings / norms
                    return embeddings
                except Exception as api_err:
                    logger.error(f"[EmbeddingClient] API 降级也失败: {api_err}")
            # 最终降级：返回随机向量
            logger.warning(
                f"[EmbeddingClient] 降级为随机向量（{len(texts)} 条文本），"
                "检索结果将无实际语义意义"
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

    def _load_local_model(self) -> bool:
        """
        懒加载本地 SentenceTransformer 模型。
        返回 True 表示加载成功，False 表示模型不存在或加载失败。
        """
        if self._local_model is not None:
            return True  # 已加载

        import os
        if not os.path.exists(self._local_model_path):
            logger.warning(
                f"[EmbeddingClient] 本地模型不存在: {self._local_model_path}\n"
                f"  运行 python scripts/download_embedding_model.py 下载"
            )
            return False

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"[EmbeddingClient] 加载本地模型: {self._local_model_path}")
            self._local_model = SentenceTransformer(self._local_model_path)
            logger.info("[EmbeddingClient] 本地模型加载成功")
            return True
        except Exception as e:
            logger.error(f"[EmbeddingClient] 本地模型加载失败: {e}")
            return False

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
