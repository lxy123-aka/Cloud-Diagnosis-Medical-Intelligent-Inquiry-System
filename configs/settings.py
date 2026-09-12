"""
全局配置模块
从 .env 文件和环境变量加载所有外部服务连接与模型配置

支持 .env 加密模式：若 .env 缺失而 .env.enc 存在（由 scripts/env_encrypt.py
用 Windows DPAPI 生成），启动时自动解密并加载到环境变量，明文不落盘。
"""

import os
import ctypes
import ctypes.wintypes
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import field_validator


# ============================================================
# .env 加密文件自动解密（Windows DPAPI，绑定当前用户）
# ============================================================

class _DATA_BLOB(ctypes.Structure):
    _fields_ = [
        ("cbData", ctypes.wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_char)),
    ]


def _dpapi_unprotect(data: bytes) -> bytes:
    """Windows DPAPI 解密（仅当前用户可解）"""
    if not data:
        return b""
    buf = ctypes.create_string_buffer(data)
    blob_in = _DATA_BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
    blob_out = _DATA_BLOB()
    ok = ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)
    )
    if not ok:
        raise ctypes.WinError()
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(blob_out.pbData)


def _decrypt_env_if_needed() -> None:
    """
    若 .env 缺失但 .env.enc 存在，解密并写入环境变量（不落盘明文）。
    环境变量优先级高于 .env 文件，因此后续 BaseSettings 会自动取到解密值。
    """
    base_dir = Path(__file__).resolve().parent.parent
    env_file = base_dir / ".env"
    enc_file = base_dir / ".env.enc"

    if env_file.exists() or not enc_file.exists():
        return

    try:
        plain = _dpapi_unprotect(enc_file.read_bytes()).decode("utf-8")
        for line in plain.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)
    except Exception as e:  # 解密失败不阻断启动，仅记录（可后续手动解密排查）
        print(f"[settings] .env.enc 自动解密失败: {e}")


_decrypt_env_if_needed()


class Settings(BaseSettings):
    """全局配置类，从 .env 自动加载"""

    # ===== 项目基础 =====
    PROJECT_NAME: str = "云诊医疗智能问诊系统"
    VERSION: str = "1.0.0"
    BASE_DIR: Path = Path(__file__).resolve().parent.parent

    # ===== LLM 配置 =====
    QWEN_API_KEY: str = ""
    QWEN_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    QWEN_MODEL_NAME: str = "qwen-plus"
    QWEN_VL_MODEL_NAME: str = "qwen-vl-max"
    QWEN_TEMPERATURE: float = 0.3
    QWEN_MAX_TOKENS: int = 2048

    # ===== Neo4j 知识图谱 =====
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = ""

    # ===== Milvus 向量库 =====
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530

    # ===== Redis =====
    REDIS_URL: str = "redis://localhost:6379/0"

    # ===== PostgreSQL =====
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "cloud_diagnosis"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = ""

    @property
    def postgres_dsn(self) -> str:
        """异步 PostgreSQL 连接字符串"""
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def postgres_asyncpg_dsn(self) -> str:
        """asyncpg 原生连接字符串"""
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # ===== 模型路径 =====
    QWEN35_4B_PATH: str = "./models_cache/Qwen3.5-4B"
    QWEN_VL_PATH: str = "./models_cache/Qwen-VL-Chat"
    NER_LORA_PATH: str = "./checkpoints/ner_lora"
    VL_LORA_PATH: str = "./checkpoints/vl_lora"

    # ===== SwanLab =====
    SWANLAB_API_KEY: str = ""
    SWANLAB_PROJECT: str = "cloud_diagnosis_medical"

    # ===== 诊断收敛参数 =====
    CONFIDENCE_THRESHOLD: float = 0.85   # Top1 置信度阈值（提高到0.85避免过早收敛）
    CONFIDENCE_GAP: float = 0.30         # Top1-Top2 差值阈值
    MIN_INQUIRY_ROUNDS: int = 3          # 最小追问轮数（至少追问3轮才允许收敛）
    MAX_INQUIRY_ROUNDS: int = 10         # 最大追问轮数

    # ===== 症状标准化权重 =====
    SYMPTOM_GRAPH_WEIGHT: float = 0.5    # 图谱匹配权重
    SYMPTOM_VECTOR_WEIGHT: float = 0.3   # 向量召回权重
    SYMPTOM_LLM_WEIGHT: float = 0.2      # LLM 提取权重

    # ===== 嵌入模型 =====
    EMBEDDING_MODEL: str = "BAAI/bge-large-zh-v1.5"
    EMBEDDING_DIMENSION: int = 1024

    # ===== 服务 =====
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @field_validator('QWEN_API_KEY')
    @classmethod
    def validate_api_key(cls, v):
        if not v or v.strip() == "":
            raise ValueError("QWEN_API_KEY 必须配置，不能为空。请在 .env 文件中设置有效的 DashScope API Key")
        return v


# 全局单例
settings = Settings()
