"""
scripts/download_embedding_model.py
====================================
从镜像源下载嵌入模型 BAAI/bge-large-zh-v1.5。

国内网络环境下 HuggingFace 直连经常超时/SSL 失败，
本脚本依次尝试以下镜像源：
  1. hf-mirror.com（HuggingFace 国内镜像）
  2. ModelScope（阿里云模型中心）
  3. 官方 HuggingFace（兜底）

下载完成后模型保存到 ./models_cache/bge-large-zh-v1.5，
并自动验证模型可正常加载和编码。

运行方式：
  python scripts/download_embedding_model.py
"""

from __future__ import annotations
import os
import sys
import shutil
import time
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "BAAI/bge-large-zh-v1.5"
LOCAL_DIR = PROJECT_ROOT / "models_cache" / "bge-large-zh-v1.5"


def print_step(step: int, msg: str):
    print(f"\n[{step}] {msg}")


# ============================================================
# 方案 1：hf-mirror.com 镜像下载
# ============================================================

def download_via_hf_mirror() -> bool:
    """通过 hf-mirror.com 国内镜像下载模型"""
    print_step(1, "尝试 hf-mirror.com 国内镜像...")

    # 设置 HuggingFace 镜像环境变量
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    try:
        from huggingface_hub import snapshot_download

        print(f"  镜像地址: https://hf-mirror.com/{MODEL_NAME}")
        print(f"  保存到: {LOCAL_DIR}")

        start = time.time()
        snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=str(LOCAL_DIR),
            local_dir_use_symlinks=False,  # 直接复制文件，不用符号链接
            resume_download=True,          # 支持断点续传
        )
        elapsed = time.time() - start
        print(f"  ✅ 下载完成！耗时 {elapsed:.1f}s")
        return True

    except ImportError:
        print("  ⚠️ huggingface_hub 未安装，尝试安装...")
        os.system(f"{sys.executable} -m pip install huggingface_hub -q")
        print("  请重新运行脚本")
        return False

    except Exception as e:
        print(f"  ❌ hf-mirror 下载失败: {e}")
        return False


# ============================================================
# 方案 2：ModelScope 下载
# ============================================================

def download_via_modelscope() -> bool:
    """通过 ModelScope（阿里云）下载模型"""
    print_step(2, "尝试 ModelScope 阿里云镜像...")

    try:
        from modelscope.hub.snapshot_download import snapshot_download as ms_download

        print(f"  ModelScope 模型 ID: {MODEL_NAME}")
        print(f"  保存到: {LOCAL_DIR}")

        start = time.time()
        # ModelScope 下载
        ms_download(
            model_id=MODEL_NAME,
            cache_dir=str(LOCAL_DIR.parent / "modelscope_cache"),
        )
        elapsed = time.time() - start

        # ModelScope 下载路径不同，需要复制到目标位置
        ms_cache = LOCAL_DIR.parent / "modelscope_cache" / "models" / MODEL_NAME.replace("/", "___")
        if ms_cache.exists() and not LOCAL_DIR.exists():
            shutil.copytree(str(ms_cache), str(LOCAL_DIR))
            print(f"  已从 ModelScope 缓存复制到: {LOCAL_DIR}")

        print(f"  ✅ 下载完成！耗时 {elapsed:.1f}s")
        return True

    except ImportError:
        print("  ⚠️ modelscope 未安装，尝试安装...")
        os.system(f"{sys.executable} -m pip install modelscope -q")
        print("  请重新运行脚本")
        return False

    except Exception as e:
        print(f"  ❌ ModelScope 下载失败: {e}")
        return False


# ============================================================
# 方案 3：官方 HuggingFace（兜底）
# ============================================================

def download_via_official_hf() -> bool:
    """通过官方 HuggingFace 下载（兜底）"""
    print_step(3, "尝试官方 HuggingFace（可能需要科学上网）...")

    # 清除镜像环境变量
    os.environ.pop("HF_ENDPOINT", None)

    try:
        from huggingface_hub import snapshot_download

        print(f"  官方地址: https://huggingface.co/{MODEL_NAME}")
        print(f"  保存到: {LOCAL_DIR}")

        start = time.time()
        snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=str(LOCAL_DIR),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        elapsed = time.time() - start
        print(f"  ✅ 下载完成！耗时 {elapsed:.1f}s")
        return True

    except Exception as e:
        print(f"  ❌ 官方 HuggingFace 下载失败: {e}")
        return False


# ============================================================
# 方案 4：sentence-transformers 直接下载
# ============================================================

def download_via_sentence_transformers() -> bool:
    """通过 sentence-transformers 库直接下载（内部会自动处理镜像）"""
    print_step(4, "尝试 sentence-transformers 内置下载...")

    # 设置镜像
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    try:
        from sentence_transformers import SentenceTransformer

        print(f"  模型: {MODEL_NAME}")
        start = time.time()
        model = SentenceTransformer(MODEL_NAME, cache_folder=str(LOCAL_DIR.parent))
        elapsed = time.time() - start

        # 如果下载成功，模型已经在缓存中
        # 复制到目标位置方便统一管理
        if not LOCAL_DIR.exists():
            # sentence-transformers 缓存路径
            st_cache = LOCAL_DIR.parent / "sentence_transformer_cache"
            if st_cache.exists():
                # 找到最新的缓存目录
                cache_dirs = list(st_cache.iterdir())
                if cache_dirs:
                    latest = max(cache_dirs, key=lambda d: d.stat().st_mtime)
                    shutil.copytree(str(latest), str(LOCAL_DIR))
                    print(f"  已从 ST 缓存复制到: {LOCAL_DIR}")

        print(f"  ✅ 下载完成！耗时 {elapsed:.1f}s")
        return True

    except ImportError:
        print("  ⚠️ sentence-transformers 未安装")
        print("  运行: pip install sentence-transformers")
        return False

    except Exception as e:
        print(f"  ❌ sentence-transformers 下载失败: {e}")
        return False


# ============================================================
# 验证模型
# ============================================================

def verify_model() -> bool:
    """验证下载的模型是否可正常加载和编码"""
    print_step(5, "验证模型...")

    if not LOCAL_DIR.exists():
        # 检查 sentence-transformers 缓存中是否有
        st_cache = LOCAL_DIR.parent / "sentence_transformer_cache"
        if st_cache.exists():
            print(f"  本地目录不存在，但发现 ST 缓存: {st_cache}")
            print(f"  将尝试从缓存加载验证")
        else:
            print(f"  ❌ 模型目录不存在: {LOCAL_DIR}")
            return False

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        # 优先从本地目录加载
        if LOCAL_DIR.exists() and (LOCAL_DIR / "config.json").exists():
            model_path = str(LOCAL_DIR)
            print(f"  从本地目录加载: {model_path}")
        else:
            model_path = MODEL_NAME
            print(f"  从缓存加载: {model_path}")

        model = SentenceTransformer(model_path)

        # 测试编码
        test_texts = ["头痛", "发热", "咳嗽", "腹痛"]
        embeddings = model.encode(test_texts, normalize_embeddings=True)

        print(f"  编码形状: {embeddings.shape}")
        print(f"  向量维度: {embeddings.shape[1]}")

        # 计算余弦相似度矩阵
        similarity = embeddings @ embeddings.T
        print(f"\n  余弦相似度矩阵:")
        print(f"  {'':>8}", end="")
        for t in test_texts:
            print(f"  {t:>6}", end="")
        print()
        for i, t1 in enumerate(test_texts):
            print(f"  {t1:>6}", end="")
            for j in range(len(test_texts)):
                print(f"  {similarity[i][j]:.3f} ", end="")
            print()

        # 基本验证
        assert embeddings.shape == (4, 1024), f"维度不匹配: {embeddings.shape}"
        # 对角线应接近 1.0（归一化后自相似度）
        for i in range(4):
            assert abs(similarity[i][i] - 1.0) < 0.01, f"自相似度异常: {similarity[i][i]}"
        # 不同症状相似度应 < 1.0
        assert similarity[0][1] < 0.95, "不同症状相似度过高，模型可能有问题"

        print(f"\n  ✅ 模型验证通过！维度={embeddings.shape[1]}，语义区分正常")
        return True

    except Exception as e:
        print(f"  ❌ 模型验证失败: {e}")
        return False


# ============================================================
# 更新 embedding_client.py 配置
# ============================================================

def print_usage_guide():
    """打印使用指南"""
    print("\n" + "=" * 60)
    print("  使用指南")
    print("=" * 60)

    if LOCAL_DIR.exists():
        print(f"\n  模型已下载到: {LOCAL_DIR}")
        print(f"\n  在 embedding_client.py 中加载本地模型：")
        print(f"""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(r"{LOCAL_DIR}")
    embeddings = model.encode(["头痛", "发热"])
""")
        print(f"  或在 .env 中配置：")
        print(f'    EMBEDDING_MODEL={LOCAL_DIR}')
    else:
        print("\n  模型未下载成功。")
        print("  系统将继续使用 Qwen Embedding API（text-embedding-v3）。")
        print("  本地模型仅作为 API 不可用时的降级方案。")

    print(f"\n  手动下载备选方案：")
    print(f"    1. 浏览器打开 https://hf-mirror.com/{MODEL_NAME}")
    print(f"    2. 点击 'Download' 下载全部文件")
    print(f"    3. 解压到 {LOCAL_DIR}")
    print("=" * 60)


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("  嵌入模型下载工具")
    print(f"  模型: {MODEL_NAME}")
    print(f"  目标: {LOCAL_DIR}")
    print("=" * 60)

    # 检查是否已存在
    if LOCAL_DIR.exists() and (LOCAL_DIR / "config.json").exists():
        print(f"\n  ✅ 模型已存在: {LOCAL_DIR}")
        print(f"  跳过下载，直接验证...")
        if verify_model():
            print_usage_guide()
            return
        print(f"  ⚠️ 已有模型验证失败，尝试重新下载...")

    # 确保目标父目录存在
    LOCAL_DIR.parent.mkdir(parents=True, exist_ok=True)

    # 依次尝试各下载方案
    success = False

    # 方案 1：hf-mirror（最推荐，国内最快）
    success = download_via_hf_mirror()
    if success and verify_model():
        print_usage_guide()
        return

    # 方案 2：ModelScope
    success = download_via_modelscope()
    if success and verify_model():
        print_usage_guide()
        return

    # 方案 3：sentence-transformers 内置下载
    success = download_via_sentence_transformers()
    if success and verify_model():
        print_usage_guide()
        return

    # 方案 4：官方 HuggingFace（兜底）
    success = download_via_official_hf()
    if success and verify_model():
        print_usage_guide()
        return

    # 全部失败
    print("\n" + "=" * 60)
    print("  ❌ 所有下载方案均失败")
    print("=" * 60)
    print("\n  建议手动下载：")
    print(f"    1. 浏览器打开: https://hf-mirror.com/{MODEL_NAME}")
    print(f"    2. 点击 'Clone' 或下载 ZIP")
    print(f"    3. 解压到: {LOCAL_DIR}")
    print(f"\n  或者继续使用 Qwen Embedding API（无需本地模型）：")
    print(f"    系统已默认使用 text-embedding-v3 API，本地模型仅作降级备用。")
    print("=" * 60)


if __name__ == "__main__":
    main()
