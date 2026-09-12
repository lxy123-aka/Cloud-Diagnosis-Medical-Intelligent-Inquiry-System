"""
scripts/env_encrypt.py
======================
使用 Windows DPAPI 加密/解密 .env 文件（绑定当前 Windows 用户）。

用法：
  python scripts/env_encrypt.py encrypt            # .env → .env.enc（默认删除明文）
  python scripts/env_encrypt.py encrypt --keep     # 保留明文 .env
  python scripts/env_encrypt.py decrypt            # .env.enc → .env（恢复明文）
  python scripts/env_encrypt.py verify             # 验证 .env.enc 可解密

原理：
  - 使用 Windows CryptProtectData（DPAPI），密文绑定当前 Windows 用户，
    其他用户 / 其他机器无法解密。
  - 加密后无需改动程序：configs/settings.py 启动时若发现 .env 缺失而
    .env.enc 存在，会自动解密并加载到环境变量（明文不落盘）。
  - 仅支持 Windows；Linux/macOS 请改用系统 keyring 或保持明文 .env。

安全说明：
  - 加密能防"磁盘泄露 / 仓库误传"，但防不了"key 已在对话或截图中出现"。
    若 key 曾经外泄过，最稳妥的做法仍是去阿里云百炼重置。
"""

from __future__ import annotations
import sys
import ctypes
import ctypes.wintypes
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# DPAPI 封装（ctypes 调用 Windows Crypt32）
# ============================================================

class DATA_BLOB(ctypes.Structure):
    _fields_ = [
        ("cbData", ctypes.wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_char)),
    ]


def protect(data: bytes) -> bytes:
    """加密字节串（绑定当前用户）"""
    if not data:
        raise ValueError("空数据无法加密")
    buf = ctypes.create_string_buffer(data)
    blob_in = DATA_BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
    blob_out = DATA_BLOB()
    ok = ctypes.windll.crypt32.CryptProtectData(
        ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)
    )
    if not ok:
        raise ctypes.WinError()
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(blob_out.pbData)


def unprotect(data: bytes) -> bytes:
    """解密字节串（仅当前用户可解密）"""
    if not data:
        raise ValueError("空数据无法解密")
    buf = ctypes.create_string_buffer(data)
    blob_in = DATA_BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
    blob_out = DATA_BLOB()
    ok = ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)
    )
    if not ok:
        raise ctypes.WinError()
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(blob_out.pbData)


# ============================================================
# 命令行入口
# ============================================================

def cmd_encrypt(keep: bool = False):
    env_file = PROJECT_ROOT / ".env"
    enc_file = PROJECT_ROOT / ".env.enc"
    if not env_file.exists():
        print("❌ 未找到 .env 文件，无需加密")
        return 1

    cipher = protect(env_file.read_bytes())
    enc_file.write_bytes(cipher)
    print(f"✅ 已加密: .env → {enc_file.name}（{len(cipher)} 字节，绑定当前用户）")

    if keep:
        print("ℹ️  明文 .env 已保留（--keep）")
    else:
        env_file.unlink()
        print("✅ 明文 .env 已删除（加密完成）")
        print("   settings.py 启动时会自动解密 .env.enc 加载，程序不受影响")
    return 0


def cmd_decrypt():
    env_file = PROJECT_ROOT / ".env"
    enc_file = PROJECT_ROOT / ".env.enc"
    if not enc_file.exists():
        print("❌ 未找到 .env.enc 文件")
        return 1

    try:
        plain = unprotect(enc_file.read_bytes()).decode("utf-8")
    except Exception as e:
        print(f"❌ 解密失败（非当前用户或文件损坏）: {e}")
        return 1

    env_file.write_text(plain, encoding="utf-8")
    print(f"✅ 已恢复明文 .env（{len(plain)} 字节）")
    return 0


def cmd_verify():
    enc_file = PROJECT_ROOT / ".env.enc"
    if not enc_file.exists():
        print("❌ 未找到 .env.enc 文件")
        return 1

    try:
        plain = unprotect(enc_file.read_bytes()).decode("utf-8")
        n_keys = sum(1 for l in plain.splitlines() if l.strip() and not l.startswith("#") and "=" in l)
        print(f"✅ 解密验证通过: {n_keys} 个配置项可正常解密")
        # 不打印明文内容
        return 0
    except Exception as e:
        print(f"❌ 解密失败: {e}")
        return 1


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("encrypt", "decrypt", "verify"):
        print(__doc__)
        return 1

    action = sys.argv[1]
    if action == "encrypt":
        return cmd_encrypt(keep="--keep" in sys.argv)
    if action == "decrypt":
        return cmd_decrypt()
    return cmd_verify()


if __name__ == "__main__":
    sys.exit(main())
