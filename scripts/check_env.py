"""
scripts/check_env.py
====================
云诊医疗智能问诊系统 —— 环境自检脚本。

功能：
  1. 检查 Python 版本 ≥ 3.10
  2. 检查关键依赖包是否已安装
  3. 读取 .env 配置，检查密钥和密码是否已填写
  4. 4 个数据库端口连通性检查（socket 连接，超时 2 秒）
  5. 汇总输出：全部通过 / 需要修复的项

运行方式：
  python scripts/check_env.py
"""

from __future__ import annotations
import sys
import os
import socket
import importlib.util
from pathlib import Path

# ============================================================
# 常量定义
# ============================================================

# 项目根目录（scripts/ 的上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 要求的 Python 最低版本
MIN_PYTHON_VERSION = (3, 10)

# 关键依赖包列表（import 名称）
REQUIRED_PACKAGES = [
    "langgraph",
    "neo4j",
    "pymilvus",
    "redis",
    "asyncpg",
    "fastapi",
    "openai",
    "pydantic_settings",
]

# 数据库端口检查列表：(服务名, 主机, 端口)
DATABASE_CHECKS = [
    ("Neo4j", "localhost", 7687),
    ("Milvus", "localhost", 19530),
    ("RedisStack", "localhost", 6379),
    ("PostgreSQL", "localhost", 5432),
]

# 分隔线
SEP = "=" * 56


# ============================================================
# 辅助函数
# ============================================================

def print_header(title: str):
    """打印章节标题"""
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def print_pass(msg: str):
    """打印通过项"""
    print(f"  ✅ {msg}")


def print_fail(msg: str):
    """打印失败项"""
    print(f"  ❌ {msg}")


def print_warn(msg: str):
    """打印警告项"""
    print(f"  ⚠️  {msg}")


def check_port(host: str, port: int, timeout: float = 2.0) -> bool:
    """
    使用 socket 检查指定主机端口是否可连接。

    :param host: 主机名或 IP
    :param port: 端口号
    :param timeout: 超时秒数
    :return: True=可连接, False=不可连接
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except (socket.error, OSError):
        return False


# ============================================================
# 检查项 1：Python 版本
# ============================================================

def check_python_version() -> bool:
    """检查 Python 版本是否 ≥ 3.10"""
    print_header("1/5  Python 版本检查")
    current = sys.version_info
    current_str = f"{current.major}.{current.minor}.{current.micro}"
    required_str = f"{MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}"

    if current >= MIN_PYTHON_VERSION:
        print_pass(f"Python {current_str} ≥ {required_str}")
        return True
    else:
        print_fail(f"Python {current_str} < {required_str}，请升级 Python")
        return False


# ============================================================
# 检查项 2：关键依赖包
# ============================================================

def check_dependencies() -> tuple[bool, list[str]]:
    """
    检查关键依赖包是否已安装。
    使用 importlib.util.find_spec 检测，不会实际导入。

    :return: (是否全部安装, 缺失包列表)
    """
    print_header("2/5  关键依赖包检查")

    missing = []
    for pkg in REQUIRED_PACKAGES:
        spec = importlib.util.find_spec(pkg)
        if spec is not None:
            # 尝试获取版本号
            try:
                mod = importlib.import_module(pkg)
                ver = getattr(mod, "__version__", "已安装")
            except Exception:
                ver = "已安装"
            print_pass(f"{pkg} ({ver})")
        else:
            print_fail(f"{pkg} —— 未安装")
            missing.append(pkg)

    if not missing:
        print(f"\n  全部 {len(REQUIRED_PACKAGES)} 个依赖包已安装")
    else:
        print(f"\n  缺失 {len(missing)} 个包: {', '.join(missing)}")
        print(f"  安装命令: pip install {' '.join(missing)}")

    return len(missing) == 0, missing


# ============================================================
# 检查项 3：.env 配置检查
# ============================================================

def check_env_config() -> tuple[bool, list[str]]:
    """
    读取 .env 文件，检查关键配置项。

    :return: (是否全部通过, 问题列表)
    """
    print_header("3/5  .env 配置检查")

    env_path = PROJECT_ROOT / ".env"
    issues = []

    if not env_path.exists():
        print_fail(".env 文件不存在，请先从 .env.example 复制并填写配置")
        return False, [".env 文件不存在"]

    # 手动解析 .env（避免依赖 pydantic_settings）
    env_vars: dict[str, str] = {}
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                env_vars[key.strip()] = value.strip()

    # 检查 QWEN_API_KEY
    qwen_key = env_vars.get("QWEN_API_KEY", "")
    if not qwen_key or qwen_key in ("sk-xxxx", "sk-请填入你的DashScope密钥", ""):
        print_warn("QWEN_API_KEY 未填写真实密钥（当前为占位符）")
        issues.append("QWEN_API_KEY 未填写")
    else:
        # 只显示前6位 + ***
        masked = qwen_key[:6] + "***"
        print_pass(f"QWEN_API_KEY 已配置 ({masked})")

    # 检查 NEO4J_PASSWORD
    neo4j_pwd = env_vars.get("NEO4J_PASSWORD", "")
    if not neo4j_pwd or neo4j_pwd in ("your_password", ""):
        print_fail("NEO4J_PASSWORD 未填写或仍为默认值")
        issues.append("NEO4J_PASSWORD 未填写")
    else:
        print_pass("NEO4J_PASSWORD 已配置")

    # 检查 POSTGRES_PASSWORD
    pg_pwd = env_vars.get("POSTGRES_PASSWORD", "")
    if not pg_pwd or pg_pwd in ("your_password", ""):
        print_fail("POSTGRES_PASSWORD 未填写或仍为默认值")
        issues.append("POSTGRES_PASSWORD 未填写")
    else:
        print_pass("POSTGRES_PASSWORD 已配置")

    # 检查 SWANLAB_API_KEY（可选）
    swanlab_key = env_vars.get("SWANLAB_API_KEY", "")
    if not swanlab_key or swanlab_key in ("xxxx", "sk-请填入你的SwanLab密钥", ""):
        print_warn("SWANLAB_API_KEY 未填写（可选，仅影响训练监控）")
    else:
        print_pass("SWANLAB_API_KEY 已配置")

    # 其他配置项简要检查
    for key in ["QWEN_BASE_URL", "NEO4J_URI", "MILVUS_HOST", "REDIS_URL", "POSTGRES_HOST"]:
        val = env_vars.get(key, "")
        if val:
            print_pass(f"{key} = {val}")
        else:
            print_warn(f"{key} 未配置，将使用默认值")

    # 只有密码类问题才算失败
    critical_issues = [i for i in issues if "PASSWORD" in i]
    return len(critical_issues) == 0, issues


# ============================================================
# 检查项 4：数据库端口连通性
# ============================================================

def check_database_ports() -> tuple[bool, list[str]]:
    """
    检查 4 个数据库端口是否可连接。
    使用 socket 连接测试，超时 2 秒。

    :return: (是否全部可连接, 不可连接的服务列表)
    """
    print_header("4/5  数据库端口连通性检查")

    unreachable = []
    for service_name, host, port in DATABASE_CHECKS:
        if check_port(host, port, timeout=2.0):
            print_pass(f"{service_name} ({host}:{port}) 可连接")
        else:
            print_fail(f"{service_name} ({host}:{port}) 不可连接")
            unreachable.append(f"{service_name}({host}:{port})")

    if not unreachable:
        print(f"\n  全部 {len(DATABASE_CHECKS)} 个数据库端口可连接")
    else:
        print(f"\n  {len(unreachable)} 个数据库不可连接: {', '.join(unreachable)}")
        print(f"  启动命令: docker compose up -d")

    return len(unreachable) == 0, unreachable


# ============================================================
# 汇总报告
# ============================================================

def print_summary(results: dict[str, bool], details: dict) -> bool:
    """
    打印最终汇总报告。

    :param results: 各检查项名称 → 是否通过
    :param details: 各检查项的详细信息
    :return: True=全部通过, False=有问题
    """
    print_header("5/5  汇总报告")

    total = len(results)
    passed = sum(1 for v in results.values() if v)

    for name, ok in results.items():
        icon = "✅" if ok else "❌"
        status = "通过" if ok else "未通过"
        print(f"  {icon} {name}: {status}")

    print(f"\n  总计: {passed}/{total} 项通过")

    if all(results.values()):
        print(f"\n  🎉 全部检查通过！可以启动系统：python main.py")
        return True
    else:
        print(f"\n  ⚠️  存在未通过的检查项，请先修复后再启动系统。")
        print(f"\n  修复建议：")

        if not results.get("Python 版本", True):
            print(f"    - 安装 Python 3.10+ : https://www.python.org/downloads/")

        if not results.get("依赖包", True):
            missing = details.get("missing_packages", [])
            if missing:
                print(f"    - 安装缺失依赖: pip install {' '.join(missing)}")

        if not results.get(".env 配置", True):
            print(f"    - 编辑 .env 文件，填写数据库密码")

        if not results.get("数据库连通性", True):
            print(f"    - 启动数据库: docker compose up -d")
            print(f"    - 等待约 30 秒后重新运行此检查脚本")

        return False


# ============================================================
# 主入口
# ============================================================

def main():
    """环境自检主入口"""
    print(f"\n{'█' * 56}")
    print(f"  云诊医疗智能问诊系统（CDMIIS）环境自检")
    print(f"{'█' * 56}")
    print(f"  项目根目录: {PROJECT_ROOT}")

    results: dict[str, bool] = {}
    details: dict = {}

    # 检查 1：Python 版本
    results["Python 版本"] = check_python_version()

    # 检查 2：依赖包
    deps_ok, missing_pkgs = check_dependencies()
    results["依赖包"] = deps_ok
    details["missing_packages"] = missing_pkgs

    # 检查 3：.env 配置
    env_ok, env_issues = check_env_config()
    results[".env 配置"] = env_ok
    details["env_issues"] = env_issues

    # 检查 4：数据库端口
    db_ok, db_unreachable = check_database_ports()
    results["数据库连通性"] = db_ok
    details["db_unreachable"] = db_unreachable

    # 汇总
    all_pass = print_summary(results, details)

    # 返回退出码
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
