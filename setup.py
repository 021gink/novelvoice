#!/usr/bin/env python3
"""
Setup script for Multi-Engine TTS Orchestrator (Simplified Version)
保留基本安装功能作为env_manager的回退方案
"""
import os
import subprocess
import sys
from pathlib import Path
import venv


def install_dependencies():
    """安装Python依赖 - 简化版本"""
    print("📦 Installing dependencies using simplified setup...")
    print("💡 建议使用 env_manager.py 进行更稳定的安装")
    
    engines = ["xtts", "chattts", "kokoro", "cosyvoice"]
    
    success_count = 0
    for engine_name in engines:
        try:
            result = install_engine_dependencies(engine_name)
            if result["success"]:
                print(f"✅ {engine_name}: 依赖安装成功")
                success_count += 1
            else:
                print(f"❌ {engine_name}: 安装失败 - {result['error']}")
        except Exception as e:
            print(f"❌ {engine_name}: 异常 - {e}")
    
    print(f"\n📊 结果: {success_count}/{len(engines)} 个引擎安装成功")
    return success_count == len(engines)


def install_engine_dependencies(engine_name: str):
    """为单个引擎安装依赖 - 简化版本"""
    requirements_file = f"requirements-{engine_name}.txt"
    venv_path = f"venvs/{engine_name}"
    
    if not Path(requirements_file).exists():
        return {"success": False, "error": f"需求文件 {requirements_file} 不存在"}
    
  
    if not is_venv_valid(venv_path):
        print(f"🔧 为 {engine_name} 创建虚拟环境...")
        venv.create(venv_path, with_pip=True)
    
    python_exe = get_python_executable(venv_path)
    

    print(f"🔧 为 {engine_name} 安装依赖...")
    
    try:

        result = subprocess.run(
            [str(python_exe), "-m", "pip", "install", "-r", requirements_file],
            capture_output=True, text=True, timeout=600
        )
        
        if result.returncode == 0:
            return {"success": True, "error": None}
        else:
            return {"success": False, "error": result.stderr[:200]}
            
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "安装超时"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def is_venv_valid(venv_path: str) -> bool:
    """检查虚拟环境是否有效"""
    path = Path(venv_path)
    if not path.exists():
        return False
    
    python_exe = get_python_executable(venv_path)
    return Path(python_exe).exists()


def get_python_executable(venv_path: str) -> str:
    """获取虚拟环境的Python可执行文件路径"""
    venv_path_obj = Path(venv_path)
    if os.name == "nt":  
        return str(venv_path_obj / "Scripts" / "python.exe")
    else:  
        return str(venv_path_obj / "bin" / "python")


def download_models():
    """下载模型文件"""
    print("🤖 模型下载功能已移动到 download_models.py")
    print("💡 请运行: python download_models.py --check")
    return True


def setup_models():
    """Download all models."""
    print("🤖 模型下载功能已移动到 download_models.py")
    print("💡 请运行以下命令:")
    print("  python download_models.py --setup-dirs    # 仅创建目录")
    print("  python download_models.py --check         # 检查模型状态")
    print("  python download_models.py --engine all    # 下载所有模型")

    try:
        from download_models import setup_model_directories
        setup_model_directories()
        print("✅ 目录结构创建完成")
    except ImportError:
        print("❌ download_models.py 未找到，请确保文件存在")
        return False
    
    return True

if __name__ == "__main__":
    setup_models()