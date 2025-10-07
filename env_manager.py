#!/usr/bin/env python3
"""
多引擎虚拟环境管理工具
提供多种方法来创建、安装和管理虚拟环境
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# 基础路径
BASE_DIR = Path(__file__).parent
VENVS_DIR = BASE_DIR / "venvs"
REQUIREMENTS_DIR = BASE_DIR
WHEEL_DIR = BASE_DIR / "wheel"  

# 引擎配置 - 统一数据结构
ENGINE_CONFIGS = {
    "xtts": {"req_file": "requirements-xtts.txt"},
    "kokoro": {"req_file": "requirements-kokoro.txt"},
    "chattts": {"req_file": "requirements-chattts.txt"},
    "cosyvoice": {"req_file": "requirements-cosyvoice.txt"}
}

# 可配置的镜像源
PIP_INDEX_URLS = {
    "tsinghua": "https://pypi.tuna.tsinghua.edu.cn/simple",
    "aliyun": "https://mirrors.aliyun.com/pypi/simple/",
    "default": "https://pypi.org/simple/"
}

# 默认镜像源
DEFAULT_PIP_INDEX = "tsinghua"

def get_python_executable(venv_path: Path) -> Path:
    """获取虚拟环境的Python可执行文件路径"""
    if os.name == "nt":  # Windows
        return venv_path / "Scripts" / "python.exe"
    else:  # Unix-like
        return venv_path / "bin" / "python"

def create_venv(engine: str) -> Path:
    """创建虚拟环境"""
    venv_path = VENVS_DIR / engine
    print(f"🔧 为 {engine} 创建虚拟环境...")
    subprocess.check_call([sys.executable, "-m", "venv", str(venv_path)])
    return venv_path

def upgrade_pip_in_venv(venv_path: Path, pip_index: str = DEFAULT_PIP_INDEX):
    """升级虚拟环境中的pip"""
    python_exe = get_python_executable(venv_path)
    pip_url = PIP_INDEX_URLS.get(pip_index, PIP_INDEX_URLS[DEFAULT_PIP_INDEX])
    subprocess.check_call([str(python_exe), "-m", "pip", "install", "--upgrade", "pip", "-i", pip_url])

def install_local_pynini(venv_path: Path):
    """安装本地的pynini wheel包（仅用于cosyvoice）"""
    python_exe = get_python_executable(venv_path)
    pynini_wheel = WHEEL_DIR / "pynini-2.1.6.post1-cp312-cp312-win_amd64.whl"
    
    if not pynini_wheel.exists():
        print(f"⚠️  本地pynini wheel包未找到: {pynini_wheel}")
        return False
    
    print(f"🔧 从本地wheel安装pynini: {pynini_wheel}")
    try:
        subprocess.check_call([str(python_exe), "-m", "pip", "install", str(pynini_wheel)])
        return True
    except subprocess.CalledProcessError:
        print(f"⚠️  pynini wheel包安装失败，尝试跳过pynini安装")
        print(f"💡  提示: pynini可能需要从源码编译，但当前环境缺少编译工具")
        return False

def install_dependencies(venv_path: Path, requirements_file: Path, pip_index: str = DEFAULT_PIP_INDEX, engine: str = ""):
    """安装依赖包，支持多镜像源自动切换"""
    python_exe = get_python_executable(venv_path)
    
    # 镜像源优先级列表（按可靠性排序）
    mirror_fallback_order = ["tsinghua", "aliyun", "default"]
    
    # 所有引擎都使用多镜像源切换逻辑
    print(f"🔧 为 {engine} 安装依赖...")
    
    # 尝试多个镜像源
    for i, mirror_key in enumerate(mirror_fallback_order):
        pip_url = PIP_INDEX_URLS[mirror_key]
        mirror_name = "清华源" if mirror_key == "tsinghua" else "阿里云源" if mirror_key == "aliyun" else "官方源"
        
        try:
            print(f"🌐 尝试使用{mirror_name} ({pip_url})...")
            subprocess.check_call([str(python_exe), "-m", "pip", "install", "-r", str(requirements_file), "-i", pip_url])
            print(f"✅ {mirror_name} 安装成功！")
            break  # 成功则退出循环
        except subprocess.CalledProcessError as e:
            if i < len(mirror_fallback_order) - 1:  # 不是最后一个源
                print(f"⚠️  {mirror_name} 安装失败，尝试下一个镜像源...")
            else:  # 最后一个源也失败
                print(f"❌ 所有镜像源安装失败")
                
    # 只有cosyvoice引擎才需要安装pynini，且只从本地安装
    if engine == "cosyvoice":
        print(f"🔧 为 {engine} 安装pynini（本地wheel包）...")
        pynini_installed = install_local_pynini(venv_path)
        if not pynini_installed:
            print(f"💡 提示: pynini本地安装失败")
            print(f"   1. 确保wheel\\pynini-2.1.6.post1-cp312-cp312-win_amd64.whl文件存在")
            print(f"   2. 考虑使用conda: conda install -c conda-forge pynini")
            print(f"   3. 安装Visual Studio Build Tools 2022")
    else:
        # 其他引擎不需要pynini
        print(f"✅ {engine} 引擎不需要安装pynini")

def dry_run_requirements(requirements_file: Path):
    """干运行模式：显示将要安装的包"""
    with open(requirements_file, 'r', encoding='utf-8') as f:
        packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    for i, package in enumerate(packages, 1):
        print(f"  {i:2d}. {package}")

def install_engine(engine: str, method: str = "venv", pip_index: str = DEFAULT_PIP_INDEX) -> bool:
    """为指定引擎安装依赖 - 统一安装方法"""
    if engine not in ENGINE_CONFIGS:
        print(f"❌ 未知引擎: {engine}")
        return False
    
    config = ENGINE_CONFIGS[engine]
    requirements_file = REQUIREMENTS_DIR / config["req_file"]
    
    if not requirements_file.exists():
        print(f"❌ 需求文件不存在: {requirements_file}")
        return False
    
    print(f"🚀 开始为 {engine} 安装依赖")
    print(f"📁 需求文件: {requirements_file}")
    
    try:
        if method == "dry-run":
            print(f"📋 {engine} 将要安装的依赖包:")
            dry_run_requirements(requirements_file)
            return True
        

        
        # 使用venv
        venv_path = create_venv(engine)
        
        if method == "venv-upgrade":
            upgrade_pip_in_venv(venv_path, pip_index)
        
        install_dependencies(venv_path, requirements_file, pip_index, engine)
        print(f"✅ {engine} 虚拟环境创建完成{'（已升级pip）' if method == 'venv-upgrade' else ''}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多引擎TTS虚拟环境管理工具")
    parser.add_argument("--engine", default="all", help="指定引擎 (xtts, kokoro, chattts, cosyvoice, all)")
    parser.add_argument("--method", choices=["venv", "venv-upgrade", "dry-run"], 
                       default="venv-upgrade", help="安装方法: venv(基础), venv-upgrade(升级pip), dry-run(仅显示)")
    parser.add_argument("--mirror", choices=list(PIP_INDEX_URLS.keys()), default=DEFAULT_PIP_INDEX,
                       help=f"镜像源选择: {', '.join(PIP_INDEX_URLS.keys())} (默认: {DEFAULT_PIP_INDEX})")
    parser.add_argument("--check", action="store_true", help="检查环境状态")
    parser.add_argument("--list", action="store_true", help="列出所有支持的引擎")
    
    args = parser.parse_args()
    
    if args.list:
        print("📋 支持的引擎:")
        for engine in ENGINE_CONFIGS.keys():
            print(f"  - {engine}")
        return
    
    if args.check:
        engines_to_check = list(ENGINE_CONFIGS.keys()) if args.engine == "all" else [args.engine]
        print("🔍 检查虚拟环境状态:")
        for engine in engines_to_check:
            venv_path = VENVS_DIR / engine
            python_exe = get_python_executable(venv_path)
            
            if venv_path.exists() and python_exe.exists():
                print(f"✅ {engine}: 虚拟环境正常 ({venv_path})")
            else:
                print(f"❌ {engine}: 虚拟环境缺失或损坏")
                
            # 检查需求文件
            req_file = REQUIREMENTS_DIR / ENGINE_CONFIGS[engine]["req_file"]
            if req_file.exists():
                print(f"   📄 需求文件: {req_file.name} (存在)")
            else:
                print(f"   ❌ 需求文件: {req_file.name} (缺失)")
        return
    
    # 安装依赖
    engines_to_install = list(ENGINE_CONFIGS.keys()) if args.engine == "all" else [args.engine]
    
    print(f"🎯 目标引擎: {', '.join(engines_to_install)}")
    print(f"🔧 使用方法: {args.method}")
    print(f"🌐 镜像源: {args.mirror} ({PIP_INDEX_URLS[args.mirror]})")
    print("-" * 50)
    
    success_count = 0
    for engine in engines_to_install:
        if install_engine(engine, args.method, args.mirror):
            success_count += 1
        print()
    
    print(f"📊 安装结果: {success_count}/{len(engines_to_install)} 成功")
    
    if success_count > 0:
        print("\n🚀 启动建议:")
        print("1. python main.py (启动主程序)")
        print("2. python setup.py (安装系统依赖)")
    else:
        print("❌ 所有安装都失败了，请检查网络连接或需求文件")

if __name__ == "__main__":
    main()