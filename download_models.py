
"""
模型下载管理器 - 独立处理所有TTS模型下载
遵循单一职责原则，与虚拟环境管理完全分离
"""
import os
import subprocess
import sys
from pathlib import Path
import requests
import hashlib
from tqdm import tqdm
import modelscope
import concurrent.futures
import time
import logging
from dataclasses import dataclass
from typing import List

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"



@dataclass
class ModelConfig:
    """模型配置类 - 统一管理所有模型配置"""
    key: str
    name: str
    repo: str
    files: List[str]
    
    @property
    def target_dir(self) -> Path:
        return MODELS_DIR / self.key
    
    def get_download_command(self) -> list:
        """返回完整的下载命令"""
        target_dir_str = str(self.target_dir).replace('\\', '/')
        return [
            sys.executable,
            "-c",
            f"from modelscope import snapshot_download; snapshot_download('{self.repo}', local_dir='{target_dir_str}')"
        ]

MODEL_CONFIGS = {
    "xtts_v2": ModelConfig(
        key="xtts_v2",
        name="XTTS-v2",
        repo="AI-ModelScope/XTTS-v2",
        files=[
            "config.json",
            "dvae.pth",
            "hash.md5",
            "mel_stats.pth",
            "model.pth",
            "README.md",
            "samples/",
            "vocab.json",
            "LICENSE.txt"
        ]
    ),
    "chattts": ModelConfig(
        key="chattts",
        name="ChatTTS",
        repo="AI-ModelScope/ChatTTS",
        files=[
            "asset/",
            "config/",
            "configuration.json",
            "README.md"
        ]
    ),
    "kokoro": ModelConfig(
        key="kokoro",
        name="Kokoro",
        repo="AI-ModelScope/Kokoro-82M-v1.1-zh",
        files=[
            "config.json",
            "kokoro-v1_1-zh.pth",
            "README.md",
            "samples/",
            "version.txt",
            "voices/",
            "configuration.json"
        ]
    ),
    # CosyVoice models - 基于ModelScope实际结构
    "cosyvoice2-0.5b": ModelConfig(
        key="cosyvoice2-0.5b",
        name="CosyVoice2-0.5B",
        repo="iic/CosyVoice2-0.5B",
        files=[
            "campplus.onnx",
            "cosyvoice2.yaml",
            "configuration.json",
            "flow.pt",
            "hift.pt",
            "llm.pt",
            "README.md",
            "speech_tokenizer_v2.onnx",
            "spk2info.pt",
            "asset/",
            "CosyVoice-BlankEN/"
        ]
    ),
    "cosyvoice-300m": ModelConfig(
        key="cosyvoice-300m",
        name="CosyVoice-300M",
        repo="iic/CosyVoice-300M",
        files=[
            "campplus.onnx",
            "configuration.json",
            "cosyvoice.yaml",
            "flow.pt",
            "hift.pt",
            "llm.pt",
            "README.md",
            "speech_tokenizer_v1.onnx",
            "spk2info.pt",
            "asset/"
        ]
    ),
    "cosyvoice-300m-25hz": ModelConfig(
        key="cosyvoice-300m-25hz",
        name="CosyVoice-300M-25Hz",
        repo="iic/CosyVoice-300M-25Hz",
        files=[
            "campplus.onnx",
            "configuration.json",
            "cosyvoice.yaml",
            "flow.pt",
            "hift.pt",
            "llm.pt",
            "README.md",
            "speech_tokenizer_v1.onnx",
            "spk2info.pt",
            "asset/"
        ]
    ),
    "cosyvoice-300m-sft": ModelConfig(
        key="cosyvoice-300m-sft",
        name="CosyVoice-300M-SFT",
        repo="iic/CosyVoice-300M-SFT",
        files=[
            "campplus.onnx",
            "configuration.json",
            "cosyvoice.yaml",
            "flow.pt",
            "hift.pt",
            "llm.pt",
            "README.md",
            "speech_tokenizer_v1.onnx",
            "spk2info.pt",
            "asset/"
        ]
    ),
    "cosyvoice-300m-instruct": ModelConfig(
        key="cosyvoice-300m-instruct",
        name="CosyVoice-300M-Instruct",
        repo="iic/CosyVoice-300M-Instruct",
        files=[
            "campplus.onnx",
            "configuration.json",
            "cosyvoice.yaml",
            "flow.pt",
            "hift.pt",
            "llm.pt",
            "README.md",
            "speech_tokenizer_v1.onnx",
            "spk2info.pt",
            "asset/"
        ]
    ),
    "SenseVoiceSmall": ModelConfig(
        key="SenseVoiceSmall",
        name="SenseVoiceSmall",
        repo="iic/SenseVoiceSmall",
        files=[
            "am.mvn",
            "config.yaml",
            "configuration.json",
            "example.ipynb",
            "model.pt",
            "README.md",
            "resource.zip"
        ]
    ),
    "punc_ct-transformer_zh-cn-common-vocab272727-pytorch": ModelConfig(
        key="punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        name="CT-Transformer Chinese Punctuation Prediction",
        repo="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        files=[
            "config.yaml",
            "configuration.json",
            "model.pt",
            "tokens.json",
            "README.md",
            "example/punc_example.txt",
            "fig/struct.png"
        ]
    )
}

DIRS_TO_CREATE = [config.target_dir for config in MODEL_CONFIGS.values()] + [
    BASE_DIR / "outputs/temp",
    BASE_DIR / "outputs/jobs", 
    BASE_DIR / "speakers"
]

def download_xtts_model():
    """下载 XTTS-v2 模型"""
    return download_model("xtts_v2")

def download_chattts_model():
    """下载 ChatTTS 模型"""
    return download_model("chattts")



def download_kokoro_model():
    """下载 Kokoro 模型"""
    return download_model("kokoro")

def download_cosyvoice2_05b_model():
    """下载 CosyVoice2-0.5B 模型"""
    return download_model("cosyvoice2-0.5b")

def download_cosyvoice_300m_model():
    """下载 CosyVoice-300M 模型"""
    return download_model("cosyvoice-300m")

def download_cosyvoice_300m_25hz_model():
    """下载 CosyVoice-300M-25Hz 模型"""
    return download_model("cosyvoice-300m-25hz")

def download_cosyvoice_300m_sft_model():
    """下载 CosyVoice-300M-SFT 模型"""
    return download_model("cosyvoice-300m-sft")

def download_cosyvoice_300m_instruct_model():
    """下载 CosyVoice-300M-Instruct 模型"""
    return download_model("cosyvoice-300m-instruct")

def download_SenseVoiceSmall_model():
    """下载 SenseVoiceSmall 模型"""
    return download_model("SenseVoiceSmall")

def download_punc_ct_transformer_model():
    """下载 CT-Transformer 中文标点符号预测模型"""
    return download_model("punc_ct-transformer_zh-cn-common-vocab272727-pytorch")

def _run_download_command(cmd: list, timeout: int) -> bool:
    """执行下载命令并处理实时输出"""
    def run_download():
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(f"   {line}", end='')
        
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)
        return True

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_download)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return False

def _handle_download_error(config: ModelConfig, error: Exception, logger) -> None:
    """集中处理下载错误"""
    if isinstance(error, subprocess.CalledProcessError):
        logger.error(f"Failed to download {config.name} model (exit code {error.returncode})")
        print(f"\n❌ Failed to download {config.name} model (exit code {error.returncode})")
    else:
        logger.error(f"Unexpected error while downloading {config.name} model: {error}")
        print(f"\n❌ Unexpected error while downloading {config.name} model: {error}")

def _print_download_progress(config: ModelConfig, attempt: int, max_retries: int) -> None:
    """专门处理进度显示"""
    print(f"🎯 Downloading {config.name} model (attempt {attempt + 1}/{max_retries + 1})...")

def _print_success_message(config: ModelConfig) -> None:
    """专门处理成功消息"""
    print(f"\n✅ {config.name} model downloaded successfully to {config.target_dir}")

def _download_with_retry(config: ModelConfig, max_retries: int, timeout: int, logger) -> bool:
    """带重试机制的下载逻辑"""
    cmd = config.get_download_command()
    
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Downloading {config.name} model (attempt {attempt + 1}/{max_retries + 1})...")
            _print_download_progress(config, attempt, max_retries)
            
            result = _run_download_command(cmd, timeout)
            if result:
                logger.info(f"{config.name} model downloaded successfully to {config.target_dir}")
                _print_success_message(config)
                return True
            else:
                logger.warning(f"Download timeout after {timeout} seconds")
                print(f"\n⚠️  Download timeout after {timeout} seconds")
                
        except subprocess.CalledProcessError as e:
            _handle_download_error(config, e, logger)
            
        except Exception as e:
            _handle_download_error(config, e, logger)

        if attempt < max_retries:
            logger.info(f"Retrying in 5 seconds...")
            print(f"   Retrying in 5 seconds...")
            time.sleep(5)
        else:
            logger.error(f"Failed to download {config.name} model after {max_retries + 1} attempts")
            print(f"\n❌ Failed to download {config.name} model after {max_retries + 1} attempts")
            return False
    
    return False

def _setup_logging() -> logging.Logger:
    """一次性配置日志"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def _ensure_directory(config: ModelConfig) -> None:
    """确保目标目录存在"""
    config.target_dir.mkdir(parents=True, exist_ok=True)

def _check_existing_files(config: ModelConfig) -> bool:
    """检查所有必需文件是否都存在"""
    existing_files = check_existing_files(config.target_dir, config.files)
    all_files_exist = len(existing_files) == len(config.files)
    
    if all_files_exist:
        print(f"✅ {config.name} 模型文件已存在 ({len(existing_files)}/{len(config.files)} 个文件)")
        for file in existing_files:
            print(f"   📄 {file}")
        return True
    elif existing_files:
        print(f"⚠️  {config.name} 模型文件不完整 ({len(existing_files)}/{len(config.files)} 个文件)")
        print("   缺失文件:")
        missing_files = set(config.files) - set(existing_files)
        for file in missing_files:
            print(f"   ❌ {file}")
        print("   现有文件:")
        for file in existing_files:
            print(f"   📄 {file}")
        return False
    else:
        return False

def download_model(model_key: str, max_retries: int = 3, timeout: int = 300) -> bool:
    """下载模型 - 现在只有核心逻辑"""
    config = MODEL_CONFIGS.get(model_key)
    if not config:
        print(f"❌ 未知模型类型: {model_key}")
        return False
    
    _ensure_directory(config)
    
    if _check_existing_files(config):
        return True
    
    logger = _setup_logging()
    print(f"🚀 开始下载 {config.name} 模型...")
    print(f"📁 目标目录: {config.target_dir}")
    print(f"🔗 ModelScope 仓库: {config.repo}")
    
    return _download_with_retry(config, max_retries, timeout, logger)

def check_existing_files(target_dir: Path, required_files: list) -> list:
    """检查目标目录中已存在的文件"""
    existing_files = []
    
    for file_pattern in required_files:
        if file_pattern.endswith('/'):
            dir_name = file_pattern.rstrip('/')
            dir_path = target_dir / dir_name
            if dir_path.exists() and dir_path.is_dir():
                existing_files.append(file_pattern)
        else:
            file_path = target_dir / file_pattern
            if file_path.exists():
                existing_files.append(file_pattern)
    
    return existing_files

def setup_model_directories():
    """创建所有模型所需的目录结构"""
    print("📁 创建模型目录结构...")
    
    for dir_path in DIRS_TO_CREATE:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {dir_path.relative_to(BASE_DIR)}")
    
    print("✅ 目录结构创建完成")

def main():
    """主函数 - 提供命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TTS模型下载管理器")
    parser.add_argument("--engine", choices=list(MODEL_CONFIGS.keys()) + ["all"], 
                       default="all", help="要下载的引擎模型")
    parser.add_argument("--setup-dirs", action="store_true", 
                       help="仅创建目录结构，不下载模型")
    parser.add_argument("--check", action="store_true", 
                       help="检查模型文件状态")
    
    args = parser.parse_args()
    
    if args.setup_dirs:
        setup_model_directories()
        return
    
    if args.check:
        print("🔍 检查模型文件状态:")
        all_models_ready = True
        for model_key, config in MODEL_CONFIGS.items():
            existing_files = check_existing_files(config.target_dir, config.files)
            all_files_exist = len(existing_files) == len(config.files)
            status = "✅" if all_files_exist else "❌"
            print(f"   {status} {config.name}: {len(existing_files)}/{len(config.files)} 个文件")
            if not all_files_exist:
                all_models_ready = False
        
        if all_models_ready:
            print("\n✅ 所有模型文件完整！")
        else:
            print("\n⚠️  部分模型文件不完整")
        return
    
    # 下载模型
    engines_to_download = [args.engine] if args.engine != "all" else list(MODEL_CONFIGS.keys())
    
    print(f"🎯 目标引擎: {', '.join(engines_to_download)}")
    print("-" * 50)
    
    success_count = 0
    for engine in engines_to_download:
        if download_model(engine):
            success_count += 1
        print()
    
    print(f"📊 完成: {success_count}/{len(engines_to_download)} 个模型就绪")
    
    if success_count == len(engines_to_download):
        print("✅ 所有模型都已就绪！")
    else:
        print("⚠️  部分模型下载失败")
        print("💡 请检查网络连接和ModelScope配置")

if __name__ == "__main__":
    main()