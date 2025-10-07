
"""
Main orchestrator that coordinates TTS workers, processes ebooks, and manages audio generation.
"""
import asyncio
import logging
import sys
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import gradio as gr
import os
import torch
from pydub import AudioSegment
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cosyvoice_venv_path = os.path.join(project_root, "venvs", "cosyvoice", "Lib", "site-packages")
if os.path.exists(cosyvoice_venv_path):
    sys.path.insert(0, cosyvoice_venv_path)
    logger.info(f"Added {cosyvoice_venv_path} to sys.path for funasr loading")


try:
    from funasr import AutoModel
    ASR_AVAILABLE = True
    logger.info("Successfully imported funasr.AutoModel")
except Exception as e:
    ASR_AVAILABLE = False
    logger.warning(f"Failed to import funasr.AutoModel: {e}")
    AutoModel = None

from .worker_manager import WorkerManager
from .ebook_processor import EbookProcessor
from .audio_processor import AudioProcessor
from .job_manager import JobManager

class PathConfig:
    """Centralized path configuration."""
    
    BASE_DIR = Path(__file__).parent.parent
    SPEAKERS_DIR = BASE_DIR / "speakers"
    MODELS_DIR = BASE_DIR / "models"
    VENVS_DIR = BASE_DIR / "venvs"
    OUTPUTS_DIR = BASE_DIR / "outputs"
    
    @classmethod
    def get_model_path(cls, model_id: str) -> Path:
        return cls.MODELS_DIR / model_id
    
    @classmethod
    def get_venv_path(cls, model_id: str) -> Path:
        return cls.VENVS_DIR / model_id

class ModelStrategy(ABC):
    """Abstract strategy for model parameter handling."""
    
    @abstractmethod
    def extract_params(self, ui_inputs: dict) -> dict:
        """Extract model-specific parameters from UI inputs."""
        pass
    
    @abstractmethod
    def get_param_fields(self) -> list:
        """Return list of parameter field names needed by this model."""
        pass

class ChatTTSStrategy(ModelStrategy):
    PARAM_FIELDS = ["chattts_speaker", "chattts_seed", "chattts_temperature", "chattts_top_p", "chattts_top_k"]
    
    def extract_params(self, ui_inputs: dict) -> dict:
        return {
            "speaker": ui_inputs["chattts_speaker"],
            "seed": int(ui_inputs["chattts_seed"]),
            "temperature": ui_inputs["chattts_temperature"],
            "top_p": ui_inputs["chattts_top_p"],
            "top_k": int(ui_inputs["chattts_top_k"])
        }
    
    def get_param_fields(self) -> list:
        return self.PARAM_FIELDS

class KokoroStrategy(ModelStrategy):
    PARAM_FIELDS = ["kokoro_speaker", "kokoro_speed", "kokoro_use_gpu"]
    
    def extract_params(self, ui_inputs: dict) -> dict:
        return {
            "speaker": ui_inputs["kokoro_speaker"],
            "speed": ui_inputs["kokoro_speed"],
            "use_gpu": ui_inputs["kokoro_use_gpu"]
        }
    
    def get_param_fields(self) -> list:
        return self.PARAM_FIELDS

class XTTSStrategy(ModelStrategy):
    PARAM_FIELDS = ["xtts_speaker_wav", "xtts_saved_speaker", "xtts_seed", "xtts_temperature", 
                "xtts_top_p", "xtts_language", "xtts_speed", "xtts_repetition_penalty", 
                "xtts_top_k", "xtts_length_penalty", "xtts_enable_text_splitting"]
    
    def extract_params(self, ui_inputs: dict) -> dict:
        speaker_wav = self._get_speaker_wav(ui_inputs)
        return {
            "speaker_wav": speaker_wav,
            "seed": int(ui_inputs["xtts_seed"]),
            "temperature": ui_inputs["xtts_temperature"],
            "top_p": ui_inputs["xtts_top_p"],
            "language": ui_inputs["xtts_language"],
            "speed": ui_inputs["xtts_speed"],
            "repetition_penalty": ui_inputs["xtts_repetition_penalty"],
            "top_k": int(ui_inputs["xtts_top_k"]),
            "length_penalty": ui_inputs["xtts_length_penalty"],
            "enable_text_splitting": ui_inputs["xtts_enable_text_splitting"]
        }
    
    def get_param_fields(self) -> list:
        return self.PARAM_FIELDS
    
    def _get_speaker_wav(self, ui_inputs: dict) -> str:
        if ui_inputs["xtts_speaker_wav"]:
            return ui_inputs["xtts_speaker_wav"].name
        elif ui_inputs["xtts_saved_speaker"] and ui_inputs["xtts_saved_speaker"] != "Upload your own WAV file":
            if ui_inputs["xtts_saved_speaker"].startswith("Custom: "):
                speaker_name = ui_inputs["xtts_saved_speaker"][8:]
                return str(PathConfig.SPEAKERS_DIR / f"{speaker_name}.wav")
            elif ui_inputs["xtts_saved_speaker"].startswith("Sample: "):
                sample_name = ui_inputs["xtts_saved_speaker"][8:]
                return str(PathConfig.MODELS_DIR / "xtts_v2" / "samples" / f"{sample_name}.wav")
            else:
                return ui_inputs["xtts_saved_speaker"]
        return None

class CosyVoiceStrategy(ModelStrategy):
    PARAM_FIELDS = [
        "cosyvoice_mode",
        "cosyvoice_speaker",
        "cosyvoice_instruct_text",
        "cosyvoice_prompt_wav_upload", 
        "cosyvoice_prompt_wav_record",
        "cosyvoice_speed",
        "cosyvoice_seed",
        "cosyvoice_stream",
        "cosyvoice_model_dropdown"
    ]
    
    MODE_ALIASES = {
        "3s极速复刻": "3s极速复刻",
        "zero_shot": "3s极速复刻",
        "跨语种复刻": "跨语种复刻",
        "cross_lingual": "跨语种复刻",
        "自然语言控制": "自然语言控制",
        "instruct": "自然语言控制"
    }
    
    def extract_params(self, ui_inputs: dict) -> dict:

        mode = ui_inputs.get("cosyvoice_mode", "预训练音色")

        mode = self.MODE_ALIASES.get(mode, mode)

        params = {
            "prompt_text": ui_inputs.get("cosyvoice_prompt_text", ""),
            "instruct_text": ui_inputs.get("cosyvoice_instruct_text", ""),
            "prompt_wav_upload": ui_inputs.get("cosyvoice_prompt_wav_upload"),
            "prompt_wav_record": ui_inputs.get("cosyvoice_prompt_wav_record"),
            "speed": float(ui_inputs.get("cosyvoice_speed", 1.0)),
            "seed": int(ui_inputs.get("cosyvoice_seed", 0)),
            "stream": bool(ui_inputs.get("cosyvoice_stream", False)),
            "mode": mode,
            "speaker": ui_inputs.get("cosyvoice_speaker", "中文女"),
            "model_dropdown": ui_inputs.get("cosyvoice_model_dropdown", "cosyvoice-300m-sft"),
            "mode_checkbox_group": mode,
            "sft_dropdown": ui_inputs.get("cosyvoice_speaker", "中文女"),
        }

        if mode == "自然语言控制":
            params["instruct_text"] = ui_inputs.get("cosyvoice_instruct_text", "")
        
        logger.info(f"CosyVoiceStrategy extracted params: {params}")
        return params
    
    def get_param_fields(self) -> list:
        return self.PARAM_FIELDS

class ModelStrategyFactory:
    """Factory for creating model strategies."""
    
    _strategies = {
        "chattts": ChatTTSStrategy(),
        "kokoro": KokoroStrategy(),
        "xtts": XTTSStrategy(),
        "cosyvoice": CosyVoiceStrategy()
    }
    
    @classmethod
    def get_strategy(cls, model_type: str) -> ModelStrategy:
        if model_type not in cls._strategies:
            raise ValueError(f"Unknown model type: {model_type}")
        return cls._strategies[model_type]

class ModelConfig:
    """Model configuration with validation and utilities."""
    
    def __init__(self, model_id: str, speakers: list, default_params: dict, venv_path: str = None):
        self.model_id = model_id
        self.speakers = speakers or []
        self.default_params = default_params or {}
        self.venv_path = Path(venv_path) if venv_path else PathConfig.get_venv_path(model_id)
        
        if not self.venv_path.exists():
            logger.warning(f"Virtual environment not found: {self.venv_path}")
    
    def get_worker_config(self) -> dict:
        return {
            "venv_path": str(self.venv_path),
            "model_id": self.model_id,
            "speakers": self.speakers,
            "default_params": self.default_params
        }

# Model configurations
MODEL_CONFIGS = {
    "chattts": ModelConfig(
        "chattts",
        ["default"],
        {
            "speaker": "default",
            "seed": 42,
            "temperature": 0.365,
            "top_p": 0.5,
            "top_k": 50
        },
        "venvs/chattts"
    ),
    "kokoro": ModelConfig(
        "kokoro",
        [],
        {
            "speaker": "zf_001",
            "speed": 1.0,
            "use_gpu": False
        },
        "venvs/kokoro"
    ),
    "xtts": ModelConfig(
        "xtts",
        [],
        {
            "seed": 42,
            "temperature": 0.7,
            "top_p": 0.5,
            "language": "zh-cn",
            "speed": 1.0,
            "repetition_penalty": 2.0,
            "top_k": 50,
            "length_penalty": 1.0,
            "enable_text_splitting": True  
        },
        "venvs/xtts"
    ),
    "cosyvoice": ModelConfig(
        "cosyvoice",
        ["中文女", "中文男", "英文女", "英文男"],
        {
            "prompt_text": "",
            "prompt_wav_upload": None,
            "prompt_wav_record": None,
            "speed": 1.0,
            "seed": 0,
            "stream": False,
            "instruct_text": "",
            "speaker": "中文女",
            "mode": "预训练音色",
            "model_dropdown": "cosyvoice-300m-sft"
        },
        "venvs/cosyvoice"
    )
}

# ==================== 核心区域 (CORE) ====================


class SpeakerDiscoveryService:
    """Centralized speaker discovery service."""
    
    @staticmethod
    def discover_speakers(directory: Path, extension: str, prefix: str = "", fallback: list = None) -> list:
        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            return fallback or []
        
        try:
            files = list(directory.glob(f"*{extension}"))
            if not files:
                logger.warning(f"No {extension} files found in {directory}")
                return fallback or []
            
            speakers = [f"{prefix}{file.stem}" for file in files]
            speakers.sort()
            return speakers
            
        except Exception as e:
            logger.error(f"Failed to discover speakers from {directory}: {e}")
            return fallback or []
    
    @classmethod
    def discover_chattts_speakers(cls) -> list:
        """Discover ChatTTS speakers."""
        speakers = cls.discover_speakers(PathConfig.SPEAKERS_DIR, ".pt")
        speakers.insert(0, "default")
        return speakers
    
    @classmethod
    def discover_kokoro_speakers(cls) -> list:
        """Discover Kokoro speakers with fallback."""
        voices_dir = PathConfig.MODELS_DIR / "kokoro" / "voices"
        fallback = ["zf_001", "zf_002", "zf_003", "zf_004", "zf_005"]
        return cls.discover_speakers(voices_dir, ".pt", fallback=fallback)
    
    @classmethod
    def discover_xtts_speakers(cls) -> list:
        """Discover XTTS speakers from multiple sources."""
        speakers = []
        
        # Custom speakers
        custom = cls.discover_speakers(PathConfig.SPEAKERS_DIR, ".wav", "Custom: ")
        speakers.extend(custom)
        
        # Default samples
        samples_dir = PathConfig.MODELS_DIR / "xtts_v2" / "samples"
        samples = cls.discover_speakers(samples_dir, ".wav", "Sample: ")
        speakers.extend(samples)
        
        # Upload option
        speakers.append("Upload your own WAV file")
        
        return speakers
    
    @classmethod
    def discover_cosyvoice_speakers(cls, model_path: str = None) -> list:
        """Discover CosyVoice speakers from models directory."""
        if model_path:
            spk2info_path = PathConfig.MODELS_DIR / model_path / "spk2info.pt"
            if spk2info_path.exists():
                try:
                    spk2info = torch.load(spk2info_path, map_location="cpu")
                    if isinstance(spk2info, dict):
                        speakers = list(spk2info.keys())
                        logger.info(f"Discovered CosyVoice speakers from {spk2info_path}: {speakers}")
                        return speakers
                except Exception as e:
                    logger.warning(f"Failed to load spk2info.pt from {spk2info_path}: {e}")
        
        cosyvoice_model_dirs = [
            "cosyvoice-300m-sft",
            "cosyvoice-300m-instruct", 
            "cosyvoice-300m",
            "cosyvoice2-0.5b"
        ]
        
        for model_dir in cosyvoice_model_dirs:
            spk2info_path = PathConfig.MODELS_DIR / model_dir / "spk2info.pt"
            if spk2info_path.exists():
                try:
                    spk2info = torch.load(spk2info_path, map_location="cpu")
                    if isinstance(spk2info, dict):
                        speakers = list(spk2info.keys())
                        logger.info(f"Discovered CosyVoice speakers from {spk2info_path}: {speakers}")
                        return speakers
                except Exception as e:
                    logger.warning(f"Failed to load spk2info.pt from {spk2info_path}: {e}")
                    continue
        
        logger.warning("Could not discover CosyVoice speakers from spk2info.pt files, returning empty list")
        return []

class Orchestrator:
    """
    Main orchestrator that coordinates TTS workers, processes ebooks, and manages audio generation.
    """
    
    def __init__(self):
        # Initialize model configs first
        self.chattts_config = MODEL_CONFIGS["chattts"]
        self.kokoro_config = MODEL_CONFIGS["kokoro"]
        self.xtts_config = MODEL_CONFIGS["xtts"]
        self.cosyvoice_config = MODEL_CONFIGS["cosyvoice"]
        
        # Create model configs dictionary for WorkerManager
        model_configs = {
            "chattts": self.chattts_config.get_worker_config(),
            "kokoro": self.kokoro_config.get_worker_config(),
            "xtts": self.xtts_config.get_worker_config(),
            "cosyvoice": self.cosyvoice_config.get_worker_config()
        }
        self.worker_manager = WorkerManager(model_configs)
        self.ebook_processor = EbookProcessor()
        self.audio_processor = AudioProcessor()
        self.job_manager = JobManager()
        
        # Initialize speakers dynamically
        self._discover_all_speakers()
        
        # Initialize CosyVoice models
        self.cosyvoice_models = self._discover_cosyvoice_models()
        if not self.cosyvoice_models:
            self.cosyvoice_models = ["cosyvoice-300m-sft", "cosyvoice-300m-instruct", "cosyvoice-300m", "cosyvoice2-0.5b"]

    def _discover_all_speakers(self):
        """Initialize all speaker lists."""
        # ChatTTS speakers
        self.chattts_config.speakers = SpeakerDiscoveryService.discover_chattts_speakers()
        
        # Kokoro speakers
        self.kokoro_config.speakers = SpeakerDiscoveryService.discover_kokoro_speakers()
        if self.kokoro_config.speakers:
            self.kokoro_config.default_params["speaker"] = self.kokoro_config.speakers[0]
        
        # XTTS speakers
        self.xtts_config.speakers = SpeakerDiscoveryService.discover_xtts_speakers()
        
        # CosyVoice speakers
        discovered_speakers = SpeakerDiscoveryService.discover_cosyvoice_speakers()

        if not discovered_speakers:
            logger.warning("Could not discover CosyVoice speakers, using default speakers")
            self.cosyvoice_config.speakers = MODEL_CONFIGS["cosyvoice"].speakers
        else:
            self.cosyvoice_config.speakers = discovered_speakers

    def _discover_cosyvoice_models(self):
        """Discover CosyVoice models from models directory."""
        models_dir = PathConfig.MODELS_DIR
        cosyvoice_models = []
        if models_dir.exists():
            for item in models_dir.iterdir():
                if item.is_dir() and 'cosyvoice' in item.name.lower():
                    cosyvoice_models.append(item.name)
        return sorted(cosyvoice_models)
    
    def _cosyvoice_prompt_wav_recognition(self, prompt_wav_path):
        """
        处理音频识别，将音频转换为文本
        与app.py中prompt_wav_recognition函数保持一致
        """
        logger.info(f"ASR识别开始，音频路径: {prompt_wav_path}")
        
        if not ASR_AVAILABLE:
            logger.warning("ASR功能不可用，funasr库未安装")
            return "ASR功能不可用，请安装funasr库"
            
        if not prompt_wav_path:
            logger.warning("ASR识别失败，音频路径为空")
            return "请先上传或录制音频文件"
            
        try:
            if not os.path.exists(prompt_wav_path):
                logger.warning(f"ASR识别失败，音频文件不存在: {prompt_wav_path}")
                return "音频文件不存在，请重新上传或录制"

            model_path = os.path.join(PathConfig.MODELS_DIR, "SenseVoiceSmall")
            logger.info(f"加载ASR模型: {model_path}")
            
            if not os.path.exists(model_path):
                logger.warning(f"ASR模型路径不存在: {model_path}")
                return "ASR模型未找到，请检查模型文件"

            asr = AutoModel(model=model_path, disable_update=True,
                            device="cuda" if torch.cuda.is_available() else "cpu")
            res = asr.generate(input=prompt_wav_path, language="auto", use_itn=True)

            text = res[0]["text"].split('|>')[-1] if res and len(res) > 0 and "text" in res[0] else ""

            try:
                del asr
            except Exception as e:
                logger.warning(f"ASR模型清理失败: {e}")
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            if not text.strip():
                logger.warning("ASR识别结果为空")
                return "ASR识别失败，未识别到有效文本"
                
            logger.info(f"ASR识别完成，识别文本: {text}")
            return text.strip()
        except Exception as e:
            logger.error(f"ASR识别失败: {e}", exc_info=True)
            try:
                del asr
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return f"ASR识别失败: {str(e)}"

    def _generate_cosyvoice_audio(self, text, mode, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, 
                                instruct_text, seed, stream, speed, model_dropdown):
        """
        CosyVoice音频生成函数，与app.py中generate_audio函数保持一致
        """
        logger.info(f"CosyVoice音频生成开始: mode={mode}, model={model_dropdown}")

        prompt_wav = prompt_wav_record if prompt_wav_record is not None else prompt_wav_upload
        
        try:

            cosyvoice_params = {
                "mode": mode,
                "sft_dropdown": sft_dropdown,
                "prompt_text": prompt_text,
                "prompt_wav": prompt_wav,
                "instruct_text": instruct_text,
                "seed": int(seed) if seed else random.randint(1, 100000000),
                "stream": bool(stream),
                "speed": speed,
                "model_dropdown": model_dropdown
            }

            async def generate_async():
                audio_file = await self.worker_manager.generate_audio(
                    "cosyvoice",
                    text,
                    cosyvoice_params,
                    "cosyvoice_generate",
                    "wav"
                )
                return audio_file
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                audio_path = loop.run_until_complete(generate_async())
                if audio_path:
                    logger.info(f"CosyVoice音频生成成功: {audio_path}")
                    return audio_path
                else:
                    logger.error("CosyVoice音频生成失败: 返回空路径")
                    return None
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"CosyVoice音频生成异常: {e}", exc_info=True)
            return None

    def run(self):
        """Start the orchestrator with Gradio UI."""
        self._create_gradio_interface().launch(
            server_name="127.0.0.1",
            server_port=7860, 
            share=False
        )
    
    def _create_gradio_interface(self):
        """Create the Gradio UI interface with model-specific tabs."""
        with gr.Blocks(title="Multi-Engine TTS Book Converter", 
                      css="""
                      .seed_button {
                          min-width: 100px;
                          align-self: flex-end;
                          margin-bottom: 8px;
                      }
                      """) as app:
            gr.Markdown("# 📚 Multi-Engine TTS Book Converter")
            gr.Markdown("Convert ebooks to audiobooks using multiple TTS engines")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # File upload
                    ebook_file = gr.File(
                        label="Upload Ebook",
                        file_types=[".epub", ".mobi", ".pdf", ".txt"]
                    )
                
                # Model tabs for parameters
                    with gr.Tabs() as tabs:
                        # ChatTTS Tab
                        with gr.Tab("ChatTTS") as chattts_tab:
                            chattts_speaker = gr.Dropdown(
                                choices=self.chattts_config.speakers,
                                value=self.chattts_config.speakers[0],
                                label="Speaker"
                            )
                            with gr.Accordion("音色管理", open=False):
                                gr.Markdown("根据推理参数配置，自定义音色，提供音色预览、保存、删除功能")
                                speaker_name = gr.Textbox(label="音色名称", placeholder="输入容易记的名称")
                                save_speaker_btn = gr.Button("保存音色")
                                preview_speaker_btn = gr.Button("预览音色")
                                delete_speaker_dropdown = gr.Dropdown(
                                    choices=[s for s in self.chattts_config.speakers if s != "default"],
                                    label="选择要删除的音色"
                                )
                                delete_speaker_btn = gr.Button("删除音色")
                                speaker_save_status = gr.Textbox(label="Status", interactive=False)

                            with gr.Row():
                                chattts_seed = gr.Number(42, label="Seed", precision=0)
                                chattts_seed_button = gr.Button("🎲", elem_classes="seed_button")
                            
                            chattts_temperature = gr.Slider(0.1, 1.0, 0.365, label="Temperature")
                            chattts_top_p = gr.Slider(0.1, 1.0, 0.5, label="Top-P")
                            chattts_top_k = gr.Slider(1, 100, 50, label="Top-K")
                            
                            
                        
                        # Kokoro Tab
                        with gr.Tab("Kokoro") as kokoro_tab:
                            kokoro_speaker = gr.Dropdown(
                                choices=self.kokoro_config.speakers,
                                value=self.kokoro_config.speakers[0] if self.kokoro_config.speakers else "zf_001",
                                label="Speaker"
                            )
                            kokoro_speed = gr.Slider(0.5, 2.0, 1.0, label="Speed")
                            kokoro_use_gpu = gr.Checkbox(value=False, label="Use GPU")
                        
                        # XTTS Tab
                        with gr.Tab("XTTS") as xtts_tab:
                            gr.Markdown("### 音色设置")

                            xtts_speaker_wav = gr.File(label="参考音频", file_types=[".wav"])
                            xtts_saved_speaker = gr.Dropdown(
                                choices=self.xtts_config.speakers,
                                value=self.xtts_config.speakers[0] if self.xtts_config.speakers else "Upload your own WAV file",
                                label="选择音色"
                            )

                            with gr.Accordion("音色管理", open=False):
                                xtts_speaker_name = gr.Textbox(label="音色名称", placeholder="输入容易记的音色名称")
                                xtts_save_speaker_btn = gr.Button("保存音色")
                                xtts_preview_speaker_btn = gr.Button("预览音色", variant="secondary")
                                xtts_delete_speaker_dropdown = gr.Dropdown(
                                     choices=[s for s in self.xtts_config.speakers if s.startswith("Custom: ")],
                                     label="选择要删除的音色"
                                 )
                                xtts_delete_speaker_btn = gr.Button("删除音色")
                                xtts_speaker_status = gr.Textbox(label="状态", interactive=False, max_lines=1)

                            gr.Markdown("### 合成参数")
                            xtts_speed = gr.Slider(0.1, 2.0, 1.0, label="语速")
                            xtts_language = gr.Dropdown(
                                choices=["zh-cn", "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "ja", "hu", "ko"],
                                value="zh-cn",
                                label="语言"
                            )

                            with gr.Row():
                                xtts_seed = gr.Number(42, label="随机种子", precision=0)
                                xtts_seed_button = gr.Button("🎲", elem_classes="seed_button")
                            
                            xtts_temperature = gr.Slider(0.1, 1.0, 0.7, label="温度")
                            xtts_repetition_penalty = gr.Slider(1.0, 10.0, 2.0, label="重复惩罚")
                            xtts_top_p = gr.Slider(0.1, 1.0, 0.5, label="Top-P")
                            xtts_top_k = gr.Slider(1, 100, 50, label="Top-K")
                            xtts_length_penalty = gr.Slider(0.1, 2.0, 1.0, label="长度惩罚")

                            with gr.Accordion("高级参数", open=False):
                                xtts_enable_text_splitting = gr.Checkbox(value=True, label="启用文本分割")

                        # CosyVoice Tab
                        with gr.Tab("CosyVoice") as cosyvoice_tab:
                            gr.Markdown("### 推理模式")
                            
                            with gr.Row():
                                cosyvoice_mode = gr.Radio(
                                    choices=["预训练音色", "3s极速复刻", "跨语种复刻", "自然语言控制"],
                                    value="预训练音色",
                                    label="选择推理模式"
                                )
                                cosyvoice_instruction_text = gr.Text(
                                    label="操作步骤",
                                    value="1. 选择预训练音色\n2. 点击生成音频按钮\n\n适用模型：CosyVoice-300M-SFT, CosyVoice-300M-Instruct\n\n",
                                    interactive=False
                                )

                            cosyvoice_model_dropdown = gr.Dropdown(
                                choices=self.cosyvoice_models,
                                value=self.cosyvoice_models[0] if self.cosyvoice_models else "cosyvoice-300m-sft",
                                label="选择CosyVoice模型"
                            )

                            cosyvoice_speaker = gr.Dropdown(
                                choices=self.cosyvoice_config.speakers,
                                value=self.cosyvoice_config.speakers[0] if self.cosyvoice_config.speakers else "中文女",
                                label="选择预训练音色"
                            )
                            
                            gr.Markdown("### 提示音频")
                            with gr.Row():
                                cosyvoice_prompt_wav_upload = gr.Audio(
                                    sources='upload', 
                                    type='filepath', 
                                    label='选择prompt音频文件（采样率不低于16kHz）'
                                )
                                cosyvoice_prompt_wav_record = gr.Audio(
                                    sources='microphone', 
                                    type='filepath', 
                                    label='录制prompt音频文件'
                                )
                            
                            cosyvoice_prompt_text = gr.Textbox(
                                label="Prompt文本", 
                                lines=2, 
                                placeholder="请输入prompt文本，需与prompt音频内容一致...",
                                visible=True
                            )
                            
                            cosyvoice_instruct_text = gr.Textbox(
                                label="Instruct文本", 
                                lines=2, 
                                placeholder="请输入instruct文本，例如：用四川话说这句话。",
                                visible=False
                            )
                            
                            gr.Markdown("### 合成参数")
                            with gr.Row():
                                with gr.Column():
                                    cosyvoice_speed = gr.Slider(0.5, 2.0, 1.0, label="语速")
                                    cosyvoice_stream = gr.Radio(
                                        choices=[("否", False), ("是", True)], 
                                        value=False, 
                                        label="是否流式推理"
                                    )
                                with gr.Column():
                                    with gr.Row():
                                        cosyvoice_seed = gr.Number(value=0, label="随机种子")
                                        cosyvoice_seed_button = gr.Button(value="🎲", elem_classes="seed_button")

                    
                    # Hidden field to track active tab
                    active_tab = gr.Textbox(value="ChatTTS", visible=False)
                    
                    # Output format selection
                    output_format = gr.Dropdown(
                        choices=["mp3", "wav"],
                        value="mp3",
                        label="Output Format"
                    )
                    
                    # Start button
                    start_btn = gr.Button("Start Conversion", variant="primary")
                
                with gr.Column(scale=3):
                    # Status display
                    status_text = gr.Textbox(
                        label="Status",
                        value="Ready",
                        interactive=False,
                        lines=5
                    )
                    
                    # Preview audio component
                    preview_audio = gr.Audio(
                        label="Generated Audio",
                        visible=True
                    )
            
            # Event handlers
            
            # Random seed buttons
            def generate_random_seed():
                return random.randint(1, 100000000)
            
            chattts_seed_button.click(fn=generate_random_seed, outputs=chattts_seed)
            xtts_seed_button.click(fn=generate_random_seed, outputs=xtts_seed)
            cosyvoice_seed_button.click(fn=generate_random_seed, outputs=cosyvoice_seed)
            
            # CosyVoice mode change handlers
            def change_cosyvoice_instruction(mode):
                instructions = {
                    '预训练音色': '1. 选择预训练音色\n2. 点击生成音频按钮\n\n适用模型：CosyVoice-300M-SFT, CosyVoice-300M-Instruct',
                    '3s极速复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 输入prompt文本\n3. 点击生成音频按钮\n\n适用模型：CosyVoice2-0.5B, CosyVoice-300M, CosyVoice-300M-SFT, CosyVoice-300M-Instruct, CosyVoice-300M-25Hz',
                    '跨语种复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 点击生成音频按钮\n\n适用模型：CosyVoice2-0.5B, CosyVoice-300M, CosyVoice-300M-SFT, CosyVoice-300M-Instruct, CosyVoice-300M-25Hz', 
                    '自然语言控制': '1. 选择预训练音色\n2. 输入instruct文本\n3. 点击生成音频按钮\n\n适用模型：CosyVoice2-0.5B'
                }
                return instructions[mode]
            
            def toggle_cosyvoice_inputs(mode):
                if mode == "3s极速复刻":
                    return (
                        gr.update(visible=True),  
                        gr.update(visible=False),  
                        gr.update(visible=True, interactive=True, choices=self.cosyvoice_config.speakers)
                    )
                elif mode == "自然语言控制":
                    return (
                        gr.update(visible=True),  
                        gr.update(visible=True),  
                        gr.update(visible=True, interactive=True, choices=self.cosyvoice_config.speakers)
                    )
                elif mode == "跨语种复刻":
                    return (
                        gr.update(visible=True), 
                        gr.update(visible=False),  
                        gr.update(visible=True, interactive=True, choices=self.cosyvoice_config.speakers)
                    )
                else:  
                    return (
                        gr.update(visible=False), 
                        gr.update(visible=False),  
                        gr.update(visible=True, interactive=True, choices=self.cosyvoice_config.speakers)
                    )
            
            cosyvoice_mode.change(
                fn=change_cosyvoice_instruction,
                inputs=[cosyvoice_mode],
                outputs=[cosyvoice_instruction_text]
            )
            
            cosyvoice_mode.change(
                fn=toggle_cosyvoice_inputs,
                inputs=[cosyvoice_mode],
                outputs=[cosyvoice_prompt_text, cosyvoice_instruct_text, cosyvoice_speaker]
            )
            
            # Model dropdown change handler
            def update_cosyvoice_speakers(model_name):
                """根据选择的模型更新音色列表"""
                speakers = SpeakerDiscoveryService.discover_cosyvoice_speakers(model_name)
                self.cosyvoice_config.speakers = speakers
                return gr.update(choices=speakers, value=speakers[0] if speakers else None)
            
            cosyvoice_model_dropdown.change(
                fn=update_cosyvoice_speakers,
                inputs=[cosyvoice_model_dropdown],
                outputs=[cosyvoice_speaker]
            )
            
            # ASR recognition with status updates
            def cosyvoice_prompt_wav_recognition_with_status(prompt_wav_path):
                """ASR识别函数，带状态更新"""
                if not prompt_wav_path:
                    return "请先上传或录制音频文件", "Ready"

                yield "正在加载ASR模型...", ""

                result = self._cosyvoice_prompt_wav_recognition(prompt_wav_path)

                if result and not result.startswith("ASR识别失败") and not result.startswith("ASR功能不可用"):
                    yield result, "ASR识别完成: " + (result[:50] + "..." if len(result) > 50 else result)
                else:
                    yield "", result if result else "ASR识别失败"
            
            cosyvoice_prompt_wav_upload.change(
                fn=cosyvoice_prompt_wav_recognition_with_status,
                inputs=[cosyvoice_prompt_wav_upload],
                outputs=[cosyvoice_prompt_text, status_text]
            )
            
            cosyvoice_prompt_wav_record.change(
                fn=cosyvoice_prompt_wav_recognition_with_status,
                inputs=[cosyvoice_prompt_wav_record],
                outputs=[cosyvoice_prompt_text, status_text]
            )
            
           
            # Speaker management functions
            async def save_chattts_speaker(name, speaker, seed, temperature, top_p, top_k):
                try:
                    result = await self.worker_manager.call_worker_method(
                        "chattts", "save_speaker", name, int(seed), temperature, top_p, int(top_k)
                    )
                    self.chattts_config.speakers = SpeakerDiscoveryService.discover_chattts_speakers()
                    delete_choices = [s for s in self.chattts_config.speakers if s != "default"]
                    return result, gr.update(choices=self.chattts_config.speakers), gr.update(choices=delete_choices)
                except Exception as e:
                    logger.error(f"Error saving speaker: {str(e)}", exc_info=True)
                    return f"Error saving speaker: {str(e)}", gr.update(choices=self.chattts_config.speakers), gr.update(choices=[])
            
            async def preview_chattts_speaker(seed, temperature, top_p, top_k):
                try:
                    message, audio_path = await self.worker_manager.call_worker_method(
                        "chattts", "preview_speaker", int(seed), temperature, top_p, int(top_k)
                    )
                    if audio_path:
                        return message, gr.update(value=audio_path, visible=True)
                    else:
                        return message, gr.update(visible=False)
                except Exception as e:
                    logger.error(f"Error previewing speaker: {str(e)}", exc_info=True)
                    return f"Error previewing speaker: {str(e)}", gr.update(visible=False)
            
            
            async def delete_chattts_speaker(speaker_name):
                try:
                    if not speaker_name or speaker_name == "default":
                        return "Please select a valid speaker to delete", gr.update(choices=self.chattts_config.speakers), gr.update(choices=[])

                    result = await self.worker_manager.call_worker_method(
                        "chattts", "delete_speaker", speaker_name
                    )

                    self.chattts_config.speakers = SpeakerDiscoveryService.discover_chattts_speakers()
                    delete_choices = [s for s in self.chattts_config.speakers if s != "default"]
                    return result, gr.update(choices=self.chattts_config.speakers), gr.update(choices=delete_choices)
                except Exception as e:
                    logger.error(f"Error deleting speaker: {str(e)}", exc_info=True)
                    return f"Error deleting speaker: {str(e)}", gr.update(choices=self.chattts_config.speakers), gr.update(choices=[])
            
            # XTTS Speaker management functions
            def save_xtts_speaker(name, speaker_wav):
                if not name:
                    return "Please enter a speaker name", gr.update(choices=self.xtts_config.speakers), gr.update(choices=[])
                
                if not name.replace("_", "").replace("-", "").replace(" ", "").isalnum():
                    return "Speaker name can only contain letters, numbers, underscores, hyphens, and spaces", gr.update(choices=self.xtts_config.speakers), gr.update(choices=[])
                
                if speaker_wav is None:
                    return "Please upload a WAV file first", gr.update(choices=self.xtts_config.speakers), gr.update(choices=[])
                
                try:
                    speakers_dir = PathConfig.SPEAKERS_DIR
                    speakers_dir.mkdir(exist_ok=True, parents=True)
                    
                    import shutil
                    destination_path = speakers_dir / f"{name}.wav"
                    
                    if destination_path.exists():
                        return f"Speaker '{name}' already exists. Please choose a different name.", gr.update(choices=self.xtts_config.speakers), gr.update(choices=[])
                    
                    shutil.copy2(speaker_wav.name, destination_path)
                    self.xtts_config.speakers = SpeakerDiscoveryService.discover_xtts_speakers()
                    delete_choices = [s for s in self.xtts_config.speakers if s.startswith("Custom: ")]
                    return f"Speaker '{name}' saved successfully!", gr.update(choices=self.xtts_config.speakers), gr.update(choices=delete_choices)
                except Exception as e:
                    logger.error(f"Error saving XTTS speaker: {str(e)}", exc_info=True)
                    return f"Error saving speaker: {str(e)}", gr.update(choices=self.xtts_config.speakers), gr.update(choices=[])
            
            def delete_xtts_speaker(speaker_name):
                if not speaker_name or not speaker_name.startswith("Custom: "):
                    return "Please select a valid custom speaker to delete", gr.update(choices=self.xtts_config.speakers), gr.update(choices=[])
                
                try:
                    actual_name = speaker_name[8:]  
                    filepath = PathConfig.SPEAKERS_DIR / f"{actual_name}.wav"
                    
                    if not filepath.exists():
                        return f"Speaker file '{actual_name}.wav' not found", gr.update(choices=self.xtts_config.speakers), gr.update(choices=[])
                    
                    filepath.unlink()
                    self.xtts_config.speakers = SpeakerDiscoveryService.discover_xtts_speakers()
                    delete_choices = [s for s in self.xtts_config.speakers if s.startswith("Custom: ")]
                    return f"Speaker '{actual_name}' deleted successfully!", gr.update(choices=self.xtts_config.speakers), gr.update(choices=delete_choices)
                except Exception as e:
                    logger.error(f"Error deleting XTTS speaker: {str(e)}", exc_info=True)
                    return f"Error deleting speaker: {str(e)}", gr.update(choices=self.xtts_config.speakers), gr.update(choices=[])
            
            def preview_xtts_speaker(speaker_wav, saved_speaker, language, speed):
                try:
                    speaker_path = None
                    if speaker_wav is not None:
                        speaker_path = speaker_wav.name
                    elif saved_speaker and saved_speaker != "Upload your own WAV file":
                        if saved_speaker.startswith("Custom: "):
                            actual_name = saved_speaker[8:]
                            speaker_path = str(PathConfig.SPEAKERS_DIR / f"{actual_name}.wav")
                        elif saved_speaker.startswith("Sample: "):
                            sample_name = saved_speaker[8:]
                            speaker_path = str(PathConfig.MODELS_DIR / "xtts_v2" / "samples" / f"{sample_name}.wav")
                        else:
                            speaker_path = saved_speaker
                    
                    if not speaker_path:
                        return None, "请上传参考音频或选择已保存的音色"
                    
                    if not Path(speaker_path).exists():
                        return None, f"音色文件不存在: {speaker_path}"
                    
                    test_texts = {
                        "zh-cn": "你好，这是一个音色预览测试，用于测试克隆音色的效果。",
                        "en": "Hello, this is a voice preview test to check the cloned voice quality.",
                    }
                    
                    test_text = test_texts.get(language, test_texts["en"])
                    
                    preview_params = {
                        "speaker_wav": speaker_path,
                        "language": language,
                        "speed": speed,
                        "seed": 42,
                        "temperature": 0.7,
                        "top_p": 0.5,
                        "repetition_penalty": 2.0,
                        "top_k": 50,
                        "length_penalty": 1.0,
                        "enable_text_splitting": False
                    }
                    
                    import asyncio
                    
                    async def generate_preview():
                        audio_file = await self.worker_manager.generate_audio(
                            "xtts",
                            test_text,
                            preview_params,
                            "xtts_preview",
                            "wav"
                        )
                        return audio_file
                    
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        audio_path = loop.run_until_complete(generate_preview())
                        if audio_path:
                            return audio_path, f"音色预览已生成: {Path(speaker_path).name}"
                        else:
                            return None, "预览生成失败"
                    finally:
                        loop.close()
                        
                except Exception as e:
                    logger.error(f"Error previewing XTTS speaker: {str(e)}", exc_info=True)
                    return None, f"预览失败: {str(e)}"
            
            # Bind event handlers
            save_speaker_btn.click(
                fn=save_chattts_speaker,
                inputs=[speaker_name, chattts_speaker, chattts_seed, chattts_temperature, chattts_top_p, chattts_top_k],
                outputs=[speaker_save_status, chattts_speaker, delete_speaker_dropdown]
            )
            
            preview_speaker_btn.click(
                fn=preview_chattts_speaker,
                inputs=[chattts_seed, chattts_temperature, chattts_top_p, chattts_top_k],
                outputs=[speaker_save_status, preview_audio]
            )
            
            
            delete_speaker_btn.click(
                fn=delete_chattts_speaker,
                inputs=delete_speaker_dropdown,
                outputs=[speaker_save_status, chattts_speaker, delete_speaker_dropdown]
            )
            
            # XTTS event handlers
            xtts_save_speaker_btn.click(
                fn=save_xtts_speaker,
                inputs=[xtts_speaker_name, xtts_speaker_wav],
                outputs=[xtts_speaker_status, xtts_saved_speaker, xtts_delete_speaker_dropdown]
            )
            
            xtts_delete_speaker_btn.click(
                fn=delete_xtts_speaker,
                inputs=xtts_delete_speaker_dropdown,
                outputs=[xtts_speaker_status, xtts_saved_speaker, xtts_delete_speaker_dropdown]
            )
            
            xtts_preview_speaker_btn.click(
                fn=preview_xtts_speaker,
                inputs=[xtts_speaker_wav, xtts_saved_speaker, xtts_language, xtts_speed],
                outputs=[preview_audio, xtts_speaker_status]
            )
            
            # Main processing
            start_btn.click(
                fn=self._process_ebook_new,
                inputs=[
                    ebook_file,
                    active_tab,
                    output_format,
                    chattts_speaker, chattts_seed, chattts_temperature, chattts_top_p, chattts_top_k,
                    kokoro_speaker, kokoro_speed, kokoro_use_gpu,
                    xtts_speaker_wav, xtts_saved_speaker, xtts_seed, xtts_temperature, xtts_top_p, xtts_language, xtts_speed, xtts_repetition_penalty, xtts_top_k, xtts_length_penalty, xtts_enable_text_splitting,
                    cosyvoice_mode, cosyvoice_speaker, cosyvoice_prompt_text, cosyvoice_prompt_wav_upload, cosyvoice_prompt_wav_record, cosyvoice_instruct_text, cosyvoice_speed, cosyvoice_seed, cosyvoice_stream, cosyvoice_model_dropdown,
                ],
                outputs=[status_text, preview_audio]
            )
            
            # Update active tab when tab changes
            chattts_tab.select(lambda: "ChatTTS", None, active_tab)
            kokoro_tab.select(lambda: "Kokoro", None, active_tab)
            xtts_tab.select(lambda: "XTTS", None, active_tab)
            cosyvoice_tab.select(lambda: "CosyVoice", None, active_tab)
        
        return app
    
    async def _process_ebook_new(self, ebook_file, active_tab, output_format,
                               chattts_speaker, chattts_seed, chattts_temperature, chattts_top_p, chattts_top_k,
                               kokoro_speaker, kokoro_speed, kokoro_use_gpu,
                               xtts_speaker_wav, xtts_saved_speaker, xtts_seed, xtts_temperature, xtts_top_p, xtts_language, xtts_speed, xtts_repetition_penalty, xtts_top_k, xtts_length_penalty, xtts_enable_text_splitting,
                               cosyvoice_mode, cosyvoice_speaker, cosyvoice_prompt_text, cosyvoice_prompt_wav_upload, cosyvoice_prompt_wav_record, cosyvoice_instruct_text, cosyvoice_speed, cosyvoice_seed, cosyvoice_stream, cosyvoice_model_dropdown,
                               progress=gr.Progress()):
        """Process ebook with selected model and parameters."""
        try:
            if not ebook_file:
                return "Please upload an ebook file.", None
            
            # Use strategy pattern to extract model parameters
            model_id = active_tab.lower()
            strategy = ModelStrategyFactory.get_strategy(model_id)
            
            # Build parameter dictionary from all UI inputs
            ui_inputs = {
                "chattts_speaker": chattts_speaker,
                "chattts_seed": chattts_seed,
                "chattts_temperature": chattts_temperature,
                "chattts_top_p": chattts_top_p,
                "chattts_top_k": chattts_top_k,
                "kokoro_speaker": kokoro_speaker,
                "kokoro_speed": kokoro_speed,
                "kokoro_use_gpu": kokoro_use_gpu,
                "xtts_speaker_wav": xtts_speaker_wav,
                "xtts_saved_speaker": xtts_saved_speaker,
                "xtts_seed": xtts_seed,
                "xtts_temperature": xtts_temperature,
                "xtts_top_p": xtts_top_p,
                "xtts_language": xtts_language,
                "xtts_speed": xtts_speed,
                "xtts_repetition_penalty": xtts_repetition_penalty,
                "xtts_top_k": xtts_top_k,
                "xtts_length_penalty": xtts_length_penalty,
                "xtts_enable_text_splitting": xtts_enable_text_splitting,
                "cosyvoice_mode": cosyvoice_mode,
                "cosyvoice_prompt_text": cosyvoice_prompt_text,
                "cosyvoice_prompt_wav_upload": cosyvoice_prompt_wav_upload,
                "cosyvoice_prompt_wav_record": cosyvoice_prompt_wav_record,
                "cosyvoice_instruct_text": cosyvoice_instruct_text,
                "cosyvoice_speed": cosyvoice_speed,
                "cosyvoice_seed": cosyvoice_seed,
                "cosyvoice_stream": cosyvoice_stream,
                "cosyvoice_model_dropdown": cosyvoice_model_dropdown,
                "cosyvoice_speaker": cosyvoice_speaker
            }
            
            model_params = strategy.extract_params(ui_inputs)
            
            progress(0, f"Starting {model_id} processing...")
            return await self._process_ebook_single(ebook_file, model_id, model_params, output_format, progress)
            
        except Exception as e:
            logger.error(f"Error processing ebook: {str(e)}")
            return f"Error: {str(e)}", None
    
    async def _process_ebook_single(self, ebook_file, model_id: str, model_params: Dict, output_format: str, progress=gr.Progress()):
        """使用单个模型处理电子书"""
        try:
            job_id = await self.job_manager.create_job(ebook_file, [model_id], model_params, output_format)
            
            logger.info(f"[DEBUG] Starting to process ebook with model {model_id}")
            logger.info(f"[DEBUG] Model parameters: {model_params}")
            
            # Process ebook
            chapters = await self.ebook_processor.process_ebook(ebook_file)
            logger.info(f"Processing {len(chapters)} chapters")
            
            # Update job information
            await self.job_manager.update_job_chapters(job_id, chapters)
            
            all_fragments = []
            total_fragments = 0

            for chapter in chapters:
                logger.info(f"Processing chapter: {chapter['title']}")
                logger.info(f"Chapter text length: {len(chapter['text'])}")
                logger.info(f"Chapter text preview: {chapter['text'][:200]}...")

                fragments = self.ebook_processor.split_chapter(chapter["text"], model_id)
                logger.info(f"Chapter split into {len(fragments)} fragments")

                for i, fragment in enumerate(fragments):
                    logger.info(f"Fragment {i+1}: length={len(fragment)}")
                    logger.info(f"Fragment {i+1} preview: {fragment[:100]}...")
                    fragment_info = {
                        "text": fragment,
                        "chapter": chapter["title"],
                        "chapter_index": chapters.index(chapter),
                        "fragment_index": i,
                        "total_fragments_in_chapter": len(fragments),
                        "char_count": len(fragment)
                    }
                    all_fragments.append(fragment_info)
                
                total_fragments += len(fragments)

            unique_fragments = []
            seen_texts = set()
            
            for fragment in all_fragments:
                text_key = fragment["text"][:100].strip()  
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    unique_fragments.append(fragment)
            
            logger.info(f"Optimized to {len(unique_fragments)} fragments (originally {total_fragments})")
            total_fragments = len(unique_fragments)
            
            processed_fragments = 0
            chapter_audio_files = {}

            progress(0, f"Starting {model_id} processing...")
            logger.info(f"Starting processing with {total_fragments} total fragments")
            
            # Process fragments with checkpoint and retry mechanism
            await self._process_fragments_with_checkpoint(
                job_id, unique_fragments, model_id, model_params, output_format,
                chapter_audio_files, progress, total_fragments
            )
            
            # Merge chapter audio files
            all_audio_files = []
            chapter_output_files = []
            
            for chapter_idx in sorted(chapter_audio_files.keys()):
                chapter_title = chapters[chapter_idx]["title"]
                audio_files = chapter_audio_files[chapter_idx]
                
                if audio_files:
                    merged_file = await self.audio_processor.merge_chapter_audio(
                        audio_files, chapter_title, job_id, output_format
                    )
                    all_audio_files.append(merged_file)
                    chapter_output_files.append(merged_file)
                    logger.info(f"Merged chapter {chapter_title} audio: {merged_file}")
            
            # Create final audiobook file
            if all_audio_files:
                book_title = Path(ebook_file).stem
                final_file = await self.audio_processor.create_final_audiobook(
                    all_audio_files, book_title, job_id, output_format
                )
                
                output_files = chapter_output_files + [final_file]
            
                # Clean up temporary files
                for chapter_idx in sorted(chapter_audio_files.keys()):
                    audio_files = chapter_audio_files[chapter_idx]
                    for audio_file in audio_files:
                        try:
                            Path(audio_file).unlink(missing_ok=True)
                        except Exception as e:
                            logger.warning(f"Cannot delete temporary file {audio_file}: {e}")
                
                await self.job_manager.complete_job(job_id, output_files)
                
                final_audio = AudioSegment.from_file(final_file)
                samples = np.array(final_audio.get_array_of_samples())
                if final_audio.channels == 2:
                    samples = samples.reshape((-1, 2))
                
                progress(1.0, f"[{model_id.upper()}] Processing completed! Audio saved to: {final_file}")
                logger.info(f"[{model_id.upper()}] Processing completed! Audio saved to: {final_file}")
                return f"[{model_id.upper()}] Processing completed! Audio saved to: {final_file}", (final_audio.frame_rate, samples)
            else:
                await self.job_manager.fail_job(job_id, "No audio files generated")
                return "Error: No audio files generated", None
                
        except Exception as e:

            try:
                error_msg = str(e).encode('utf-8', errors='ignore').decode('utf-8')
            except Exception:
                error_msg = repr(e)
            
            logger.error(f"Error processing ebook with {model_id}: {error_msg}")
            if 'job_id' in locals():
                await self.job_manager.fail_job(job_id, error_msg)
            return f"[{model_id.upper()}] Error: {error_msg}", None

    async def _process_fragments_with_checkpoint(
        self, job_id: str, fragments: List[Dict], model_id: str, model_params: Dict, 
        output_format: str, chapter_audio_files: Dict, progress, total_fragments: int
    ):
        """Process fragments with checkpoint and retry mechanism."""
        processed_fragments = 0
        max_retries = 3
        
        # Get checkpoint info
        checkpoint_info = self.job_manager.get_checkpoint_info(job_id)
        processed_fragments_dict = checkpoint_info.get("processed_fragments", {})
        failed_fragments_dict = checkpoint_info.get("failed_fragments", {})
        
        # Count already processed fragments
        for chapter_idx_str, fragment_indices in processed_fragments_dict.items():
            processed_fragments += len(fragment_indices)
        
        logger.info(f"Resuming job {job_id} with {processed_fragments} already processed fragments")
        
        # Process each fragment
        for fragment_idx, fragment_info in enumerate(fragments):
            chapter_idx = fragment_info["chapter_index"]
            
            # Check if fragment is already processed
            if self.job_manager.is_fragment_processed(job_id, chapter_idx, fragment_info["fragment_index"]):
                logger.info(f"Fragment {fragment_idx+1}/{total_fragments} already processed, skipping")
                current_progress = (processed_fragments + 1) / total_fragments
                status_message = f"[{model_id.upper()}] Skipping already processed chapter {chapter_idx + 1} - fragment {fragment_info['fragment_index'] + 1}/{fragment_info['total_fragments_in_chapter']} - {current_progress*100:.1f}%"
                progress(current_progress, status_message)
                processed_fragments += 1
                continue
            
            # Check retry count
            retry_count = self.job_manager.get_fragment_retry_count(job_id, chapter_idx, fragment_info["fragment_index"])
            if retry_count >= max_retries:
                logger.warning(f"Fragment {fragment_idx+1}/{total_fragments} has failed {retry_count} times, skipping")
                current_progress = (processed_fragments + 1) / total_fragments
                status_message = f"[{model_id.upper()}] Skipping failed chapter {chapter_idx + 1} - fragment {fragment_info['fragment_index'] + 1}/{fragment_info['total_fragments_in_chapter']} - {current_progress*100:.1f}%"
                progress(current_progress, status_message)
                processed_fragments += 1
                continue
            
            # Process fragment with retry logic
            success = False
            for attempt in range(max_retries):
                try:

                    progress_percent = (processed_fragments + 1) / total_fragments * 100
                    status_message = f"[{model_id.upper()}] Processing chapter {chapter_idx + 1} - fragment {fragment_info['fragment_index'] + 1}/{fragment_info['total_fragments_in_chapter']} (attempt {attempt+1}/{max_retries}) - {progress_percent:.1f}%"

                    current_progress = (processed_fragments + 1) / total_fragments
                    progress(current_progress, status_message)

                    logger.info(f"[DEBUG] Processing fragment {fragment_idx+1}/{total_fragments}")
                    logger.info(f"[DEBUG] Fragment text: {fragment_info['text']}")
                    logger.info(f"[DEBUG] Fragment length: {len(fragment_info['text'])}")

                    worker_params = model_params.copy()
                    if model_id == "xtts":

                        worker_params["enable_text_splitting"] = True
                        logger.info("[ORCHESTRATOR] Enabled XTTS internal text splitting capability")

                    # Generate audio
                    audio_file = await self.worker_manager.generate_audio(
                        model_id, 
                        fragment_info["text"],
                        worker_params,  
                        f"{job_id}_{chapter_idx}_{fragment_info['fragment_index']}",
                        output_format
                    )
                    
                    if audio_file:
                        if chapter_idx not in chapter_audio_files:
                            chapter_audio_files[chapter_idx] = []
                        chapter_audio_files[chapter_idx].append(audio_file)
                        
                        # Mark fragment as processed
                        self.job_manager.update_fragment_status(job_id, chapter_idx, fragment_info["fragment_index"], "completed")
                        success = True
                        break
                    else:
                        raise Exception("Audio generation failed")
                        
                except Exception as e:
                    logger.error(f"Error processing fragment {fragment_idx+1}/{total_fragments} (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        # Update retry count
                        self.job_manager.update_fragment_status(job_id, chapter_idx, fragment_info["fragment_index"], "failed")
                        # Wait before retry
                        await asyncio.sleep(2 ** attempt)  
                        current_progress = (processed_fragments + 1) / total_fragments
                        status_message = f"[{model_id.upper()}] Retrying chapter {chapter_idx + 1} - fragment {fragment_info['fragment_index'] + 1}/{fragment_info['total_fragments_in_chapter']} (attempt {attempt+2}/{max_retries}) - {current_progress*100:.1f}%"
                        progress(current_progress, status_message)
                    else:
                        # Mark as failed after max retries
                        self.job_manager.update_fragment_status(job_id, chapter_idx, fragment_info["fragment_index"], "failed")
                        logger.error(f"Fragment {fragment_idx+1}/{total_fragments} failed after {max_retries} attempts")
            
            if success:
                processed_fragments += 1
                await self.job_manager.update_progress(job_id, processed_fragments / total_fragments)
                current_progress = processed_fragments / total_fragments
                status_message = f"[{model_id.upper()}] Completed chapter {chapter_idx + 1} - fragment {fragment_info['fragment_index'] + 1}/{fragment_info['total_fragments_in_chapter']} - {current_progress*100:.1f}%"
                progress(current_progress, status_message)
