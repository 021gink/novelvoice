"""
Kokoro TTS engine implementation.
"""
import asyncio
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import numpy as np
import torch
import soundfile as sf

logger = logging.getLogger(__name__)

class KokoroWorker:
    """Kokoro TTS worker implementation."""
    
    def __init__(self):
        self.model = None
        self.device = None
        self.sample_rate = 24000
        self.model_loaded = False 
        self.kmodel = None
        self.pipeline_zh = None

        # self.pipeline_en = None

    async def initialize(self):
        """Initialize Kokoro model."""
        try:
            from kokoro import KModel, KPipeline
            
            # Auto-detect device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # Load Kokoro model with custom path to use pre-downloaded models
            local_models_dir = Path(__file__).parent.parent / "models" / "kokoro"
            local_model_path = local_models_dir / "kokoro-v1_1-zh.pth"
            local_config_path = local_models_dir / "config.json"
            
            if local_model_path.exists():
                logger.info(f"Found local kokoro model: {local_model_path}")
                logger.info(f"Model size: {local_model_path.stat().st_size / 1024 / 1024:.1f} MB")
                
                # Load model with local paths
                self.kmodel = KModel(
                    repo_id='hexgrad/Kokoro-82M-v1.1-zh',
                    config=str(local_config_path) if local_config_path.exists() else None,
                    model=str(local_model_path)
                ).to(self.device).eval()

                self.pipeline_zh = KPipeline(lang_code='z', model=self.kmodel, trf=False)

                
                self.model_loaded = True
                logger.info("Kokoro model loaded successfully")
            else:
                logger.error(f"Local model not found at: {local_model_path}")
                raise FileNotFoundError("Local model not available")
                
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro model: {e}")
            raise RuntimeError(f"Failed to initialize Kokoro model: {e}")

    def _resolve_speaker(self, speaker: str) -> str:
        """解析speaker名称，返回有效的语音标识符。"""

        voices_dir = Path(__file__).parent.parent / "models" / "kokoro" / "voices"
        voice_file = voices_dir / f"{speaker}.pt"
        
        if voice_file.exists():
            return speaker

        logger.warning(f"Speaker '{speaker}' not found, using default 'zf_001'")
        return "zf_001"

    def _is_chinese_text(self, text: str) -> bool:
        """判断文本是否为中文。"""

        return True

    def _get_pipeline(self, is_chinese: bool):
        """获取对应的pipeline。"""

        return self.pipeline_zh

    async def _synthesize_with_pipeline(self, text: str, speaker: str, speed: float) -> Optional[torch.Tensor]:
        """使用pipeline合成语音。"""

        pipeline = self.pipeline_zh
        
        if pipeline is None:
            return None
        
        try:

            voices_dir = Path(__file__).parent.parent / "models" / "kokoro" / "voices"
            voice_path = voices_dir / f"{speaker}.pt"
            
            if voice_path.exists():
                pack = pipeline.load_single_voice(str(voice_path))
                
                ps, _ = pipeline.g2p(text)
                
                pack = pack.to(self.kmodel.device)

                output = self.kmodel(ps, pack[len(ps)-1], speed=speed, return_output=True)
                
                if output is not None and hasattr(output, 'audio'):
                    return output.audio
                
            return None
        except Exception as e:
            logger.warning(f"Pipeline synthesis failed for '{speaker}': {e}")
            return None
        
        return None

    async def _synthesize_with_fallback(self, text: str, speaker: str, speed: float) -> Optional[torch.Tensor]:
        """使用fallback方法合成语音。"""
        try:
            # 创建临时pipeline用于fallback，禁用自动下载
            from kokoro import KPipeline
            fallback_pipeline = KPipeline(
                lang_code='z',  # 只使用中文pipeline
                model=self.kmodel,
                trf=False
            )
            
            # 直接加载本地语音文件
            voices_dir = Path(__file__).parent.parent / "models" / "kokoro" / "voices"
            voice_path = voices_dir / f"{speaker}.pt"
            
            if voice_path.exists():
                pack = fallback_pipeline.load_single_voice(str(voice_path))
                
                # 将文本转换为音素
                ps, _ = fallback_pipeline.g2p(text)
                
                # 确保在正确的设备上
                pack = pack.to(self.kmodel.device)
                
                # 使用模型直接合成
                output = self.kmodel(ps, pack[len(ps)-1], speed=speed, return_output=True)
                
                if output is not None and hasattr(output, 'audio'):
                    return output.audio
            
            return None
        except Exception as e:
            logger.error(f"Fallback synthesis failed: {e}")
            return None

    def _ensure_audio_valid(self, audio: Optional[torch.Tensor], sample_rate: int) -> torch.Tensor:
        """确保音频数据有效。"""
        if audio is None or (isinstance(audio, np.ndarray) and len(audio) == 0):
            logger.warning("No valid audio generated, creating silence")
            return torch.zeros(sample_rate, dtype=torch.float32)

        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio.float() if hasattr(audio, 'float') else torch.tensor(audio).float()

        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.squeeze()

        if len(audio_tensor) < 1000:
            logger.warning("Audio too short, padding to minimum length")
            audio_tensor = torch.cat([audio_tensor, torch.zeros(1000 - len(audio_tensor))])
        
        return audio_tensor

    def _save_audio(self, audio_tensor: torch.Tensor, output_path: Path, sample_rate: int) -> bool:
        """保存音频文件。"""
        try:

            output_path.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(audio_tensor, torch.Tensor):
                audio_np = audio_tensor.cpu().numpy()
            else:
                audio_np = np.array(audio_tensor)

            sf.write(str(output_path), audio_np, sample_rate)

            if output_path.exists() and output_path.stat().st_size > 1000:
                logger.info(f"Audio saved: {output_path} ({output_path.stat().st_size} bytes)")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            return False

    async def synthesize(self, text: str, **kwargs) -> Dict[str, Any]:
        """Synthesize text to speech using Kokoro."""

        speed = kwargs.get("speed", 1.0)
        seed = kwargs.get("seed", 42)
        speaker = kwargs.get("speaker", "zf_001")  
        sample_rate = kwargs.get("sample_rate", self.sample_rate)
        use_gpu = kwargs.get("use_gpu", False)  
        output_format = kwargs.get("output_format", "wav")  
        

        torch.manual_seed(seed)
        np.random.seed(seed)

        if not use_gpu:
            if hasattr(self.kmodel, 'to'):
                self.kmodel = self.kmodel.to('cpu')
            if hasattr(self.pipeline_zh, 'to'):
                self.pipeline_zh = self.pipeline_zh.to('cpu')

        resolved_speaker = self._resolve_speaker(speaker)
        logger.info(f"Final speaker selection: '{speaker}' -> '{resolved_speaker}', use_gpu={use_gpu}")

        temp_dir = Path("outputs/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)

        suffix = f".{output_format}" if output_format.startswith(".") else f".{output_format}"
        with tempfile.NamedTemporaryFile(
            suffix=suffix, 
            delete=False,
            dir=temp_dir
        ) as tmp_file:
            output_path = Path(tmp_file.name)
        
        try:

            if not self.model_loaded:
                raise RuntimeError("Kokoro model is not loaded")

            audio = None

            audio = await self._synthesize_with_pipeline(text, resolved_speaker, speed)

            if audio is None:
                audio = await self._synthesize_with_fallback(text, resolved_speaker, speed)

            audio_tensor = self._ensure_audio_valid(audio, sample_rate)

            success = self._save_audio(audio_tensor, output_path, sample_rate)
            
            if not success:
                raise RuntimeError("Failed to save audio file")

            duration_ms = int(len(audio_tensor) / sample_rate * 1000)
            
            return {
                "path": str(output_path),
                "duration_ms": duration_ms,
                "sample_rate": sample_rate
            }
            
        except Exception as e:

            try:
                if output_path.exists():
                    output_path.unlink()
            except:
                pass
            
            logger.error(f"Synthesis failed: {e}")
            raise

    async def cleanup(self):
        """Clean up resources."""
        try:

            if self.kmodel is not None:
                del self.kmodel
            if self.pipeline_zh is not None:
                del self.pipeline_zh

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.model_loaded = False
            logger.info("Kokoro resources cleaned up")
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

KokoroEngine = KokoroWorker