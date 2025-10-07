"""
ChatTTS engine implementation.
"""
import asyncio
import tempfile
from pathlib import Path
from typing import Dict, Any
import logging
import numpy as np
import torch
import torchaudio
import os  

logger = logging.getLogger(__name__)

class ChatTTSWorker:
    """ChatTTS worker implementation."""
    
    def __init__(self):
        self.model = None
        self.device = None
        self.sample_rate = 24000
        self.speaker_embedding = None  
        # 定义speakers_dir属性
        self.speakers_dir = Path(__file__).parent.parent / "speakers"
        
    async def initialize(self):
        """Initialize ChatTTS model."""
        try:
            import ChatTTS
            
            # Auto-detect device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # Load ChatTTS model with custom path to use pre-downloaded models
            self.model = ChatTTS.Chat()
            # Use custom source and specify the path to our pre-downloaded models
            model_path = Path(__file__).parent.parent / "models" / "chattts"
            logger.info(f"Loading ChatTTS model from: {model_path}")
            self.model.load(compile=False, source="custom", custom_path=str(model_path))
            
            # Sample a random speaker embedding for consistent voice
            self.speaker_embedding = self.model.sample_random_speaker()
            logger.info("ChatTTS model loaded successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import ChatTTS library: {e}")
            raise RuntimeError(
                "ChatTTS library not found. Install with: pip install ChatTTS"
            )
        except Exception as e:
            logger.error(f"Failed to load ChatTTS model: {e}")
            # Fallback to local source if custom path fails
            try:
                logger.info("Falling back to local source loading")
                self.model.load(compile=False, source="local")
                # Sample a random speaker embedding for consistent voice
                self.speaker_embedding = self.model.sample_random_speaker()
                logger.info("ChatTTS model loaded successfully with local source")
            except Exception as e2:
                logger.error(f"Failed to load ChatTTS model with local source: {e2}")
                raise RuntimeError(f"Failed to load ChatTTS model: {e}")
    
    def load_speaker_embedding(self, filename: str):
        """Load speaker embedding from a file."""
        filepath = self.speakers_dir / f"{filename}.pt"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Speaker embedding file not found: {filepath}")
        
        # 加载音色嵌入
        self.speaker_embedding = torch.load(filepath, map_location=self.device)
        logger.info(f"Speaker embedding loaded from: {filepath}")
    
    def save_speaker(self, name: str, seed: int, temperature: float, top_p: float, top_k: int) -> str:
        """Save a speaker embedding with given parameters."""
        if not name:
            return "Please enter a speaker name"
        
        if not name.replace("_", "").replace("-", "").isalnum():
            return "Speaker name can only contain letters, numbers, underscores, and hyphens"
        
        try:
            # 确保模型已加载
            if self.model is None:
                raise RuntimeError("ChatTTS model not initialized")
            
            # 设置随机种子
            torch.manual_seed(int(seed))
            
            # 生成音色嵌入
            speaker_embedding = self.model.sample_random_speaker()
            
            # 保存音色嵌入
            self.speaker_embedding = speaker_embedding
            self._save_speaker_embedding_to_file(name)
            
            return f"Speaker '{name}' saved successfully!"
        except Exception as e:
            logger.error(f"Error saving speaker: {str(e)}", exc_info=True)
            return f"Error saving speaker: {str(e)}"
    
    def _save_speaker_embedding_to_file(self, filename: str):
        """Save current speaker embedding to a file."""
        if self.speaker_embedding is None:
            raise RuntimeError("No speaker embedding available to save")
        
        # 确保speakers目录存在
        self.speakers_dir.mkdir(exist_ok=True, parents=True)
        
        # 保存音色嵌入到文件
        filepath = self.speakers_dir / f"{filename}.pt"
        torch.save(self.speaker_embedding, filepath)
        logger.info(f"Speaker embedding saved to: {filepath}")
    
    async def preview_speaker(self, seed: int, temperature: float, top_p: float, top_k: int) -> tuple:
        """Preview a speaker with given parameters."""
        try:
            preview_text = "你好，这是一个音色预览测试。"
            
            # 调用synthesize方法生成预览音频
            result = await self.synthesize(
                preview_text,
                seed=int(seed),
                temperature=temperature,
                top_p=top_p,
                top_k=int(top_k),
                speaker="default",
                output_format="wav"
            )
            
            return f"Preview generated successfully! Seed: {seed}", result["path"]
        except Exception as e:
            logger.error(f"Error previewing speaker: {str(e)}", exc_info=True)
            return f"Error previewing speaker: {str(e)}", None
    
    def delete_speaker(self, name: str) -> str:
        """Delete a speaker by name."""
        if not name or name == "default":
            return "Please select a valid speaker to delete"
        
        try:
            filepath = self.speakers_dir / f"{name}.pt"
            
            if not filepath.exists():
                return f"Speaker file '{name}.pt' not found"
            
            filepath.unlink()
            logger.info(f"Deleted speaker file: {filepath}")
            
            return f"Speaker '{name}' deleted successfully!"
        except Exception as e:
            logger.error(f"Error deleting speaker: {str(e)}", exc_info=True)
            return f"Error deleting speaker: {str(e)}"
    
    async def synthesize(self, text: str, **kwargs) -> Dict[str, Any]:
        """Synthesize text to speech using ChatTTS."""
        speed = kwargs.get("speed", 1.0)
        seed = kwargs.get("seed", 42)
        speaker = kwargs.get("speaker", "default")
        sample_rate = kwargs.get("sample_rate", self.sample_rate)
        temperature = kwargs.get("temperature", 0.3)
        top_p = kwargs.get("top_p", 0.7)
        top_k = kwargs.get("top_k", 20)
        output_format = kwargs.get("output_format", "wav")  # 添加output_format参数处理
        
        # Validate and sanitize input text to prevent issues
        if not isinstance(text, str):
            text = str(text)
        
        # Ensure text is not empty
        if len(text.strip()) == 0:
            text = "."  # Use a simple dot for empty text
            logger.info("Empty text detected, using '.' as fallback")
        
        # Limit text length to prevent issues with very long texts
        max_text_length = 1000  # Limit to 1000 characters
        if len(text) > max_text_length:
            text = text[:max_text_length]
            logger.info(f"Text truncated to {max_text_length} characters")
        
        # Log detailed parameters for debugging
        logger.info(f"ChatTTS synthesis parameters: text='{text}' (length={len(text)}), "
                   f"speed={speed}, seed={seed}, speaker={speaker}, "
                   f"temperature={temperature}, top_p={top_p}, top_k={top_k}")
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # For 'default' speaker, generate a new speaker embedding based on seed
        # FIX: Generate speaker embedding based on seed for consistent voice generation
        if speaker == "default":
            # Generate a new speaker embedding based on the seed
            speaker_embedding = self.model.sample_random_speaker()
        else:
            # Use fixed speaker embedding for non-default speakers
            # 检查是否已加载音色嵌入
            if self.speaker_embedding is None:
                logger.warning("Speaker embedding not loaded, using random speaker")
                speaker_embedding = self.model.sample_random_speaker()
            else:
                speaker_embedding = self.speaker_embedding
        
        # 确保临时目录存在
        temp_dir = Path("outputs/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建临时文件，根据output_format设置后缀
        suffix = f".{output_format}" if output_format.startswith(".") else f".{output_format}"
        with tempfile.NamedTemporaryFile(
            suffix=suffix, 
            delete=False,
            dir=temp_dir
        ) as tmp_file:
            output_path = Path(tmp_file.name)
        
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate speech using proper parameter names for ChatTTS
            logger.info("Starting ChatTTS inference...")
            
            # Use proper parameter names for ChatTTS
            try:
                # Create parameter objects with correct names and speaker embedding
                infer_code_params = self.model.InferCodeParams(
                    spk_emb=speaker_embedding,  # Use speaker embedding based on seed
                    temperature=temperature,
                    top_P=top_p,  # Note: top_P not top_p
                    top_K=top_k,   # Note: top_K not top_k
                    manual_seed=seed  # Pass seed to parameters
                )
                
                # For text refinement, we might also need RefineTextParams
                refine_text_params = self.model.RefineTextParams(
                    temperature=temperature,
                    top_P=top_p,
                    top_K=top_k,
                    manual_seed=seed  # Pass seed to parameters
                )
                
                # Call infer with correct parameter names
                wavs = self.model.infer(
                    [text],
                    params_infer_code=infer_code_params,
                    params_refine_text=refine_text_params,
                    # Add additional safety parameters to prevent cache length issues
                    skip_refine_text=False,  # Enable text refinement but with safety
                    do_text_normalization=True,
                    do_homophone_replacement=True
                )
            except Exception as infer_error:
                logger.error(f"Error during ChatTTS inference with parameters: {infer_error}")
                # Try with text refinement disabled as a fallback
                try:
                    logger.info("Trying with text refinement disabled...")
                    wavs = self.model.infer(
                        [text],
                        skip_refine_text=True,  # Skip text refinement
                        params_infer_code=self.model.InferCodeParams(
                            spk_emb=speaker_embedding,  # Use fixed speaker embedding
                            temperature=temperature,
                            top_P=top_p,
                            top_K=top_k,
                            manual_seed=seed
                        )
                    )
                except Exception as simple_infer_error:
                    logger.error(f"Error during simple ChatTTS inference: {simple_infer_error}")
                    # Last resort: try with minimal text and additional parameters
                    if len(text.strip()) == 0:
                        wavs = self.model.infer(
                            ["."],
                            skip_refine_text=True,
                            params_infer_code=self.model.InferCodeParams(
                                spk_emb=speaker_embedding,  # Use fixed speaker embedding
                                temperature=0.3,  # Lower temperature for stability
                                top_P=0.7,
                                top_K=20,
                                manual_seed=seed
                            )
                        )
                    else:
                        # Try with a simple text and reduced parameters
                        wavs = self.model.infer(
                            [text[:50]],  # Limit text length
                            skip_refine_text=True,
                            params_infer_code=self.model.InferCodeParams(
                                spk_emb=speaker_embedding,  # Use fixed speaker embedding
                                temperature=0.3,  # Lower temperature for stability
                                top_P=0.7,
                                top_K=20,
                                manual_seed=seed
                            )
                        )
            
            # Process audio
            if wavs and len(wavs) > 0:
                audio = wavs[0]
                logger.info(f"Generated audio with shape: {audio.shape if hasattr(audio, 'shape') else 'N/A'}")
                
                # Ensure audio is numpy array
                if isinstance(audio, torch.Tensor):
                    audio = audio.cpu().numpy()
                
                # Ensure audio is 1D
                if audio.ndim > 1:
                    audio = audio.flatten()
                
                # Apply speed adjustment if needed
                if speed != 1.0:
                    audio = self._adjust_speed(audio, speed)
                
                # Prepare tensor for saving (ensure it's 2D: [channels, samples])
                if isinstance(audio, np.ndarray):
                    audio_tensor = torch.from_numpy(audio).float()
                else:
                    audio_tensor = audio.float()
                
                # Ensure 2D tensor for torchaudio.save
                if audio_tensor.ndim == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)  # Shape: [1, samples]
                
                # Save audio file with format-specific encoding
                if output_format.lower() == "wav":
                    torchaudio.save(
                        str(output_path),
                        audio_tensor,
                        sample_rate=sample_rate,
                        format="wav"
                    )
                elif output_format.lower() == "mp3":
                    # Convert to MP3 format with proper encoding
                    import soundfile as sf
                    audio_np = audio_tensor.numpy()
                    if audio_np.ndim > 1:
                        audio_np = audio_np.squeeze()
                    sf.write(str(output_path), audio_np, sample_rate, format='MP3')
                    # 强制刷新到磁盘，确保文件完全写入
                    # FIX: 使用更安全的文件同步方法，避免Bad file descriptor错误
                    try:
                        with open(str(output_path), 'rb') as f:
                            os.fsync(f.fileno())
                    except OSError as e:
                        logger.warning(f"Failed to sync file to disk: {e}")
                        # 文件可能已经关闭，继续执行
                else:
                    # Default to WAV for unsupported formats
                    torchaudio.save(
                        str(output_path),
                        audio_tensor,
                        sample_rate=sample_rate,
                        format="wav"
                    )
                    logger.warning(f"Unsupported output format '{output_format}', defaulting to WAV")
                
                # Calculate duration
                if isinstance(audio, np.ndarray):
                    duration_ms = int(len(audio) / sample_rate * 1000)
                else:
                    duration_ms = int(audio_tensor.shape[-1] / sample_rate * 1000)
                
                logger.info(f"Successfully synthesized audio: path={output_path}, duration={duration_ms}ms")
                
                return {
                    "path": str(output_path),
                    "duration_ms": duration_ms,
                    "sample_rate": sample_rate
                }
            else:
                raise RuntimeError("No audio generated")
                
        except Exception as e:
            # Clean up on error
            if output_path.exists():
                output_path.unlink()
            logger.error(f"ChatTTS synthesis failed: {e}", exc_info=True)
            raise RuntimeError(f"ChatTTS synthesis failed: {e}")
    
    def _adjust_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        """Adjust audio speed."""
        import librosa
        return librosa.effects.time_stretch(audio, rate=speed)
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.model:
            del self.model
            torch.cuda.empty_cache()
            logger.info("ChatTTS model cleaned up")