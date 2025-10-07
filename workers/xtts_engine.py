import asyncio
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import numpy as np
import torch
import torchaudio
import re
import os
import sys

logger = logging.getLogger(__name__)

class XTTSWorker:
    """XTTS TTS worker implementation."""
    
    def __init__(self):
        self.model = None
        self.device = None
        self.sample_rate = 24000
        self.model_loaded = False
        self.custom_path = None
        self.default_speaker_wav = None 

    async def initialize(self):
        """Initialize XTTS model from local model files."""
        try:
            logger.info("Starting XTTS initialization...")
            
        
            base_dir = Path(__file__).parent.parent
            venv_path = base_dir / "venvs" / "xtts"
            
       
            if not venv_path.exists():
                logger.error(f"XTTS virtual environment not found at: {venv_path}")
                raise FileNotFoundError(f"XTTS virtual environment not found at: {venv_path}")
            
        
            site_packages = venv_path / "Lib" / "site-packages"
            if site_packages.exists():
                sys.path.insert(0, str(site_packages))
                logger.info(f"Added virtual environment to path: {site_packages}")
            
       
            models_dir = base_dir / "models" / "xtts_v2"
            if not models_dir.exists():
                logger.error(f"XTTS models directory not found at: {models_dir}")
                raise FileNotFoundError(f"XTTS models directory not found at: {models_dir}")
            
            logger.info(f"Found XTTS models directory: {models_dir}")
          
            self.default_speaker_wav = models_dir / "samples" / "zh-cn-sample.wav"
            if not self.default_speaker_wav.exists():
                logger.warning(f"Default speaker audio not found at: {self.default_speaker_wav}")
        
                for sample_file in models_dir.glob("samples/*.wav"):
                    self.default_speaker_wav = sample_file
                    logger.info(f"Using alternative default speaker audio: {self.default_speaker_wav}")
                    break
                else:
                    self.default_speaker_wav = None
            else:
                logger.info(f"Default speaker audio set to: {self.default_speaker_wav}")
            
       
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
   
            logger.info("Importing TTS library...")
            from TTS.api import TTS
            from TTS.utils.synthesizer import Synthesizer
            
   
            logger.info("Loading XTTS v2 model from local directory...")
            

            logger.info(f"Creating Synthesizer with model_dir: {models_dir}")
            self.synthesizer = Synthesizer(
                model_dir=str(models_dir),
                use_cuda=(self.device == "cuda")
            )
            
            logger.info("Synthesizer created successfully")
            

            logger.info("Creating TTS instance...")
            self.model = TTS()
            self.model.synthesizer = self.synthesizer
            
            self.model_loaded = True
            logger.info("XTTS model loaded successfully from local directory")
                
        except ImportError as e:
            logger.error(f"Failed to import TTS from virtual environment: {e}")
            raise RuntimeError(f"Failed to import TTS from virtual environment: {e}")
                
        except Exception as e:
            logger.error(f"Failed to initialize XTTS model: {e}")
            raise RuntimeError(f"Failed to initialize XTTS model: {e}")
    
    async def synthesize(self, text: str, **kwargs) -> Dict[str, Any]:
        """Synthesize text to speech using XTTS."""
      
        speed = kwargs.get("speed", 1.0)
        seed = kwargs.get("seed", 42)
        speaker_wav = kwargs.get("speaker_wav") 
        language = kwargs.get("language", "zh-cn") 
        sample_rate = kwargs.get("sample_rate", self.sample_rate)
        output_format = kwargs.get("output_format", "wav")
        

        temperature = kwargs.get("temperature", 0.65)          
        length_penalty = kwargs.get("length_penalty", 1.0)       
        repetition_penalty = kwargs.get("repetition_penalty", 2.0) 
        top_k = kwargs.get("top_k", 50)                         
        top_p = kwargs.get("top_p", 0.8)                        
        split_sentences = kwargs.get("split_sentences", True)    
        enable_text_splitting = kwargs.get("enable_text_splitting", True) 
        

        logger.info("XTTS internal text splitting is enabled and will be used if needed")
        

        if self._is_chinese(text):
            logger.info("Detected Chinese text, XTTS internal splitting will be used")
            enable_text_splitting = True
        
   
        if len(text) > 400:
            logger.warning(f"Text length ({len(text)}) exceeds XTTS limit (400). XTTS internal splitting will be used.")
        else:
            logger.info(f"Text length ({len(text)}) is within XTTS limit (400)")
        
  
        torch.manual_seed(seed)
        
 
        if speaker_wav:
            speaker_wav_path = Path(speaker_wav)
            if not speaker_wav_path.exists():
                logger.warning(f"Provided speaker audio not found: {speaker_wav}")
       
                if self.default_speaker_wav and self.default_speaker_wav.exists():
                    speaker_wav = str(self.default_speaker_wav)
                    logger.info(f"Falling back to default speaker audio: {speaker_wav}")
                else:
                    speaker_wav = None  
            else:
                logger.info(f"Using provided speaker audio: {speaker_wav}")
        elif self.default_speaker_wav and self.default_speaker_wav.exists():
        
            speaker_wav = str(self.default_speaker_wav)
            logger.info(f"Using default speaker audio: {speaker_wav}")
        
        if not speaker_wav:
            raise ValueError("XTTS requires a speaker_wav for voice cloning. Please provide a reference audio.")
        
    
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
                raise RuntimeError("XTTS model is not loaded")
            
          
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
        
            logger.info(f"XTTS synthesis parameters: text='{text}' (length={len(text)}), "
                       f"speed={speed}, language={language}, temperature={temperature}, "
                       f"top_k={top_k}, top_p={top_p}, repetition_penalty={repetition_penalty}, "
                       f"speaker_wav={speaker_wav}, enable_text_splitting={enable_text_splitting}, "
                       f"split_sentences={split_sentences}")
            
          
            logger.info("Starting XTTS synthesis...")
            self.model.tts_to_file(
                text=text,
                file_path=str(output_path),
                speaker_wav=speaker_wav,  
                language=language,
                speed=speed,
                
                temperature=temperature,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                top_k=top_k,
                top_p=top_p,
                
            
                split_sentences=split_sentences,
                enable_text_splitting=enable_text_splitting
            )
            
            logger.info("XTTS synthesis completed")
            
         
            waveform, sr = torchaudio.load(str(output_path))
            duration_ms = int(waveform.shape[1] / sr * 1000)
            
            logger.info(f"Successfully synthesized audio: path={output_path}, duration={duration_ms}ms")
            
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
            
            logger.error(f"XTTS synthesis failed: {e}", exc_info=True)
            raise RuntimeError(f"XTTS synthesis failed: {e}")
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self.model:
                del self.model
                self.model = None
                self.model_loaded = False
                
           
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info("XTTS model cleaned up")
                
        except Exception as e:
            logger.warning(f"Error during XTTS cleanup: {e}")
        
        base_dir = Path(__file__).parent.parent
        venv_path = base_dir / "venvs" / "xtts"
        site_packages = venv_path / "Lib" / "site-packages"
        
        if site_packages.exists() and str(site_packages) in sys.path:
            sys.path.remove(str(site_packages))
            logger.info("Virtual environment path removed from sys.path")
    
    def _is_chinese(self, text: str) -> bool:
        """判断文本是否为中文。"""

        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        return len(chinese_chars) / max(len(text), 1) > 0.3