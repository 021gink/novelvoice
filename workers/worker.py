"""
TTS Worker process that runs in isolated venv.
Provides HTTP API for text-to-speech synthesis.
Fixed version with enhanced error handling and API endpoints.
"""
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import torch

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

matcha_path = str(Path(__file__).parent.parent / "third_party" / "Matcha-TTS")
if matcha_path not in sys.path:
    sys.path.insert(0, matcha_path)

# Import model-specific engines
try:
    from workers.xtts_engine import XTTSWorker
except ImportError:
    XTTSWorker = None

try:
    from workers.kokoro_engine import KokoroWorker
except ImportError:
    KokoroWorker = None

try:
    from workers.chattts_engine import ChatTTSWorker
except ImportError:
    ChatTTSWorker = None

try:
    from workers.cosyvoice_engine import CosyVoiceWorker
except ImportError:
    CosyVoiceWorker = None

# Parameter mapping service is not available
get_param_mapper = None

class BaseWorker:
    """Base class for all TTS workers."""
    
    async def initialize(self):
        """Initialize the worker. To be implemented by subclasses."""
        pass
    
    async def synthesize(self, text: str, **kwargs) -> Dict[str, Any]:
        """Synthesize text to speech. To be implemented by subclasses."""
        raise NotImplementedError("synthesize method must be implemented by subclass")
    
    async def cleanup(self):
        """Cleanup resources. To be implemented by subclasses."""
        pass

logger = logging.getLogger(__name__)

class SynthesisRequest(BaseModel):
    """Request model for TTS synthesis."""
    text: str
    speed: float = 1.0
    seed: int = 42
    speaker: str = "default"
    speaker_wav: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 0.5
    repetition_penalty: float = 1.1
    sample_rate: int = 24000
    output_format: str = "wav"
    language: str = "zh-cn"  
    # CosyVoice specific parameters
    prompt_text: str = ""
    prompt_wav_upload: Optional[str] = None
    prompt_wav_record: Optional[str] = None
    instruct_text: Optional[str] = ""
    stream: bool = False
    mode: str = "预训练音色"
    text_frontend: bool = True
    zero_shot_spk_id: str = ""
    model_dropdown: Optional[str] = None
    # XTTS specific parameters
    top_k: int = 50
    length_penalty: float = 1.0
    enable_text_splitting: bool = False
    # Kokoro specific parameters
    use_gpu: bool = False

class SynthesisResponse(BaseModel):
    """Response model for TTS synthesis."""
    ok: bool
    path: str
    duration_ms: int
    error: Optional[str] = None

class HealthResponse(BaseModel):
    """Response model for health check."""
    ok: bool
    uptime_s: float
    model_id: str

class ModelsResponse(BaseModel):
    """Response model for models endpoint."""
    models: list
    current_model: str

class TTSWorker:
    """
    TTS Worker that loads a specific model and provides synthesis API.
    """
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.engine = None
        self.start_time = None
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging for the worker."""
        import os
        import sys

        if sys.platform.startswith('win'):
            os.environ['PYTHONIOENCODING'] = 'utf-8'
        
        logging.basicConfig(
            level=logging.INFO,
            format=f'%(asctime)s [{self.model_id}] %(levelname)s: %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("app.log", encoding='utf-8')
            ]
        )
    
    async def initialize(self):
        """Initialize the TTS engine."""
        self.start_time = asyncio.get_event_loop().time()
        
        # Map model IDs to engines
        engines = {}
        
        if XTTSWorker:
            engines["xtts"] = XTTSWorker
        if KokoroWorker:
            engines["kokoro"] = KokoroWorker
        if ChatTTSWorker:
            engines["chattts"] = ChatTTSWorker
        if CosyVoiceWorker:
            engines["cosyvoice"] = CosyVoiceWorker

        if self.model_id not in engines:
            raise ValueError(f"Unsupported model: {self.model_id}")
        
        logger.info(f"Initializing {self.model_id} engine...")
        self.engine = engines[self.model_id]()
        await self.engine.initialize()

        if self.model_id == "chattts":

            pass
            
        logger.info(f"{self.model_id} engine initialized successfully")
    
    async def synthesize(self, text: str, **kwargs) -> Dict[str, Any]:
        """Synthesize text to speech."""
        if not self.engine:
            raise RuntimeError("Engine not initialized")

        if self.model_id == "chattts" and kwargs.get("speaker") != "default":
            speaker_name = kwargs.get("speaker")
            try:

                speakers_dir = Path("speakers")
                speaker_file = speakers_dir / f"{speaker_name}.pt"
                
                if speaker_file.exists():

                    speaker_embedding = torch.load(speaker_file, map_location="cpu")

                    if hasattr(self.engine, 'speaker_embedding'):
                        self.engine.speaker_embedding = speaker_embedding
                        logger.info(f"Loaded custom speaker embedding: {speaker_name}")
                    else:
                        logger.warning("Engine does not support speaker embedding")
                else:
                    logger.warning(f"Speaker file not found: {speaker_file}")
            except Exception as e:
                logger.error(f"Failed to load speaker embedding: {e}")
        
        try:
            result = await self.engine.synthesize(text, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            raise

def create_app(model_id: str) -> FastAPI:
    """Create FastAPI app for the worker."""
    worker = TTSWorker(model_id)
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan event handler for FastAPI app."""
        try:
            # Startup: Initialize the worker
            await worker.initialize()
            yield
        except Exception as e:
            logger.error(f"Error during worker initialization: {e}")
            raise
        finally:
            # Shutdown: Cleanup the worker
            if hasattr(worker, 'engine') and worker.engine:
                try:
                    await worker.engine.cleanup()
                except Exception as e:
                    logger.error(f"Error during engine cleanup: {e}")
    
    try:
        app = FastAPI(
            title=f"{model_id.upper()} TTS Worker",
            description=f"Text-to-speech worker for {model_id} model",
            version="1.0.0",
            lifespan=lifespan
        )
    except Exception as e:
        logger.error(f"Error creating FastAPI app: {e}")
        raise
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        uptime = asyncio.get_event_loop().time() - worker.start_time if worker.start_time else 0
        return HealthResponse(
            ok=True,
            uptime_s=uptime,
            model_id=model_id
        )
    
    @app.get("/v1/models", response_model=ModelsResponse)
    async def list_models():
        """List available models endpoint - fixes 404 error."""
        return ModelsResponse(
            models=[model_id],
            current_model=model_id
        )
    
    @app.post("/synthesize", response_model=SynthesisResponse)
    async def synthesize(request: SynthesisRequest):
        """Synthesize text to speech."""
        try:

            logger.info(f"Received synthesis request: text='{request.text}', text_length={len(request.text)}, "
                       f"model={worker.model_id}, speaker={request.speaker}, mode={getattr(request, 'mode', 'N/A')}")

            logger.info(f"Request parameters: {request.dict()}")
            

            engine_kwargs = {
                "text": request.text,
                "speed": request.speed,
                "seed": request.seed,
                "speaker": request.speaker,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "repetition_penalty": request.repetition_penalty,
                "sample_rate": request.sample_rate,
                "output_format": request.output_format,
                "language": request.language
            }

            if worker.model_id == "xtts":
                engine_kwargs["speaker_wav"] = request.speaker_wav
                engine_kwargs["top_k"] = request.top_k
                engine_kwargs["length_penalty"] = request.length_penalty
                engine_kwargs["enable_text_splitting"] = request.enable_text_splitting

            if worker.model_id == "kokoro":
                engine_kwargs["use_gpu"] = request.use_gpu
 
            if worker.model_id == "cosyvoice":
                cosyvoice_params = {
                    "prompt_text": request.prompt_text,
                    "prompt_wav_upload": request.prompt_wav_upload,
                    "prompt_wav_record": request.prompt_wav_record,
                    "instruct_text": request.instruct_text,
                    "stream": request.stream,
                    "mode": request.mode,
                    "text_frontend": request.text_frontend,
                    "zero_shot_spk_id": request.zero_shot_spk_id,
                    "model_dropdown": request.model_dropdown,
                    "speaker": request.speaker
                }
                engine_kwargs.update(cosyvoice_params)
            
            logger.info(f"Engine kwargs: {engine_kwargs}")
            
            result = await worker.synthesize(**engine_kwargs)
            
            return SynthesisResponse(
                ok=True,
                path=result["path"],
                duration_ms=result["duration_ms"]
            )
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}", exc_info=True)
            logger.error(f"Full request data: {request}", exc_info=True)
            return SynthesisResponse(
                ok=False,
                path="",
                duration_ms=0,
                error=str(e)
            )
    
    @app.get("/audio/{filename}")
    async def get_audio(filename: str):
        """Serve generated audio files."""
        # Ensure filename is safe
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        file_path = Path("outputs") / "temp" / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(str(file_path))
    
    @app.post("/call_method")
    async def call_method(request: dict):
        """Call a specific method on the worker engine."""
        try:
            method_name = request.get("method_name")
            args = request.get("args", [])
            kwargs = request.get("kwargs", {})
            
            if not method_name:
                raise HTTPException(status_code=400, detail="Method name is required")

            if not hasattr(worker, 'engine') or worker.engine is None:
                raise HTTPException(status_code=500, detail="Engine not initialized")

            if not hasattr(worker.engine, method_name):
                raise HTTPException(status_code=404, detail=f"Method {method_name} not found")

            method = getattr(worker.engine, method_name)

            if asyncio.iscoroutinefunction(method):
                result = await method(*args, **kwargs)
            else:
                result = method(*args, **kwargs)
            
            return {"ok": True, "result": result}
            
        except Exception as e:
            logger.error(f"Error calling method {request.get('method_name', 'unknown')}: {e}", exc_info=True)
            return {"ok": False, "error": str(e)}
    
    return app

def main():
    """Main entry point for the worker."""
    parser = argparse.ArgumentParser(description="TTS Worker Process")
    parser.add_argument("--model", required=True, help="Model ID to load")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    
    args = parser.parse_args()
    
    app = create_app(args.model)

    import logging
    uvicorn_logger = logging.getLogger("uvicorn.access")
    uvicorn_logger.setLevel(logging.WARNING)  
    
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port,
        log_level="warning"  
    )

if __name__ == "__main__":
    main()
