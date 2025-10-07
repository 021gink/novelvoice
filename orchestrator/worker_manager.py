
"""
Manages TTS worker processes, including starting, stopping, and health checking.
Fixed version with improved error handling and parameter validation.
"""
import asyncio
import logging
import subprocess
import time
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import psutil
import requests
import json
import numpy as np

try:
    from .config import RESOURCE_LIMITS, WORKER_PORT_RANGE
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import RESOURCE_LIMITS, WORKER_PORT_RANGE

logger = logging.getLogger(__name__)

class WorkerManager:
    """
    Manages lifecycle of TTS worker processes.
    Each worker runs in its own venv and exposes HTTP API.
    """
    
    def __init__(self, model_configs: dict):
        self.workers: Dict[str, Dict] = {} 
        self.model_configs = model_configs
        self.port_counter = WORKER_PORT_RANGE.start

        temp_dir = Path("outputs/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Temporary directory ensured: {temp_dir}")
        
    async def start_workers(self, model_ids: list):
        """Start workers for the specified models."""
        for model_id in model_ids:
            if model_id not in self.model_configs:
                raise ValueError(f"Unknown model: {model_id}")
            
            if model_id not in self.workers:
                timeout = 180 if model_id in ["xtts", "cosyvoice"] else 120
                await self._start_worker(model_id, timeout)
            else:
                logger.info(f"Worker {model_id} is already running on port {self.workers[model_id]['port']}")
    
    async def _start_worker(self, model_id: str, timeout: int = 120):
        """Start a single worker process with pre-cleanup."""
        await self._cleanup_residual_processes(model_id)
        
        config = self.model_configs[model_id]
        port = await self._get_next_port()
        venv_path = Path(config["venv_path"])
        
        if not venv_path.exists():
            raise RuntimeError(f"Virtual environment not found: {venv_path}")
        
        if Path.cwd().drive:  
            python_exe = venv_path / "Scripts" / "python.exe"
        else:  
            python_exe = venv_path / "bin" / "python"
        
        worker_script = Path(__file__).parent.parent / "workers" / "worker.py"
        
        cmd = [
            str(python_exe),
            str(worker_script),
            "--model", model_id,
            "--port", str(port)
        ]
        
        logger.info(f"Starting worker for {model_id} on port {port} with {timeout}s timeout")
        logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            env = os.environ.copy()
            project_root = str(Path(__file__).parent.parent)
            venv_site_packages = str(venv_path / "Lib" / "site-packages")

            env['PYTHONPATH'] = venv_site_packages + os.pathsep + project_root

            cosyvoice_path = project_root + os.sep + 'cosyvoice'
            matcha_path = cosyvoice_path + os.sep + 'third_party' + os.sep + 'Matcha-TTS'
            workers_path = project_root + os.sep + 'workers'
            
            env['PYTHONPATH'] = cosyvoice_path + os.pathsep + matcha_path + os.pathsep + workers_path + os.pathsep + env.get('PYTHONPATH', '')
            
            if model_id == "cosyvoice":
                try:
                    import wetext
                    wetext_fst_path = str(Path(wetext.__file__).parent / 'fsts')
                    if Path(wetext_fst_path).exists():
                        env['WETEXT_FST_PATH'] = wetext_fst_path
                except ImportError:
                    pass
                
            logger.info(f"Environment variables for worker process:")
            logger.info(f"  PYTHONPATH: {env.get('PYTHONPATH')}")
            if model_id == "cosyvoice":
                logger.info(f"  WETEXT_FST_PATH: {env.get('WETEXT_FST_PATH')}")

            process = subprocess.Popen(
                cmd,
                stdout=None,
                stderr=None,
                text=True,
                env=env
            )
            
            logger.info(f"Worker process started with PID: {process.pid}")

            time.sleep(1)
            
            if process.poll() is not None:
                logger.error(f"Worker process exited immediately with return code: {process.returncode}")
                raise RuntimeError(f"Worker process for {model_id} exited immediately")
            
            logger.info(f"Worker process is running, waiting for health check...")

            if await self._wait_for_worker(port, timeout):
                self.workers[model_id] = {
                    "process": process,
                    "port": port,
                    "url": f"http://localhost:{port}",
                    "start_time": time.time(),
                    "last_health_check": time.time()
                }
                logger.info(f"Worker {model_id} started successfully on port {port}")
            else:
                if process.poll() is None:
                    logger.error(f"Worker {model_id} process is still running but not responding to health checks")
                    process.terminate()
                else:
                    logger.error(f"Worker {model_id} process exited with return code: {process.returncode}")
                
                raise RuntimeError(f"Failed to start worker {model_id} after {timeout}s timeout")
                
        except Exception as e:
            logger.error(f"Error starting worker {model_id}: {e}")
            raise
    
    async def _cleanup_residual_processes(self, model_id: str):
        """Clean up any residual processes for the specified model."""
        logger.info(f"Checking for residual processes for {model_id}")
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline') or []
                    cmdline_str = ' '.join(cmdline)

                    if ('worker.py' in cmdline_str and 
                        f'--model {model_id}' in cmdline_str):
                        logger.warning(f"Found residual worker process PID {proc.info['pid']}: {cmdline_str}")

                        process = psutil.Process(proc.info['pid'])
                        process.terminate()
                        
                 
                        try:
                            process.wait(timeout=3)
                            logger.info(f"Successfully terminated residual process PID {proc.info['pid']}")
                        except psutil.TimeoutExpired:
                            process.kill()
                            logger.warning(f"Force killed residual process PID {proc.info['pid']}")
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
                    
        except Exception as e:
            logger.error(f"Error during residual process cleanup: {e}")
    
    async def _get_next_port(self) -> int:
        """Get the next available port for a worker with enhanced checking."""
        import socket
        
        for _ in range(20):
            port = self.port_counter
            self.port_counter += 1
            
            if self.port_counter >= WORKER_PORT_RANGE.stop:
                self.port_counter = WORKER_PORT_RANGE.start

            if await self._is_port_available(port):
                logger.info(f"Selected available port: {port}")
                return port
        
        raise RuntimeError("No available ports found")
    
    async def _is_port_available(self, port: int) -> bool:
        """Check if a port is available with enhanced validation."""
        import socket
        
        try:
            for conn in psutil.net_connections():
                if conn.laddr.port == port:
                    logger.warning(f"Port {port} is already in use by PID {conn.pid}")
                    return False

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.bind(('localhost', port))
                return True
                
        except (socket.error, OSError) as e:
            logger.warning(f"Port {port} check failed: {e}")
            return False
    
    async def _wait_for_worker(self, port: int, timeout: int = 120) -> bool:
        """Wait for worker to start responding with enhanced error handling."""
        start_time = time.time()
        url = f"http://localhost:{port}/health"
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"Worker on port {port} is healthy")
                    return True
            except requests.RequestException as e:
                logger.debug(f"Waiting for worker on port {port}: {e}")
            
            await asyncio.sleep(2)
        
        logger.error(f"Worker on port {port} failed to start within {timeout}s")
        return False
    
    async def health_check_workers(self):
        """Check health of all workers with enhanced error handling."""
        failed_workers = []
        
        for model_id, worker_info in self.workers.items():
            try:
                url = f"{worker_info['url']}/health"
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    worker_info['last_health_check'] = time.time()
                    logger.debug(f"Worker {model_id} is healthy")
                else:
                    logger.warning(f"Worker {model_id} health check failed: {response.status_code}")
                    failed_workers.append(model_id)
                    
            except requests.RequestException as e:
                logger.warning(f"Worker {model_id} health check error: {e}")
                failed_workers.append(model_id)

        for model_id in failed_workers:
            try:
                await self._restart_worker(model_id)
            except Exception as e:
                logger.error(f"Failed to restart worker {model_id}: {e}")
    
    async def _restart_worker(self, model_id: str):
        """Restart a failed worker with enhanced cleanup."""
        logger.info(f"Restarting worker {model_id}")
        
        if model_id in self.workers:

            await self._stop_worker(model_id)
            port = self.workers[model_id]['port']
            await self._wait_for_port_release(port)
            del self.workers[model_id]

        timeout = 180 if model_id in ["xtts", "cosyvoice"] else 120
        await self._start_worker(model_id, timeout)
    
    async def _stop_worker(self, model_id: str):
        """Stop a worker process with enhanced cleanup."""
        if model_id not in self.workers:
            logger.warning(f"Worker {model_id} not found")
            return
        
        worker_info = self.workers[model_id]
        process = worker_info['process']
        port = worker_info['port']
        
        try:
            requests.post(f"http://localhost:{port}/shutdown", timeout=5)
            process.wait(timeout=10)
            logger.info(f"Worker {model_id} stopped gracefully")
        except Exception:

            try:
                process.terminate()
                process.wait(timeout=5)
                logger.info(f"Worker {model_id} terminated")
            except Exception:
                process.kill()
                logger.warning(f"Worker {model_id} killed")
    
    async def _wait_for_port_release(self, port: int, timeout: int = 30):
        """Wait for port to be released after worker stops."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if await self._is_port_available(port):
                logger.info(f"Port {port} is now available")
                return True
            
            await asyncio.sleep(1)
        
        logger.warning(f"Port {port} did not release within {timeout}s")
        return False
    
    async def stop_all_workers(self):
        """Stop all worker processes with enhanced cleanup."""
        for model_id in list(self.workers.keys()):
            try:
                await self._stop_worker(model_id)
            except Exception as e:
                logger.error(f"Error stopping worker {model_id}: {e}")
        
        self.workers.clear()
        logger.info("All workers stopped")
    
    async def generate_audio(self, model_id: str, text: str, params: dict, filename_prefix: str, output_format: str = "wav") -> Optional[str]:
        """
        Generate audio using specified model and parameters.
        
        Args:
            model_id: The TTS model to use
            text: Text to synthesize
            params: Synthesis parameters
            filename_prefix: Prefix for the output filename
            output_format: Output audio format (wav, mp3, etc.)
            
        Returns:
            Path to the generated audio file
        """

        logger.debug(f"Processing text with length: {len(text)} characters")

        if model_id == "xtts" and len(text) > 400:
            logger.warning(f"Text length ({len(text)}) exceeds XTTS limit (400) - this may cause synthesis to fail!")
        
        if model_id == "cosyvoice":
            mapped_params = params.copy()
            mapped_params["mode"] = (params.get("mode") or 
                                   params.get("mode_checkbox_group") or 
                                   params.get("cosyvoice_mode") or 
                                   "预训练音色")
            mapped_params["speaker"] = (params.get("speaker") or 
                                       params.get("sft_dropdown") or 
                                       params.get("cosyvoice_speaker") or 
                                       "中文女")
            mapped_params["model_dropdown"] = (params.get("model_dropdown") or 
                                              params.get("cosyvoice_model_dropdown") or 
                                              "cosyvoice-300m-sft")
            mapped_params["prompt_text"] = params.get("prompt_text", "")
            mapped_params["instruct_text"] = params.get("instruct_text", "")
            mapped_params["speed"] = float(params.get("speed", 1.0))
            mapped_params["seed"] = int(params.get("seed", 0))
            mapped_params["stream"] = bool(params.get("stream", False))

            for wav_key in ("prompt_wav_upload", "prompt_wav_record"):
                val = (params.get(wav_key) or 
                       params.get(f"cosyvoice_{wav_key}") or
                       params.get(f"cosyvoice_prompt_wav") or 
                       None)
                
                if val is not None:
                    if hasattr(val, "name"):
                        val = val.name
                    elif isinstance(val, dict) and "name" in val:
                        val = val["name"]

                    if isinstance(val, str):
                        val = val.strip()
                        if val:
                            val = os.path.normpath(val)
                            if not os.path.exists(val):
                                logger.warning(f"Audio file not found: {val}")
                                val = None
                        else:
                            val = None
                    else:
                        val = None
                else:
                    val = None
                    
                mapped_params[wav_key] = val

            logger.info(f"CosyVoice参数映射完成")

            params = mapped_params
        
        if model_id not in self.workers:

            await self.start_workers([model_id])
            
        worker_url = self.get_worker_url(model_id)
        

        payload = {
            "text": text,
            "model_id": model_id,
            **params  
        }
        

        string_fields = ["instruct_text", "prompt_text", "zero_shot_spk_id"]
        for field in string_fields:
            if field in payload and payload[field] is None:
                payload[field] = ""

        if model_id == "kokoro":
            payload["use_gpu"] = params.get("use_gpu", False)

        if model_id == "xtts":
            payload["language"] = params.get("language", "zh-cn")

            if "speaker_wav" in params and params["speaker_wav"]:
                payload["speaker_wav"] = params["speaker_wav"]
            elif "speaker" in params and params["speaker"] and params["speaker"] != "default":
                payload["speaker_wav"] = params["speaker"]
            else:

                payload["speaker_wav"] = params.get("speaker_wav", None)
            
            payload["repetition_penalty"] = params.get("repetition_penalty", 2.0)
            payload["top_k"] = params.get("top_k", 50)
            payload["length_penalty"] = params.get("length_penalty", 1.0)

            payload["enable_text_splitting"] = params.get("enable_text_splitting", True)

            if "temperature" not in payload:
                payload["temperature"] = params.get("temperature", 0.7)
            if "top_p" not in payload:
                payload["top_p"] = params.get("top_p", 0.5)
            if "seed" not in payload:
                payload["seed"] = params.get("seed", 42)
            
            logger.info(f"XTTS specific parameters: enable_text_splitting={payload['enable_text_splitting']}, "
                       f"repetition_penalty={payload['repetition_penalty']}, top_k={payload['top_k']}, "
                       f"length_penalty={payload['length_penalty']}")

        speaker_info = params.get("speaker", "中文女")  
        if model_id == "xtts":
            speaker_info = params.get("speaker_wav", speaker_info)
        elif model_id == "cosyvoice":
            speaker_info = params.get("speaker", "中文女")  
        
        logger.info(f"Generating audio with {model_id}, speaker: {speaker_info}, text length: {len(text)}")
        logger.info(f"Text preview: '{text[:100]}...'")
        
        try:
            time = 240 if model_id == "cosyvoice" else 120
            
            response = requests.post(
                f"{worker_url}/synthesize",
                json=payload,
                timeout=time
            )
            logger.info(f"Response status code: {response.status_code}")

            if not response.ok:
                logger.error(f"Request failed with status {response.status_code}")
                logger.error(f"Response content: {response.text}")
                try:
                    error_data = response.json()
                    logger.error(f"Error response JSON: {error_data}")
                except:
                    logger.error("Response is not valid JSON")

                if response.status_code == 422:
                    logger.error("422 Unprocessable Entity - This usually indicates a validation error in the request data")
                    logger.error("Check if all required fields are present and have correct types")
                    
                response.raise_for_status()
            
            result = response.json()
            if not result.get("ok"):
                raise RuntimeError(f"Worker synthesis failed: {result.get('error', 'Unknown error')}")
            
            return result["path"]
            
        except requests.RequestException as e:
            logger.error(f"Worker request failed: {e}")
            if "timeout" in str(e).lower():
                logger.error(f"Worker request timed out. This might be due to a slow model initialization.")
            raise RuntimeError(f"Failed to communicate with {model_id} worker: {e}")

    def _postprocess(self, wav):
        """后处理音频数据"""
        wav = np.clip(wav, -1, 1)
        return wav
    
    async def call_worker_method(self, model_id: str, method_name: str, *args, **kwargs):
        """Call a specific method on a worker process."""
        try:

            if model_id not in self.workers:
                await self.start_workers([model_id])

            worker_info = self.workers[model_id]
            worker_url = worker_info["url"]

            method_url = f"{worker_url}/call_method"

            data = {
                "method_name": method_name,
                "args": args,
                "kwargs": kwargs
            }

            response = requests.post(method_url, json=data, timeout=30)
            response.raise_for_status()

            result = response.json()
            if result.get("ok", False):
                return result.get("result")
            else:
                raise RuntimeError(f"Worker method call failed: {result.get('error', 'Unknown error')}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling worker method {method_name} on {model_id}: {e}")
            raise RuntimeError(f"Failed to call worker method: {e}")
        except Exception as e:
            logger.error(f"Unexpected error calling worker method {method_name} on {model_id}: {e}")
            raise
    
    def get_worker_url(self, model_id: str) -> str:
        """Get the URL for a worker."""
        if model_id not in self.workers:
            raise ValueError(f"Worker for model {model_id} not found")
        return self.workers[model_id]["url"]

