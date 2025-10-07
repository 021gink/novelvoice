"""
System utilities and helpers.
"""
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List
import psutil

class SystemInfo:
    """System information and utilities."""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get comprehensive system information."""
        return {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "disk_usage": SystemInfo.get_disk_usage(),
            "gpu_info": SystemInfo.get_gpu_info()
        }
    
    @staticmethod
    def get_disk_usage() -> Dict[str, float]:
        """Get disk usage information."""
        usage = psutil.disk_usage('/')
        return {
            "total_gb": round(usage.total / (1024**3), 2),
            "used_gb": round(usage.used / (1024**3), 2),
            "free_gb": round(usage.free / (1024**3), 2),
            "percent": usage.percent
        }
    
    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """Get GPU information."""
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    "cuda_available": True,
                    "cuda_version": torch.version.cuda,
                    "gpu_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name()
                }
            else:
                return {"cuda_available": False}
        except ImportError:
            return {"cuda_available": False, "error": "PyTorch not available"}
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage."""
        memory = psutil.virtual_memory()
        return {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "percent": memory.percent
        }
    
    @staticmethod
    def get_cpu_usage() -> Dict[str, float]:
        """Get current CPU usage."""
        return {
            "percent": psutil.cpu_percent(interval=1),
            "per_cpu": psutil.cpu_percent(interval=1, percpu=True)
        }

class ProcessManager:
    """Process management utilities."""
    
    @staticmethod
    def find_processes_by_name(name: str) -> List[Dict[str, Any]]:
        """Find processes by name."""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if name.lower() in proc.info['name'].lower():
                    processes.append({
                        "pid": proc.info['pid'],
                        "name": proc.info['name'],
                        "cmdline": proc.info['cmdline']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return processes
    
    @staticmethod
    def kill_process(pid: int) -> bool:
        """Kill process by PID."""
        try:
            process = psutil.Process(pid)
            process.terminate()
            return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
    
    @staticmethod
    def kill_processes_by_name(name: str) -> int:
        """Kill all processes matching name."""
        killed = 0
        for proc in ProcessManager.find_processes_by_name(name):
            if ProcessManager.kill_process(proc["pid"]):
                killed += 1
        return killed
    
    @staticmethod
    def is_port_available(port: int) -> bool:
        """Check if port is available."""
        for conn in psutil.net_connections():
            if conn.laddr.port == port:
                return False
        return True
    
    @staticmethod
    def get_process_by_port(port: int) -> Dict[str, Any]:
        """Get process using specific port."""
        for conn in psutil.net_connections():
            if conn.laddr.port == port and conn.pid:
                try:
                    proc = psutil.Process(conn.pid)
                    return {
                        "pid": conn.pid,
                        "name": proc.name(),
                        "cmdline": proc.cmdline()
                    }
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        return {}

class FileUtils:
    """File and directory utilities."""
    
    @staticmethod
    def ensure_directory(path: str) -> Path:
        """Ensure directory exists, create if not."""
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        return directory
    
    @staticmethod
    def get_directory_size(path: str) -> int:
        """Get total size of directory in bytes."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size
    
    @staticmethod
    def clean_directory(path: str, max_age_days: int = 7) -> int:
        """Clean old files from directory."""
        import time
        cleaned = 0
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        
        for item in Path(path).iterdir():
            if item.is_file() and item.stat().st_mtime < cutoff_time:
                try:
                    item.unlink()
                    cleaned += 1
                except OSError:
                    continue
        return cleaned
    
    @staticmethod
    def copy_file(src: str, dst: str) -> bool:
        """Copy file with error handling."""
        try:
            shutil.copy2(src, dst)
            return True
        except (IOError, OSError):
            return False
    
    @staticmethod
    def move_file(src: str, dst: str) -> bool:
        """Move file with error handling."""
        try:
            shutil.move(src, dst)
            return True
        except (IOError, OSError):
            return False

class NetworkUtils:
    """Network utilities."""
    
    @staticmethod
    def find_available_port(start_port: int = 8000, max_port: int = 9000) -> int:
        """Find an available port in range."""
        for port in range(start_port, max_port + 1):
            if ProcessManager.is_port_available(port):
                return port
        raise RuntimeError(f"No available ports in range {start_port}-{max_port}")
    
    @staticmethod
    def check_url(url: str, timeout: int = 5) -> bool:
        """Check if URL is accessible."""
        import requests
        try:
            response = requests.get(url, timeout=timeout)
            return response.status_code == 200
        except requests.RequestException:
            return False

class RequirementsChecker:
    """Check system requirements."""
    
    @staticmethod
    def check_python_version(min_version: str = "3.8") -> bool:
        """Check if Python version meets minimum requirement."""
        import sys
        current = sys.version_info
        required = tuple(map(int, min_version.split('.')))
        return current >= required
    
    @staticmethod
    def check_disk_space(min_gb: float = 5.0) -> bool:
        """Check if enough disk space is available."""
        usage = psutil.disk_usage('/')
        free_gb = usage.free / (1024**3)
        return free_gb >= min_gb
    
    @staticmethod
    def check_memory(min_gb: float = 4.0) -> bool:
        """Check if enough memory is available."""
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        return available_gb >= min_gb


