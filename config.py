"""
Configuration file for multi-engine TTS isolation system.
Defines ports, model paths, sampling rates, and other system parameters.
"""
import os
from pathlib import Path
from typing import Dict, List

# Base paths
BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
WORKERS_DIR = BASE_DIR / "workers"

# Ensure directories exist
OUTPUTS_DIR.mkdir(exist_ok=True)
WORKERS_DIR.mkdir(exist_ok=True)

# Worker configuration
WORKER_BASE_PORT = 8000
WORKER_PORT_RANGE = range(WORKER_BASE_PORT, WORKER_BASE_PORT + 100)

# Adding missing variable definitions
RESOURCE_LIMITS = {
    "max_concurrent_jobs": 4,
    "max_workers_per_model": 2,
    "max_text_length": 5000
}

MODEL_LIMITS = {
    "xtts": 380,
    "kokoro": 170,
    "chattts": 100,
    "cosyvoice": 380
}

# Audio settings
AUDIO_SETTINGS = {
    "sample_rate": 24000,
    "bit_depth": 16,
    "channels": 1,
    "output_formats": ["wav", "mp3"],
    "default_output_format": "mp3",  
    "mp3_bitrate": "320k",  
    "silence_between_fragments_ms": 300
}

# Resource management
RESOURCE_LIMITS = {
    "max_concurrent_jobs": 4,
    "max_workers_per_model": 2,
    "max_text_length": 5000,
    "max_concurrent_workers": 3,
    "cpu_usage_threshold": 90,
    "idle_timeout_seconds": 300,
    "health_check_interval": 30
}

# Ebook processing
EBOOK_SETTINGS = {
    "supported_formats": [".epub", ".mobi", ".pdf", ".txt"],
    "chapter_detection_patterns": [
        # 英文章节格式
        r"^#+\s*Chapter\s+\d+",
        r"^#+\s*\d+\.\s+",
        r"^Chapter\s+\d+",
        r"^\d+\.\s+[A-Z]",
        # 中文章节格式
        r"^#+\s*第[一二三四五六七八九十百千万]+[章回节]",
        r"^#+\s*\d+[\.、]\s+[\u4e00-\u9fa5]",
        r"^第[一二三四五六七八九十百千万]+[章回节]",
        r"^[一二三四五六七八九十]+[\.、]\s+[\u4e00-\u9fa5]",
        r"^第\d+[章回节]",
        r"^\d+章\s+[\u4e00-\u9fa5]"
    ],
    "max_overlap_chars": 30
}

# Logging
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard"
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": "app.log",
            "level": "DEBUG",
            "formatter": "standard",
            "encoding": "utf-8"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"]
    }
}

