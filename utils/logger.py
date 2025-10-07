"""
Logging configuration and utilities.
"""
import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file, if None logs only to console
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
    """
    # Ensure logs directory exists
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set third-party loggers to WARNING to reduce noise
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)

class ProgressLogger:
    """Logger that also tracks progress for UI updates."""
    
    def __init__(self, logger_name: str):
        self.logger = get_logger(logger_name)
        self.progress_callback = None
        self.current_step = 0
        self.total_steps = 100
    
    def set_progress_callback(self, callback):
        """Set progress callback function."""
        self.progress_callback = callback
    
    def set_total_steps(self, total: int):
        """Set total number of steps."""
        self.total_steps = max(total, 1)
    
    def update_progress(self, step: int, message: str = ""):
        """Update progress and log message."""
        self.current_step = step
        progress = min(step / self.total_steps, 1.0)
        
        if self.progress_callback:
            self.progress_callback(progress, message)
        
        if message:
            self.logger.info(message)
    
    def increment_progress(self, message: str = ""):
        """Increment progress by one step."""
        self.current_step += 1
        self.update_progress(self.current_step, message)
    
    def log(self, level: str, message: str):
        """Log message at specified level."""
        getattr(self.logger, level.lower())(message)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)


def safe_exception(logger: logging.Logger):
    """Decorator to safely log exceptions with traceback (only in DEBUG)."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{func.__name__} | {type(e).__name__} | {e}")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.error(traceback.format_exc())
                raise
        return wrapper
    return decorator
