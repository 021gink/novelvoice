import logging
import time
import torch
import traceback
from functools import wraps
from typing import Callable, Optional

MAX_VAL = 0.8  # 保持与 worker_engine.py 一致

logger = logging.getLogger("CosyVoice")

def log_execution(func: Callable) -> Callable:
    """记录函数执行入口和出口"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Entering {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Exiting {func.__name__} successfully")
            return result
        except Exception as e:
            logger.debug(f"Exiting {func.__name__} with error: {e}")
            raise
    return wrapper

def safe_exception(func: Callable) -> Callable:
    """捕获异常并记录详细上下文"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (RuntimeError, ValueError, FileNotFoundError, AttributeError, NameError) as e:
            logger.warning(f"{func.__name__} expected error: {type(e).__name__}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"{func.__name__} failed: {e}") from e
    return wrapper

def memory_monitor(func: Callable) -> Callable:
    """监控 GPU 内存使用"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            return func(*args, **kwargs)
        initial = torch.cuda.memory_allocated() / 1024**2
        try:
            result = func(*args, **kwargs)
            final = torch.cuda.memory_allocated() / 1024**2
            delta = final - initial
            if delta > 100:
                logger.warning(f"{func.__name__} increased GPU memory by {delta:.1f}MB")
            return result
        except Exception as e:
            final = torch.cuda.memory_allocated() / 1024**2
            delta = final - initial
            logger.error(f"{func.__name__} failed after GPU memory increase of {delta:.1f}MB: {e}")
            raise
    return wrapper

def performance_monitor(threshold_ms: int = 1000) -> Callable:
    """记录函数执行时间"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.time() - start) * 1000
                if elapsed_ms > threshold_ms:
                    logger.warning(f"{func.__name__} took {elapsed_ms:.2f}ms (threshold {threshold_ms}ms)")
                return result
            except Exception as e:
                elapsed_ms = (time.time() - start) * 1000
                logger.error(f"{func.__name__} failed after {elapsed_ms:.2f}ms: {e}")
                raise
        return wrapper
    return decorator
