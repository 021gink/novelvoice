# # =============================================================================
# # 位置: workers/cosyvoice_engine.py (1-549)
# # 作用: CosyVoice TTS引擎核心实现，支持多模式语音合成与模型热加载
# # 依赖: 调用: torch(张量计算)/soundfile(音频IO)/Matcha-TTS(语音处理) ; 被调用: orchestrator(任务调度)
# # 概念: 单例模式(中级)、CUDA显存管理(高级)、流式推理(高级)
# # 关键词: 语音合成, 模型热加载, 多线程安全, 显存优化
# # =============================================================================
# # workers/cosyvoice_engine.py (Fixed version based on app.py)
# #!/usr/bin/env python3
# """
# CosyVoice Engine for TTS synthesis - Fixed version based on app.py perfect implementation
# """

# # =============================================================================
# # 位置: 标准库导入 (8-15)
# # 作用: 系统基础功能与异步编程支持
# # 依赖: 调用: Python标准库 ; 被调用: 全局模块
# # 概念: 标准库导入(初级)、异步编程(中级)、路径处理(初级)
# # 关键词: 系统库, 异步IO, 路径操作, 类型提示
# # =============================================================================
# import logging
# import sys
# import os
# from pathlib import Path
# import time
# import json
# import asyncio
# from typing import Dict, Any, Optional

# # =============================================================================
# # 位置: 第三方库导入 (16-19)
# # 作用: 科学计算与深度学习框架导入
# # 依赖: 调用: PyTorch/NumPy/Random ; 被调用: 全局模块
# # 概念: 张量计算(中级)、随机数生成(初级)、数组操作(中级)
# # 关键词: 深度学习, 张量操作, 随机种子, 数值计算
# # =============================================================================
# import torch
# import numpy as np
# import random

# # =============================================================================
# # 位置: 路径配置 (20-25)
# # 作用: Matcha-TTS依赖库路径动态添加
# # 依赖: 调用: sys.path(系统路径) ; 被调用: Matcha-TTS导入
# # 概念: 动态路径(中级)、模块导入(中级)、第三方依赖(中级)
# # 关键词: 路径注入, 依赖管理, 模块加载
# # =============================================================================
# # 添加Matcha-TTS路径到系统路径
# sys.path.insert(0, str(Path(__file__).parent.parent))
# matcha_path = str(Path(__file__).parent.parent / "third_party" / "Matcha-TTS")
# if matcha_path not in sys.path:
#     sys.path.insert(0, matcha_path)

# # =============================================================================
# # 位置: 日志配置 (27-29)
# # 作用: 全局日志系统初始化
# # 依赖: 调用: logging(日志库) ; 被调用: 全局模块
# # 概念: 日志系统(初级)、级别配置(初级)、全局实例(初级)
# # 关键词: 日志初始化, 级别设置, 全局记录器
# # =============================================================================
# # 配置日志
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # =============================================================================
# # 位置: CosyVoice模块导入 (31-43)
# # 作用: 核心TTS引擎与音频处理工具导入
# # 依赖: 调用: CosyVoice(语音合成)/Matcha-TTS(音频处理) ; 被调用: 全局模块
# # 概念: 条件导入(中级)、异常处理(中级)、模块状态(中级)
# # 关键词: 语音合成, 音频处理, 条件导入, 异常安全
# # =============================================================================
# try:
#     from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
#     from cosyvoice.utils.file_utils import load_wav
#     from cosyvoice.utils.common import set_all_random_seed
    
#     # 尝试导入Matcha-TTS
#     try:
#         import matcha.utils.audio as matcha_audio
#     except ImportError as e:
#         logger.warning(f"无法导入 matcha.utils.audio: {e}")
#         matcha_audio = None
        
#     COSYVOICE_AVAILABLE = True
#     logger.info("CosyVoice engine initialized successfully")
# except ImportError as e:
#     COSYVOICE_AVAILABLE = False
#     logger.error(f"CosyVoice import error: {e}")
#     CosyVoice = None
#     CosyVoice2 = None
#     matcha_audio = None

# # =============================================================================
# # 位置: ASR模块导入 (44-46)
# # 作用: 语音识别模型导入
# # 依赖: 调用: funasr(语音识别) ; 被调用: asr_prompt_wav_recognition
# # 概念: 语音识别(中级)、条件导入(中级)、异常处理(中级)
# # 关键词: 语音识别, 模型导入, 异常安全
# # =============================================================================
# try:
#     from funasr import AutoModel
#     ASR_AVAILABLE = True
# except ImportError as e:
#     ASR_AVAILABLE = False
#     logger.warning(f"ASR import error: {e}")
#     AutoModel = None

# # =============================================================================
# # 位置: 音频处理库导入 (47-48)
# # 作用: 音频文件读写支持库导入
# # 依赖: 调用: soundfile(音频IO) ; 被调用: 音频保存
# # 概念: 音频IO(中级)、文件格式(初级)、采样率(中级)
# # 关键词: 音频保存, 文件IO, 采样率处理
# # =============================================================================
# import soundfile as sf

# # =============================================================================
# # 位置: 全局常量定义 (55-65)
# # 作用: 音频处理参数配置与推理模式枚举定义
# # 依赖: 调用: 无 ; 被调用: synthesize(参数验证)/postprocess(音频处理)
# # 概念: 常量配置(初级)、枚举类型(中级)
# # 关键词: 采样率, 音频长度, 推理模式
# # =============================================================================
# # 常量定义 - 基于app.py
# PROMPT_SR = 16000
# TARGET_SR = 24000
# MAX_PROMPT_SEC = 30
# MAX_TEXT_LEN = 200
# MAX_VAL = 0.8

# inference_mode_list = ['预训练音色', '3s极速复刻', '跨语种复刻', '自然语言控制']
# instruct_dict = {
#     '预训练音色': '1. 选择预训练音色\n2. 点击生成音频按钮',
#     '3s极速复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 输入prompt文本\n3. 点击生成音频按钮',
#     '跨语种复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 点击生成音频按钮',
#     '自然语言控制': '1. 选择预训练音色\n2. 输入instruct文本\n3. 点击生成音频按钮'
# }

# # =============================================================================
# # 位置: 模型管理变量 (68-72)
# # 作用: 全局模型状态管理与线程安全控制
# # 依赖: 调用: threading(线程锁) ; 被调用: load_cosyvoice_model(模型加载)
# # 概念: 单例模式(高级)、线程安全(高级)、显存管理(高级)
# # 关键词: 全局状态, 线程锁, 模型缓存
# # =============================================================================
# # 模型管理变量 - 基于app.py的全局模型管理
# import threading
# _model_lock = threading.Lock()
# _current_model = None
# _current_model_path = None
# _current_model_mode = None
# _feat_ready = False

# # =============================================================================
# # 位置: Matcha-TTS工具函数 (74-120)
# # 作用: Matcha音频处理库的窗口与梅尔谱缓存管理
# # 依赖: 调用: torch(张量操作)/matcha_audio(音频处理) ; 被调用: reset_feat_extractor(特征提取器重置)
# # 概念: 傅里叶变换(高级)、梅尔滤波器组(高级)、设备感知(中级)
# # 关键词: hann_window, mel_basis, 设备同步
# # =============================================================================
# def force_reset_matcha_window(device, win_length=1920):
#     """
#     强制为 matcha 中对应 device 的 hann_window 生成正确大小并放到 device。
#     """
#     if matcha_audio is None:
#         logger.warning("matcha_audio 未导入，无法重置 hann_window")
#         return

#     try:
#         device_str = str(device)
#         hw = torch.hann_window(win_length).to(device)
#         if hasattr(matcha_audio, "hann_window"):
#             try:
#                 matcha_audio.hann_window[device_str] = hw
#             except Exception:
#                 matcha_audio.hann_window = {device_str: hw}
#         else:
#             matcha_audio.hann_window = {device_str: hw}
#         logger.info(f"[INFO] Matcha hann_window 已重置为 size={win_length} @ {device}")
#     except Exception as e:
#         logger.exception(f"[WARN] 重置 matcha hann_window 失败: {e}")

# def reset_matcha_mel_basis(n_fft, num_mels, sampling_rate, fmin, fmax, device):
#     """重置Matcha mel_basis"""
#     if matcha_audio is None:
#         logger.warning("matcha_audio 未导入，无法重置 mel_basis")
#         return
        
#     try:
#         key = f"{str(fmax)}_{str(device)}"
#         if hasattr(matcha_audio, "mel_basis") and key in matcha_audio.mel_basis:
#             del matcha_audio.mel_basis[key]
#     except Exception as e:
#         logger.warning(f"重置 mel_basis 失败: {e}")

# # =============================================================================
# # 位置: 设备推断函数 (122-175)
# # 作用: 自动推断PyTorch模型所在设备（CPU/GPU）
# # 依赖: 调用: torch.nn(神经网络模块) ; 被调用: reset_feat_extractor(特征提取器初始化)
# # 概念: 设备探测(中级)、参数遍历(中级)、异常处理(中级)
# # 关键词: 设备推断, 参数检查, 模块遍历
# # =============================================================================
# def get_model_device(model):
#     """
#     尝试从 model 或其常见子模块推断设备（返回 torch.device）。
#     兼容 CosyVoice 封装（可能不是 nn.Module 本身）。
#     """
#     try:
#         import torch.nn as nn

#         # 直接属性 device
#         try:
#             if hasattr(model, "device"):
#                 d = getattr(model, "device")
#                 if isinstance(d, torch.device):
#                     return d
#                 else:
#                     return torch.device(str(d))
#         except Exception:
#             pass

#         # 常见子模块名
#         candidates = ('hift', 'frontend', 'llm', 'flow', 'encoder', 'decoder', 'acoustic')
#         for name in candidates:
#             try:
#                 sub = getattr(model, name, None)
#                 if sub is None:
#                     continue
#                 if isinstance(sub, nn.Module):
#                     try:
#                         params = list(sub.parameters())
#                         if params:
#                             return params[0].device
#                     except Exception:
#                         pass
#             except Exception:
#                 pass

#         # 最后在 model.__dict__ 中搜索第一个 nn.Module
#         try:
#             for v in model.__dict__.values():
#                 if isinstance(v, nn.Module):
#                     try:
#                         params = list(v.parameters())
#                         if params:
#                             return params[0].device
#                     except Exception:
#                         pass
#         except Exception:
#             pass

#         # 回退 CPU
#         return torch.device("cpu")
#     except Exception as e:
#         logger.warning(f"获取模型设备失败，回退到CPU: {e}")
#         return torch.device("cpu")

# # =============================================================================
# # 位置: 特征提取器重置函数 (177-235)
# # 作用: 统一管理Matcha-TTS与CosyVoice的音频特征提取缓存
# # 依赖: 调用: force_reset_matcha_window(窗口重置)/reset_matcha_mel_basis(梅尔谱重置) ; 被调用: load_cosyvoice_model(模型加载)
# # 概念: 缓存一致性(高级)、跨库协同(高级)、设备同步(中级)
# # 关键词: 特征提取, 缓存管理, 窗口同步
# # =============================================================================
# def reset_feat_extractor(model):
#     """
#     重置 feat_extractor 相关缓存，并刷新 Matcha-TTS 全局 hann_window 和 mel_basis
#     基于app.py的实现
#     """
#     global _feat_ready
#     _feat_ready = False

#     try:
#         if hasattr(model, 'frontend') and hasattr(model.frontend, 'feat_extractor'):
#             feat = model.frontend.feat_extractor

#             # 删除/清理旧 window 和缓存（兼容性）
#             try:
#                 if hasattr(feat, 'window'):
#                     feat.window = None
#             except Exception:
#                 pass
#             try:
#                 if hasattr(feat, '_cached_window'):
#                     feat._cached_window = None
#             except Exception:
#                 pass

#             # 尝试从 feat 提取 win_length/n_fft
#             win_length = None
#             try:
#                 if hasattr(feat, 'win_length') and getattr(feat, 'win_length') is not None:
#                     win_length = int(getattr(feat, 'win_length'))
#                 elif hasattr(feat, 'n_fft') and getattr(feat, 'n_fft') is not None:
#                     win_length = int(getattr(feat, 'n_fft'))
#                 elif hasattr(feat, 'fft_size') and getattr(feat, 'fft_size') is not None:
#                     win_length = int(getattr(feat, 'fft_size'))
#             except Exception:
#                 win_length = None

#             if win_length is None:
#                 win_length = 1920  # 默认值

#             # 推断模型设备
#             device = get_model_device(model)

#             # 强制重置 Matcha hann_window
#             force_reset_matcha_window(device, win_length=win_length)

#             # 重置 mel_basis
#             reset_matcha_mel_basis(
#                 n_fft=win_length,
#                 num_mels=80,           # CosyVoice 默认
#                 sampling_rate=16000,   # prompt 音频采样率
#                 fmin=0,
#                 fmax=8000,
#                 device=device
#             )

#             logger.info("feat_extractor 初始化完成，Matcha window 和 mel_basis 已统一重置")
#             _feat_ready = True
#         else:
#             logger.info("模型无 frontend/feat_extractor，跳过 feat 初始化")
#             _feat_ready = True
#     except Exception as e:
#         logger.exception(f"reset_feat_extractor 出错: {e}")
#         # 为了不阻塞使用者，降级处理
#         _feat_ready = True

# # =============================================================================
# # 位置: 模型类推断函数 (237-252)
# # 作用: 根据模型路径自动选择CosyVoice/CosyVoice2模型类
# # 依赖: 调用: 无 ; 被调用: load_cosyvoice_model(模型加载)/validate_inputs(输入验证)
# # 概念: 路径解析(中级)、模型版本检测(中级)、异常处理(中级)
# # 关键词: 模型选择, 路径匹配, 版本检测
# # =============================================================================
# def get_model_class(model_path):
#     """根据模型路径确定模型类 - 采用app.py的判断逻辑"""
#     try:
#         model_path_str = str(model_path).lower()
#         logger.info(f"Determining model class for path: {model_path_str}")
        
#         # 优先通过路径关键词判断
#         if 'cosyvoice2' in model_path_str:
#             logger.info("Selected CosyVoice2 model class")
#             return CosyVoice2
        
#         # 检查是否为sft/instruct模型
#         if any(kw in model_path_str for kw in ['sft', 'instruct']):
#             logger.warning(f"检测到SFT模型路径但未配置自然语言控制模式: {model_path}")
            
#         logger.info("Selected CosyVoice model class")
#         return CosyVoice
#     except Exception as e:
#         logger.exception(f"模型类判断异常: {e}")
#         return CosyVoice

# # =============================================================================
# # 位置: 模型加载函数 (254-294)
# # 作用: 实现模型热切换与显存优化加载（支持FP16/JIT/TRT加速）
# # 依赖: 调用: get_model_class(模型选择)/reset_feat_extractor(特征重置) ; 被调用: synthesize(推理入口)
# # 概念: 双检锁模式(高级)、显存碎片整理(高级)、懒加载(中级)
# # 关键词: 模型单例, 显存释放, 线程安全
# # =============================================================================
# def load_cosyvoice_model(model_path, mode=None, load_jit=False, load_trt=False, fp16=False):
#     """
#     加载 CosyVoice 模型或刷新模式参数 - 基于app.py的完美实现
#     """
#     global _current_model, _current_model_path, _current_model_mode
#     with _model_lock:
#         # 模型已加载且路径相同
#         if _current_model_path == model_path:
#             if _current_model_mode != mode:
#                 _current_model_mode = mode
#                 reset_feat_extractor(_current_model)
#             return _current_model

#         # 模型切换 → 释放旧模型
#         if _current_model is not None:
#             try:
#                 del _current_model
#             except Exception:
#                 pass
#             _current_model = None
#             _current_model_path = None
#             _current_model_mode = None
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()

#         # 加载新模型
#         ModelClass = get_model_class(model_path)
#         model = ModelClass(model_path, load_jit=load_jit, load_trt=load_trt, fp16=fp16)
#         reset_feat_extractor(model)

#         _current_model = model
#         _current_model_path = model_path
#         _current_model_mode = mode
#         return _current_model

# # =============================================================================
# # 位置: 模型释放函数 (296-310)
# # 作用: 安全卸载模型并清理CUDA显存资源
# # 依赖: 调用: torch.cuda(显存管理) ; 被调用: 外部清理流程
# # 概念: 资源释放(中级)、显存回收(高级)、异常安全(中级)
# # 关键词: 显存清理, 模型卸载, 资源回收
# # =============================================================================
# def release_cosyvoice_model():
#     """
#     卸载当前模型，释放显存 - 基于app.py
#     """
#     global _current_model, _current_model_path, _current_model_mode
#     with _model_lock:
#         if _current_model is not None:
#             try:
#                 del _current_model
#             except Exception:
#                 pass
#             _current_model = None
#             _current_model_path = None
#             _current_model_mode = None
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()

# # =============================================================================
# # 位置: 音色列表获取函数 (312-320)
# # 作用: 从模型文件加载预训练音色信息
# # 依赖: 调用: torch.load(模型加载) ; 被调用: 外部UI组件
# # 概念: 模型元数据(中级)、文件IO(初级)、字典操作(初级)
# # 关键词: 音色管理, 模型元数据, 文件读取
# # =============================================================================
# def get_sft_speakers(model_path):
#     """获取预训练音色列表 - 基于app.py"""
#     spk2info_path = os.path.join(model_path, "spk2info.pt")
#     if os.path.exists(spk2info_path):
#         info = torch.load(spk2info_path, map_location="cpu")
#         return list(info.keys())
#     return []

# # =============================================================================
# # 位置: 输入验证函数 (322-342)
# # 作用: 验证模型路径与推理模式的兼容性
# # 依赖: 调用: get_model_class(模型类型检测) ; 被调用: synthesize(参数校验)
# # 概念: 输入验证(中级)、文件存在性检查(初级)、模式兼容(中级)
# # 关键词: 参数校验, 文件检查, 模式验证
# # =============================================================================
# def validate_inputs(text, mode, prompt_wav, prompt_text, instruct_text, model_path):
#     # 基于app.py的完整校验逻辑
#     logger.info(f"Validating inputs: text={text[:50]}..., mode={mode}, prompt_wav={prompt_wav}, prompt_text={prompt_text[:50] if prompt_text else ''}, instruct_text={instruct_text[:50] if instruct_text else ''}, model_path={model_path}")
    
#     ModelClass = get_model_class(model_path)
#     logger.info(f"Model class: {ModelClass}")
    
#     # 模型与推理模式兼容性检查
#     if ModelClass == CosyVoice2:
#         valid_modes = ['预训练音色']
#         if mode not in valid_modes:
#             error_msg = f"CosyVoice2仅支持{valid_modes}模式"
#             logger.error(error_msg)
#             raise ValueError(error_msg)
#     else:
#         if 'sft' in str(model_path).lower() and mode != '自然语言控制':
#             error_msg = "SFT模型需使用自然语言控制模式"
#             logger.error(error_msg)
#             raise ValueError(error_msg)
#         elif mode == '自然语言控制' and 'sft' not in str(model_path).lower():
#             error_msg = "自然语言控制需要SFT模型"
#             logger.error(error_msg)
#             raise ValueError(error_msg)

#     # 路径有效性检查
#     if not os.path.isdir(model_path):
#         error_msg = f"模型目录无效: {model_path}"
#         logger.error(error_msg)
#         raise FileNotFoundError(error_msg)
    
#     # 配置文件存在性检查
#     # 根据模型类型检查不同的必需文件
#     ModelClass = get_model_class(model_path)
#     if ModelClass == CosyVoice2:
#         # CosyVoice2需要的文件
#         required_files = ['llm.pt', 'flow.pt', 'hift.pt', 'cosyvoice2.yaml']
#     else:
#         # CosyVoice需要的文件
#         required_files = ['llm.pt', 'flow.pt', 'hift.pt', 'cosyvoice.yaml']
    
#     for f in required_files:
#         file_path = os.path.join(model_path, f)
#         if not os.path.exists(file_path):
#             error_msg = f"模型文件缺失: {f} at {file_path}"
#             logger.error(error_msg)
#             raise FileNotFoundError(error_msg)
    
#     logger.info("Input validation passed")
#     return True, None


# # =============================================================================
# # 位置: L2-配置层 - 项目根目录配置 (新增)
# # 作用: 定义项目根目录路径，支持相对路径计算
# # 依赖: 调用: Path(__file__) ; 被调用: MODELS_DIR等路径计算
# # 概念: 路径管理/相对路径/项目结构（难度：初级）
# # 关键词: 项目根目录,路径计算,相对路径
# # =============================================================================
# BASE_DIR = Path(__file__).parent.parent  # 项目根目录

# # =============================================================================
# # 位置: L2-配置层 - 模型路径配置 (新增)
# # 作用: 定义模型存储目录路径，支持模型文件的统一管理
# # 依赖: 调用: Path ; 被调用: asr_prompt_wav_recognition, CosyVoiceWorker.initialize
# # 概念: 路径管理/资源定位/文件系统（难度：初级）
# # 关键词: 模型路径,资源定位,文件系统
# # =============================================================================
# MODELS_DIR = BASE_DIR / "models"          # 模型存储

# # =============================================================================
# # 位置: L2-配置层 - 输出路径配置 (新增)
# # 作用: 定义音频文件输出路径，支持自动创建目录结构
# # 依赖: 调用: Path/Path.mkdir方法 ; 被调用: synthesize方法
# # 概念: 路径管理/目录创建/文件组织（难度：初级）
# # 关键词: 输出路径,目录创建,文件组织
# # =============================================================================
# # 将OUTPUT_DIR定义移至函数内部，避免模块导入时的路径问题
# # OUTPUT_DIR = Path("outputs") / "temp"
# # OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # =============================================================================
# # 位置: CosyVoiceWorker类 (381-515)
# # 作用: 多模式推理流程调度器（支持预训练/极速复刻/跨语种/自然语言控制模式）
# # 依赖: 调用: load_cosyvoice_model(模型加载)/postprocess(后处理) ; 被调用: orchestrator(任务调度)
# # 概念: 策略模式(中级)、流式拼接(高级)、参数路由(中级)
# # 关键词: 多模态推理, 参数路由, 流式处理
# # =============================================================================
# class CosyVoiceWorker:
#     def __init__(self, models_dir: str = "models"):
#         self.models_dir = Path(models_dir)

#     # =============================================================================
#     # 位置: CosyVoiceWorker.initialize方法 (516-518)
#     # 作用: Worker初始化方法（延迟加载模型）
#     # 依赖: 调用: 无 ; 被调用: orchestrator(任务调度)
#     # 概念: 延迟加载(中级)、异步初始化(中级)
#     # 关键词: 延迟加载, 异步初始化, Worker生命周期
#     # =============================================================================
#     async def initialize(self):
#         """初始化worker - 延迟加载模型"""
#         return

#     # =============================================================================
#     # 位置: CosyVoiceWorker.synthesize方法 (520-629)
#     # 作用: 多模式语音合成核心流程（参数兼容/模型加载/推理执行/音频后处理）
#     # 依赖: 调用: load_cosyvoice_model(模型加载)/postprocess(后处理)/validate_inputs(参数验证) ; 被调用: orchestrator(任务调度)
#     # 概念: 策略模式(中级)、流式处理(高级)、参数路由(中级)
#     # 关键词: 多模态推理, 参数兼容, 流式输出, 音频保存
#     # =============================================================================
#     async def synthesize(self, text: str, **kwargs):
#         """
#         合成语音 - 统一参数处理版本
#         """
#         global _feat_ready
#         try:
#             # 提取参数并设置默认值
#             mode = kwargs.get("mode", "预训练音色")
#             speaker = kwargs.get("speaker", "中文女")
#             prompt_wav = kwargs.get("prompt_wav_upload") or kwargs.get("prompt_wav_record")
#             prompt_text = kwargs.get("prompt_text", "")
#             instruct_text = kwargs.get("instruct_text", "")
#             stream = kwargs.get("stream", False)
#             speed = float(kwargs.get("speed", 1.0))
#             seed = int(kwargs.get("seed", 0))
#             model_dropdown = kwargs.get("model_dropdown") or "cosyvoice-300m-sft"
            
#             logger.info(f"Synthesizing with mode: {mode}, speaker: {speaker}, text length: {len(text)}")
#             logger.info(f"All kwargs: {kwargs}")
            
#             # 验证文本长度
#             if len(text) > MAX_TEXT_LEN:
#                 logger.warning(f"Text length {len(text)} exceeds recommended limit {MAX_TEXT_LEN}, truncating")
#                 text = text[:MAX_TEXT_LEN]

#             # 确保 prompt_wav 是字符串路径
#             logger.info(f"Processing prompt_wav: {prompt_wav}, type: {type(prompt_wav)}")
#             if prompt_wav and hasattr(prompt_wav, "name"):
#                 prompt_wav = prompt_wav.name
#                 logger.info(f"prompt_wav after name extraction: {prompt_wav}")
#             if prompt_wav:
#                 prompt_wav = os.path.normpath(str(prompt_wav))
#                 logger.info(f"prompt_wav after normpath: {prompt_wav}")
#                 if not os.path.exists(prompt_wav):
#                     raise FileNotFoundError(f"Prompt audio file not found: {prompt_wav}")

#             # 模型路径处理
#             model_path = str(self.models_dir / model_dropdown)
#             logger.info(f"Model path: {model_path}")
#             if not os.path.exists(model_path):
#                 raise FileNotFoundError(f"Model directory not found: {model_path}")

#             # 验证输入参数
#             logger.info("Validating inputs...")
#             ok, err = validate_inputs(text, mode, prompt_wav, prompt_text, instruct_text, model_path)
#             if not ok:
#                 raise ValueError(err)

#             # 加载模型
#             logger.info("Loading model...")
#             try:
#                 cosyvoice = load_cosyvoice_model(model_path, mode=mode)
#             except Exception as e:
#                 raise RuntimeError(f"模型加载失败: {e}")

#             # 检查 window 是否已重置
#             if not _feat_ready:
#                 raise RuntimeError("等待 window 重置，请稍后再试（正在初始化 feat_extractor / matcha window）")

#             # 设置随机种子
#             set_all_random_seed(seed if seed else random.randint(1, 100000000))

#             # 根据模式执行推理 - 基于app.py的完美逻辑
#             logger.info("Starting inference...")
#             try:
#                 if mode == '预训练音色':
#                     logger.info("Using inference_sft mode")
#                     inference_method = cosyvoice.inference_sft
#                     method_args = [text, speaker]

#                 elif mode == '3s极速复刻':
#                     logger.info("Using inference_zero_shot mode")
#                     prompt_speech = load_wav(prompt_wav, PROMPT_SR)
#                     if not prompt_text:
#                         # 添加ASR识别逻辑
#                         prompt_text = asr_prompt_wav_recognition(prompt_wav)
#                         if prompt_text:
#                             logger.info(f"ASR识别结果: {prompt_text}")
#                         else:
#                             logger.warning("ASR识别失败，请手动输入prompt文本")
#                             # 如果ASR识别失败且用户未提供prompt_text，抛出明确错误
#                             if not prompt_text:
#                                 raise ValueError("ASR识别失败且未提供prompt文本，请手动输入prompt文本或重新上传清晰的音频")
#                     inference_method = cosyvoice.inference_zero_shot
#                     method_args = [text, prompt_text, prompt_speech]

#                 elif mode == '跨语种复刻':
#                     logger.info("Using inference_cross_lingual mode")
#                     prompt_speech = load_wav(prompt_wav, PROMPT_SR)
#                     inference_method = cosyvoice.inference_cross_lingual
#                     method_args = [text, prompt_speech]

#                 else:  # 自然语言控制
#                     logger.info("Using inference_instruct mode")
#                     ModelClass = get_model_class(model_path)
#                     prompt_speech = load_wav(prompt_wav, PROMPT_SR) if prompt_wav else None
#                     if ModelClass == CosyVoice2:
#                         inference_method = cosyvoice.inference_instruct2
#                         method_args = [text, instruct_text, prompt_speech]
#                     else:
#                         if 'sft' in model_dropdown.lower():
#                             raise ValueError("当前模型为SFT，不支持自然语言控制模式，请切换非SFT模型")
#                         inference_method = cosyvoice.inference_instruct
#                         method_args = [text, speaker, instruct_text]

#                 kwargs_infer = {"stream": stream, "speed": speed}
#                 logger.info(f"Inference method: {inference_method}, args: {method_args}, kwargs: {kwargs_infer}")
#                 output_iter = inference_method(*method_args, **kwargs_infer)

#                 # 处理输出
#                 if stream:
#                     # 流式输出处理
#                     tts_speeches = []
#                     for item in output_iter:
#                         wav = postprocess(item['tts_speech'])
#                         tts_speeches.append(wav)
                    
#                     if tts_speeches:
#                         final_audio = np.concatenate(tts_speeches, axis=0)
#                     else:
#                         final_audio = np.zeros(TARGET_SR)
#                 else:
#                     # 非流式输出处理
#                     tts_speeches = [item['tts_speech'] if isinstance(item['tts_speech'], torch.Tensor)
#                                     else torch.tensor(item['tts_speech']) for item in output_iter]
#                     if tts_speeches:
#                         audio_t = torch.concat(tts_speeches, dim=1)
#                         final_audio = audio_t.detach().cpu().numpy().flatten()
#                         final_audio = postprocess(final_audio)
#                     else:
#                         final_audio = np.zeros(TARGET_SR)

#             except Exception as e:
#                 logger.exception("推理出错")
#                 raise RuntimeError(f"生成失败: {e}")

#             # 保存音频文件
#             OUTPUT_DIR = Path("outputs") / "temp"
#             OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#             out_name = f"cosyvoice_{int(time.time()*1000)}.wav"
#             out_path = OUTPUT_DIR / out_name
#             logger.info(f"Saving audio to: {out_path}")
#             sf.write(str(out_path), final_audio, TARGET_SR)
            
#             duration_ms = int(len(final_audio) / TARGET_SR * 1000)
#             logger.info(f"Audio generation successful: {out_path} ({duration_ms}ms)")
            
#             return {"path": str(out_path), "duration_ms": duration_ms}
                
#         except Exception as e:
#             logger.error(f"CosyVoice synthesis error: {str(e)}")
#             raise

#     # =============================================================================
#     # 位置: CosyVoiceWorker.cleanup方法 (630-647)
#     # 作用: 资源清理与模型卸载（确保显存释放）
#     # 依赖: 调用: release_cosyvoice_model(模型释放) ; 被调用: orchestrator(任务调度)
#     # 概念: 资源管理(中级)、异常安全(中级)、显存回收(高级)
#     # 关键词: 资源清理, 模型卸载, 异常安全
#     # =============================================================================
#     async def cleanup(self):
#         """清理资源"""
#         try:
#             release_cosyvoice_model()
#         except Exception:
#             pass

# # =============================================================================
# # 位置: L3-功能层 - ASR识别函数 (362-379)
# # 作用: 语音转文本识别，支持提示音频内容提取
# # 依赖: 调用: AutoModel(语音识别模型) ; 被调用: synthesize(提示处理)
# # 概念: 语音转文本, 模型验证, 版本检查（难度：中级）
# # 关键词: 语音转文本, 模型验证, 版本检查
# # =============================================================================
# def asr_prompt_wav_recognition(prompt_wav_path):
#     # 检查ASR模块是否可用
#     if not ASR_AVAILABLE or AutoModel is None:
#         logger.warning("ASR module not available, skipping recognition")
#         return None
        
#     # 强化模型加载验证
#     model_path = os.path.join(MODELS_DIR, "SenseVoiceSmall")
#     if not os.path.exists(os.path.join(model_path, "model.pt")):
#         logger.critical(f"ASR模型未安装: {model_path}")
#         raise RuntimeError("请执行模型下载命令: python download_models.py --model=SenseVoiceSmall")
    
#     try:
#         # 新增模型版本检查
#         with open(os.path.join(model_path, "version.txt"), 'r') as f:
#             assert f.read().strip() == "1.0.0"
        
#         # 加载模型（保持原有逻辑）
#         asr = AutoModel(model=model_path)
        
#         # 执行语音识别
#         rec_result = asr.generate(prompt_wav_path)
        
#         # 提取识别结果文本
#         if rec_result and len(rec_result) > 0:
#             text = rec_result[0]["text"]
#             logger.info(f"ASR recognition result: {text}")
#             return text
#         else:
#             logger.warning("ASR recognition returned empty result")
#             return None
            
#     except Exception as e:
#         logger.error(f"ASR模型加载失败: {e}")
#         raise

# # =============================================================================
# # 位置: L3-功能层 - 音频后处理函数 (380-397)
# # 作用: 语音后处理模块（音量归一化/静音裁剪/尾音添加）
# # 依赖: 调用: librosa(音频处理)/numpy(数组操作) ; 被调用: synthesize(输出处理)
# # 概念: 动态范围压缩(中级)、零交叉检测(中级)、尾音平滑(中级)
# # 关键词: 音频标准化, 尾音平滑, 响度控制
# # =============================================================================
# def postprocess(speech_tensor, top_db=60, hop_length=220, win_length=440):
#     """对生成的音频进行后处理 - 基于app.py"""
#     import librosa
#     if isinstance(speech_tensor, torch.Tensor):
#         arr = speech_tensor.detach().cpu().squeeze().numpy()
#     else:
#         arr = np.asarray(speech_tensor).squeeze()
#     if arr.size == 0:
#         arr = np.zeros(int(TARGET_SR * 0.5))
#     trimmed, _ = librosa.effects.trim(arr, top_db=top_db, frame_length=win_length, hop_length=hop_length)
#     if trimmed.size == 0:
#         trimmed = arr
#     max_val = np.max(np.abs(trimmed)) if np.max(np.abs(trimmed)) > 0 else 1.0
#     if max_val > MAX_VAL:
#         trimmed = trimmed / max_val * MAX_VAL
#     tail = np.zeros(int(TARGET_SR * 0.2), dtype=trimmed.dtype)
#     final = np.concatenate([trimmed, tail], axis=0)
#     return final

# =============================================================================
# 位置: workers/cosyvoice_engine.py (1-549)
# 作用: CosyVoice TTS引擎核心实现，支持多模式语音合成与模型热加载
# 依赖: 调用: torch(张量计算)/soundfile(音频IO)/Matcha-TTS(语音处理) ; 被调用: orchestrator(任务调度)
# 概念: 单例模式(中级)、CUDA显存管理(高级)、流式推理(高级)
# 关键词: 语音合成, 模型热加载, 多线程安全, 显存优化
# =============================================================================
# workers/cosyvoice_engine.py (Fixed version based on app.py)
#!/usr/bin/env python3
# """
# CosyVoice Engine for TTS synthesis - Fixed version based on app.py perfect implementation
# """

# # =============================================================================
# # 位置: 标准库导入 (8-15)
# # 作用: 系统基础功能与异步编程支持
# # 依赖: 调用: Python标准库 ; 被调用: 全局模块
# # 概念: 标准库导入(初级)、异步编程(中级)、路径处理(初级)
# # 关键词: 系统库, 异步IO, 路径操作, 类型提示
# # =============================================================================
# import logging
# import sys
# import os
# from pathlib import Path
# import time
# import json
# import asyncio
# from typing import Dict, Any, Optional

# # =============================================================================
# # 位置: 第三方库导入 (16-19)
# # 作用: 科学计算与深度学习框架导入
# # 依赖: 调用: PyTorch/NumPy/Random ; 被调用: 全局模块
# # 概念: 张量计算(中级)、随机数生成(初级)、数组操作(中级)
# # 关键词: 深度学习, 张量操作, 随机种子, 数值计算
# # =============================================================================
# import torch
# import numpy as np
# import random

# # =============================================================================
# # 位置: 路径配置 (20-25)
# # 作用: Matcha-TTS依赖库路径动态添加
# # 依赖: 调用: sys.path(系统路径) ; 被调用: Matcha-TTS导入
# # 概念: 动态路径(中级)、模块导入(中级)、第三方依赖(中级)
# # 关键词: 路径注入, 依赖管理, 模块加载
# # =============================================================================
# # 添加Matcha-TTS路径到系统路径
# sys.path.insert(0, str(Path(__file__).parent.parent))
# matcha_path = str(Path(__file__).parent.parent / "third_party" / "Matcha-TTS")
# if matcha_path not in sys.path:
#     sys.path.insert(0, matcha_path)

# # =============================================================================
# # 位置: 日志配置 (27-29)
# # 作用: 全局日志系统初始化
# # 依赖: 调用: logging(日志库) ; 被调用: 全局模块
# # 概念: 日志系统(初级)、级别配置(初级)、全局实例(初级)
# # 关键词: 日志初始化, 级别设置, 全局记录器
# # =============================================================================
# # 配置日志
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # =============================================================================
# # 位置: CosyVoice模块导入 (31-43)
# # 作用: 核心TTS引擎与音频处理工具导入
# # 依赖: 调用: CosyVoice(语音合成)/Matcha-TTS(音频处理) ; 被调用: 全局模块
# # 概念: 条件导入(中级)、异常处理(中级)、模块状态(中级)
# # 关键词: 语音合成, 音频处理, 条件导入, 异常安全
# # =============================================================================
# try:
#     from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
#     from cosyvoice.utils.file_utils import load_wav
#     from cosyvoice.utils.common import set_all_random_seed
    
#     # 尝试导入Matcha-TTS
#     try:
#         import matcha.utils.audio as matcha_audio
#     except ImportError as e:
#         logger.warning(f"无法导入 matcha.utils.audio: {e}")
#         matcha_audio = None
        
#     COSYVOICE_AVAILABLE = True
#     logger.info("CosyVoice engine initialized successfully")
# except ImportError as e:
#     COSYVOICE_AVAILABLE = False
#     logger.error(f"CosyVoice import error: {e}")
#     CosyVoice = None
#     CosyVoice2 = None
#     matcha_audio = None

# # =============================================================================
# # 位置: ASR模块导入 (44-46)
# # 作用: 语音识别模型导入
# # 依赖: 调用: funasr(语音识别) ; 被调用: asr_prompt_wav_recognition
# # 概念: 语音识别(中级)、条件导入(中级)、异常处理(中级)
# # 关键词: 语音识别, 模型导入, 异常安全
# # =============================================================================
# try:
#     from funasr import AutoModel
#     ASR_AVAILABLE = True
# except ImportError as e:
#     ASR_AVAILABLE = False
#     logger.warning(f"ASR import error: {e}")
#     AutoModel = None

# # =============================================================================
# # 位置: 音频处理库导入 (47-48)
# # 作用: 音频文件读写支持库导入
# # 依赖: 调用: soundfile(音频IO) ; 被调用: 音频保存
# # 概念: 音频IO(中级)、文件格式(初级)、采样率(中级)
# # 关键词: 音频保存, 文件IO, 采样率处理
# # =============================================================================
# import soundfile as sf

# # =============================================================================
# # 位置: 全局常量定义 (55-65)
# # 作用: 音频处理参数配置与推理模式枚举定义
# # 依赖: 调用: 无 ; 被调用: synthesize(参数验证)/postprocess(音频处理)
# # 概念: 常量配置(初级)、枚举类型(中级)
# # 关键词: 采样率, 音频长度, 推理模式
# # =============================================================================
# # 常量定义 - 基于app.py
# PROMPT_SR = 16000
# TARGET_SR = 24000
# MAX_PROMPT_SEC = 30
# MAX_TEXT_LEN = 200
# MAX_VAL = 0.8

# inference_mode_list = ['预训练音色', '3s极速复刻', '跨语种复刻', '自然语言控制']
# instruct_dict = {
#     '预训练音色': '1. 选择预训练音色\n2. 点击生成音频按钮',
#     '3s极速复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 输入prompt文本\n3. 点击生成音频按钮',
#     '跨语种复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 点击生成音频按钮',
#     '自然语言控制': '1. 选择预训练音色\n2. 输入instruct文本\n3. 点击生成音频按钮'
# }

# # =============================================================================
# # 位置: 模型管理变量 (68-72)
# # 作用: 全局模型状态管理与线程安全控制
# # 依赖: 调用: threading(线程锁) ; 被调用: load_cosyvoice_model(模型加载)
# # 概念: 单例模式(高级)、线程安全(高级)、显存管理(高级)
# # 关键词: 全局状态, 线程锁, 模型缓存
# # =============================================================================
# # 模型管理变量 - 基于app.py的全局模型管理
# import threading
# _model_lock = threading.Lock()
# _current_model = None
# _current_model_path = None
# _current_model_mode = None
# _feat_ready = False

# # =============================================================================
# # 位置: Matcha-TTS工具函数 (74-120)
# # 作用: Matcha音频处理库的窗口与梅尔谱缓存管理
# # 依赖: 调用: torch(张量操作)/matcha_audio(音频处理) ; 被调用: reset_feat_extractor(特征提取器重置)
# # 概念: 傅里叶变换(高级)、梅尔滤波器组(高级)、设备感知(中级)
# # 关键词: hann_window, mel_basis, 设备同步
# # =============================================================================
# def force_reset_matcha_window(device, win_length=1920):
#     """
#     强制为 matcha 中对应 device 的 hann_window 生成正确大小并放到 device。
#     """
#     if matcha_audio is None:
#         logger.warning("matcha_audio 未导入，无法重置 hann_window")
#         return

#     try:
#         device_str = str(device)
#         hw = torch.hann_window(win_length).to(device)
#         if hasattr(matcha_audio, "hann_window"):
#             try:
#                 matcha_audio.hann_window[device_str] = hw
#             except Exception:
#                 matcha_audio.hann_window = {device_str: hw}
#         else:
#             matcha_audio.hann_window = {device_str: hw}
#         logger.info(f"[INFO] Matcha hann_window 已重置为 size={win_length} @ {device}")
#     except Exception as e:
#         logger.exception(f"[WARN] 重置 matcha hann_window 失败: {e}")

# def reset_matcha_mel_basis(n_fft, num_mels, sampling_rate, fmin, fmax, device):
#     """重置Matcha mel_basis"""
#     if matcha_audio is None:
#         logger.warning("matcha_audio 未导入，无法重置 mel_basis")
#         return
        
#     try:
#         key = f"{str(fmax)}_{str(device)}"
#         if hasattr(matcha_audio, "mel_basis") and key in matcha_audio.mel_basis:
#             del matcha_audio.mel_basis[key]
#     except Exception as e:
#         logger.warning(f"重置 mel_basis 失败: {e}")

# # =============================================================================
# # 位置: 设备推断函数 (122-175)
# # 作用: 自动推断PyTorch模型所在设备（CPU/GPU）
# # 依赖: 调用: torch.nn(神经网络模块) ; 被调用: reset_feat_extractor(特征提取器初始化)
# # 概念: 设备探测(中级)、参数遍历(中级)、异常处理(中级)
# # 关键词: 设备推断, 参数检查, 模块遍历
# # =============================================================================
# def get_model_device(model):
#     """
#     尝试从 model 或其常见子模块推断设备（返回 torch.device）。
#     兼容 CosyVoice 封装（可能不是 nn.Module 本身）。
#     """
#     try:
#         import torch.nn as nn

#         # 直接属性 device
#         try:
#             if hasattr(model, "device"):
#                 d = getattr(model, "device")
#                 if isinstance(d, torch.device):
#                     return d
#                 else:
#                     return torch.device(str(d))
#         except Exception:
#             pass

#         # 常见子模块名
#         candidates = ('hift', 'frontend', 'llm', 'flow', 'encoder', 'decoder', 'acoustic')
#         for name in candidates:
#             try:
#                 sub = getattr(model, name, None)
#                 if sub is None:
#                     continue
#                 if isinstance(sub, nn.Module):
#                     try:
#                         params = list(sub.parameters())
#                         if params:
#                             return params[0].device
#                     except Exception:
#                         pass
#             except Exception:
#                 pass

#         # 最后在 model.__dict__ 中搜索第一个 nn.Module
#         try:
#             for v in model.__dict__.values():
#                 if isinstance(v, nn.Module):
#                     try:
#                         params = list(v.parameters())
#                         if params:
#                             return params[0].device
#                     except Exception:
#                         pass
#         except Exception:
#             pass

#         # 回退 CPU
#         return torch.device("cpu")
#     except Exception as e:
#         logger.warning(f"获取模型设备失败，回退到CPU: {e}")
#         return torch.device("cpu")

# # =============================================================================
# # 位置: 特征提取器重置函数 (177-235)
# # 作用: 统一管理Matcha-TTS与CosyVoice的音频特征提取缓存
# # 依赖: 调用: force_reset_matcha_window(窗口重置)/reset_matcha_mel_basis(梅尔谱重置) ; 被调用: load_cosyvoice_model(模型加载)
# # 概念: 缓存一致性(高级)、跨库协同(高级)、设备同步(中级)
# # 关键词: 特征提取, 缓存管理, 窗口同步
# # =============================================================================
# def reset_feat_extractor(model):
#     """
#     重置 feat_extractor 相关缓存，并刷新 Matcha-TTS 全局 hann_window 和 mel_basis
#     基于app.py的实现
#     """
#     global _feat_ready
#     _feat_ready = False

#     try:
#         if hasattr(model, 'frontend') and hasattr(model.frontend, 'feat_extractor'):
#             feat = model.frontend.feat_extractor

#             # 删除/清理旧 window 和缓存（兼容性）
#             try:
#                 if hasattr(feat, 'window'):
#                     feat.window = None
#             except Exception:
#                 pass
#             try:
#                 if hasattr(feat, '_cached_window'):
#                     feat._cached_window = None
#             except Exception:
#                 pass

#             # 尝试从 feat 提取 win_length/n_fft
#             win_length = None
#             try:
#                 if hasattr(feat, 'win_length') and getattr(feat, 'win_length') is not None:
#                     win_length = int(getattr(feat, 'win_length'))
#                 elif hasattr(feat, 'n_fft') and getattr(feat, 'n_fft') is not None:
#                     win_length = int(getattr(feat, 'n_fft'))
#                 elif hasattr(feat, 'fft_size') and getattr(feat, 'fft_size') is not None:
#                     win_length = int(getattr(feat, 'fft_size'))
#             except Exception:
#                 win_length = None

#             if win_length is None:
#                 win_length = 1920  # 默认值

#             # 推断模型设备
#             device = get_model_device(model)

#             # 强制重置 Matcha hann_window
#             force_reset_matcha_window(device, win_length=win_length)

#             # 重置 mel_basis
#             reset_matcha_mel_basis(
#                 n_fft=win_length,
#                 num_mels=80,           # CosyVoice 默认
#                 sampling_rate=16000,   # prompt 音频采样率
#                 fmin=0,
#                 fmax=8000,
#                 device=device
#             )

#             logger.info("feat_extractor 初始化完成，Matcha window 和 mel_basis 已统一重置")
#             _feat_ready = True
#         else:
#             logger.info("模型无 frontend/feat_extractor，跳过 feat 初始化")
#             _feat_ready = True
#     except Exception as e:
#         logger.exception(f"reset_feat_extractor 出错: {e}")
#         # 为了不阻塞使用者，降级处理
#         _feat_ready = True

# # =============================================================================
# # 位置: 模型类推断函数 (237-252)
# # 作用: 根据模型路径自动选择CosyVoice/CosyVoice2模型类
# # 依赖: 调用: 无 ; 被调用: load_cosyvoice_model(模型加载)/validate_inputs(输入验证)
# # 概念: 路径解析(中级)、模型版本检测(中级)、异常处理(中级)
# # 关键词: 模型选择, 路径匹配, 版本检测
# # =============================================================================
# def get_model_class(model_path):
#     """根据模型路径确定模型类 - 对齐app.py的判断逻辑"""
#     try:
#         # 优先依据配置文件判断
#         if os.path.exists(os.path.join(str(model_path), "cosyvoice2.yaml")):
#             logger.info("Selected CosyVoice2 model class (by cosyvoice2.yaml)")
#             return CosyVoice2
#         # 再依据目录名关键词判断
#         base_name = os.path.basename(str(model_path)).lower()
#         if "cosyvoice2" in base_name:
#             logger.info("Selected CosyVoice2 model class (by folder name)")
#             return CosyVoice2
#     except Exception as e:
#         logger.warning(f"get_model_class 检测异常，回退 CosyVoice: {e}")
#     logger.info("Selected CosyVoice model class")
#     return CosyVoice

# # =============================================================================
# # 位置: 模型加载函数 (254-294)
# # 作用: 实现模型热切换与显存优化加载（支持FP16/JIT/TRT加速）
# # 依赖: 调用: get_model_class(模型选择)/reset_feat_extractor(特征重置) ; 被调用: synthesize(推理入口)
# # 概念: 双检锁模式(高级)、显存碎片整理(高级)、懒加载(中级)
# # 关键词: 模型单例, 显存释放, 模型切换
# # =============================================================================
# def load_cosyvoice_model(model_path, mode=None, load_jit=False, load_trt=False, fp16=False):
#     """
#     加载 CosyVoice 模型或刷新模式参数 - 基于app.py的完美实现
#     """
#     global _current_model, _current_model_path, _current_model_mode
#     with _model_lock:
#         # 模型已加载且路径相同
#         if _current_model_path == model_path:
#             if _current_model_mode != mode:
#                 _current_model_mode = mode
#                 reset_feat_extractor(_current_model)
#             return _current_model

#         # 模型切换 → 释放旧模型
#         if _current_model is not None:
#             try:
#                 del _current_model
#             except Exception:
#                 pass
#             _current_model = None
#             _current_model_path = None
#             _current_model_mode = None
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()

#         # 加载新模型
#         ModelClass = get_model_class(model_path)
#         model = ModelClass(model_path, load_jit=load_jit, load_trt=load_trt, fp16=fp16)
#         reset_feat_extractor(model)

#         _current_model = model
#         _current_model_path = model_path
#         _current_model_mode = mode
#         return _current_model

# # =============================================================================
# # 位置: 模型释放函数 (296-310)
# # 作用: 安全卸载模型并清理CUDA显存资源
# # 依赖: 调用: torch.cuda(显存管理) ; 被调用: 外部清理流程
# # 概念: 资源释放(中级)、显存回收(高级)、异常安全(中级)
# # 关键词: 显存清理, 模型卸载, 资源回收
# # =============================================================================
# def release_cosyvoice_model():
#     """
#     卸载当前模型，释放显存 - 基于app.py
#     """
#     global _current_model, _current_model_path, _current_model_mode
#     with _model_lock:
#         if _current_model is not None:
#             try:
#                 del _current_model
#             except Exception:
#                 pass
#             _current_model = None
#             _current_model_path = None
#             _current_model_mode = None
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()

# # =============================================================================
# # 位置: 音色列表获取函数 (312-320)
# # 作用: 从模型文件加载预训练音色信息
# # 依赖: 调用: torch.load(模型加载) ; 被调用: 外部UI组件
# # 概念: 模型元数据(中级)、文件IO(初级)、字典操作(初级)
# # 关键词: 音色管理, 模型元数据, 文件读取
# # =============================================================================
# def get_sft_speakers(model_path):
#     """获取预训练音色列表 - 基于app.py"""
#     spk2info_path = os.path.join(model_path, "spk2info.pt")
#     if os.path.exists(spk2info_path):
#         info = torch.load(spk2info_path, map_location="cpu")
#         return list(info.keys())
#     return []

# # =============================================================================
# # 位置: 输入验证函数 (322-342)
# # 作用: 验证模型路径与推理模式的兼容性
# # 依赖: 调用: get_model_class(模型类型检测) ; 被调用: synthesize(参数校验)
# # 概念: 输入验证(中级)、文件存在性检查(初级)、模式兼容(中级)
# # 关键词: 参数校验, 文件检查, 模式验证
# # =============================================================================
# def validate_inputs(text, mode, prompt_wav, prompt_text, instruct_text, model_path):
#     # 对齐 app.py 的核心校验逻辑，并保留模型文件完整性检查
#     logger.info(f"Validating inputs: text={text[:50]}..., mode={mode}, prompt_wav={prompt_wav}, prompt_text={(prompt_text[:50] if prompt_text else '')}, instruct_text={(instruct_text[:50] if instruct_text else '')}, model_path={model_path}")
    
#     # 基础文本校验
#     if not text or len(text) == 0:
#         raise ValueError("请输入合成文本")
#     if len(text) > MAX_TEXT_LEN:
#         raise ValueError(f"文字过长，请限制在{MAX_TEXT_LEN}字以内")
#     if mode not in inference_mode_list:
#         raise ValueError("不支持的推理模式")

#     # 模式相关输入要求
#     if mode in ['3s极速复刻', '跨语种复刻'] and not prompt_wav:
#         raise ValueError("请提供参考音频")
#     if mode == '自然语言控制':
#         if not instruct_text or instruct_text.strip() == "":
#             raise ValueError("自然语言控制需要填写instruct文本")

#     # 模型类型与模式兼容性（与 app.py 对齐）
#     ModelClass = get_model_class(model_path)
#     if ModelClass == CosyVoice2 and mode == '预训练音色':
#         # 与 app.py: CosyVoice2 模型不支持预训练音色模式
#         raise ValueError("CosyVoice2 模型不支持预训练音色模式")
#     # SFT 模型不支持自然语言控制（与 app.py generate 分支一致）
#     if mode == '自然语言控制' and 'sft' in os.path.basename(str(model_path)).lower():
#         raise ValueError("当前模型为SFT，不支持自然语言控制模式，请切换非SFT模型")

#     # 路径有效性检查
#     if not os.path.isdir(model_path):
#         error_msg = f"模型目录无效: {model_path}"
#         logger.error(error_msg)
#         raise FileNotFoundError(error_msg)
    
#     # 必需文件检查（兼容 CosyVoice/CosyVoice2）
#     if ModelClass == CosyVoice2:
#         required_files = ['llm.pt', 'flow.pt', 'hift.pt', 'cosyvoice2.yaml']
#     else:
#         required_files = ['llm.pt', 'flow.pt', 'hift.pt', 'cosyvoice.yaml']
    
#     for f in required_files:
#         file_path = os.path.join(model_path, f)
#         if not os.path.exists(file_path):
#             error_msg = f"模型文件缺失: {f} at {file_path}"
#             logger.error(error_msg)
#             raise FileNotFoundError(error_msg)
    
#     logger.info("Input validation passed")
#     return True, None


# # =============================================================================
# # 位置: L2-配置层 - 项目根目录配置 (新增)
# # 作用: 定义项目根目录路径，支持相对路径计算
# # 依赖: 调用: Path(__file__) ; 被调用: MODELS_DIR等路径计算
# # 概念: 路径管理/相对路径/项目结构（难度：初级）
# # 关键词: 项目根目录,路径计算,相对路径
# # =============================================================================
# BASE_DIR = Path(__file__).parent.parent  # 项目根目录

# # =============================================================================
# # 位置: L2-配置层 - 模型路径配置 (新增)
# # 作用: 定义模型存储目录路径，支持模型文件的统一管理
# # 依赖: 调用: Path ; 被调用: asr_prompt_wav_recognition, CosyVoiceWorker.initialize
# # 概念: 路径管理/资源定位/文件系统（难度：初级）
# # 关键词: 模型路径,资源定位,文件系统
# # =============================================================================
# MODELS_DIR = BASE_DIR / "models"          # 模型存储

# # =============================================================================
# # 位置: L2-配置层 - 输出路径配置 (新增)
# # 作用: 定义音频文件输出路径，支持自动创建目录结构
# # 依赖: 调用: Path/Path.mkdir方法 ; 被调用: synthesize方法
# # 概念: 路径管理/目录创建/文件组织（难度：初级）
# # 关键词: 输出路径,目录创建,文件组织
# # =============================================================================
# # 将OUTPUT_DIR定义移至函数内部，避免模块导入时的路径问题
# # OUTPUT_DIR = Path("outputs") / "temp"
# # OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # =============================================================================
# # 位置: CosyVoiceWorker类 (381-515)
# # 作用: 多模式推理流程调度器（支持预训练/极速复刻/跨语种/自然语言控制模式）
# # 依赖: 调用: load_cosyvoice_model(模型加载)/postprocess(后处理) ; 被调用: orchestrator(任务调度)
# # 概念: 策略模式(中级)、流式拼接(高级)、参数路由(中级)
# # 关键词: 多模态推理, 参数路由, 流式处理
# # =============================================================================
# class CosyVoiceWorker:
#     def __init__(self, models_dir: str = "models"):
#         self.models_dir = Path(models_dir)

#     # =============================================================================
#     # 位置: CosyVoiceWorker.initialize方法 (516-518)
#     # 作用: Worker初始化方法（延迟加载模型）
#     # 依赖: 调用: 无 ; 被调用: orchestrator(任务调度)
#     # 概念: 延迟加载(中级)、异步初始化(中级)
#     # 关键词: 延迟加载, 异步初始化, Worker生命周期
#     # =============================================================================
#     async def initialize(self):
#         """初始化worker - 延迟加载模型"""
#         return

#     # =============================================================================
#     # 位置: CosyVoiceWorker.synthesize方法 (520-629)
#     # 作用: 多模式语音合成核心流程（参数兼容/模型加载/推理执行/音频后处理）
#     # 依赖: 调用: load_cosyvoice_model(模型加载)/postprocess(后处理)/validate_inputs(参数验证) ; 被调用: orchestrator(任务调度)
#     # 概念: 策略模式(中级)、流式处理(高级)、参数路由(中级)
#     # 关键词: 多模态推理, 参数兼容, 流式输出, 音频保存
#     # =============================================================================
#     async def synthesize(self, text: str, **kwargs):
#         """
#         合成语音 - 统一参数处理版本
#         """
#         global _feat_ready
#         try:
#             # 提取参数并设置默认值
#             mode = kwargs.get("mode", "预训练音色")
#             speaker = kwargs.get("speaker", "中文女")
#             prompt_wav = kwargs.get("prompt_wav_upload") or kwargs.get("prompt_wav_record")
#             prompt_text = kwargs.get("prompt_text", "")
#             instruct_text = kwargs.get("instruct_text", "")
#             stream = kwargs.get("stream", False)
#             speed = float(kwargs.get("speed", 1.0))
#             seed = int(kwargs.get("seed", 0))
#             model_dropdown = kwargs.get("model_dropdown") or "cosyvoice-300m-sft"
            
#             logger.info(f"Synthesizing with mode: {mode}, speaker: {speaker}, text length: {len(text)}")
#             logger.info(f"All kwargs: {kwargs}")
            
#             # 验证文本长度
#             if len(text) > MAX_TEXT_LEN:
#                 logger.warning(f"Text length {len(text)} exceeds recommended limit {MAX_TEXT_LEN}, truncating")
#                 text = text[:MAX_TEXT_LEN]

#             # 确保 prompt_wav 是字符串路径
#             logger.info(f"Processing prompt_wav: {prompt_wav}, type: {type(prompt_wav)}")
#             if prompt_wav and hasattr(prompt_wav, "name"):
#                 prompt_wav = prompt_wav.name
#                 logger.info(f"prompt_wav after name extraction: {prompt_wav}")
#             if prompt_wav:
#                 prompt_wav = os.path.normpath(str(prompt_wav))
#                 logger.info(f"prompt_wav after normpath: {prompt_wav}")
#                 if not os.path.exists(prompt_wav):
#                     raise FileNotFoundError(f"Prompt audio file not found: {prompt_wav}")

#             # 模型路径处理
#             model_path = str(self.models_dir / model_dropdown)
#             logger.info(f"Model path: {model_path}")
#             if not os.path.exists(model_path):
#                 raise FileNotFoundError(f"Model directory not found: {model_path}")

#             # 验证输入参数（完全对齐 demo 的关键规则）
#             logger.info("Validating inputs...")
#             ok, err = validate_inputs(text, mode, prompt_wav, prompt_text, instruct_text, model_path)
#             if not ok:
#                 raise ValueError(err)

#             # 加载模型
#             logger.info("Loading model...")
#             try:
#                 cosyvoice = load_cosyvoice_model(model_path, mode=mode)
#             except Exception as e:
#                 raise RuntimeError(f"模型加载失败: {e}")

#             # 检查 window 是否已重置
#             if not _feat_ready:
#                 raise RuntimeError("等待 window 重置，请稍后再试（正在初始化 feat_extractor / matcha window）")

#             # 设置随机种子
#             set_all_random_seed(seed if seed else random.randint(1, 100000000))

#             # 根据模式执行推理 - 基于app.py的完美逻辑
#             logger.info("Starting inference...")
#             try:
#                 if mode == '预训练音色':
#                     logger.info("Using inference_sft mode")
#                     inference_method = cosyvoice.inference_sft
#                     method_args = [text, speaker]

#                 elif mode == '3s极速复刻':
#                     logger.info("Using inference_zero_shot mode")
#                     prompt_speech = load_wav(prompt_wav, PROMPT_SR)
#                     if not prompt_text:
#                         # 添加ASR识别逻辑
#                         prompt_text = asr_prompt_wav_recognition(prompt_wav)
#                         if prompt_text:
#                             logger.info(f"ASR识别结果: {prompt_text}")
#                         else:
#                             logger.warning("ASR识别失败，请手动输入prompt文本")
#                             # 如果ASR识别失败且用户未提供prompt_text，抛出明确错误
#                             if not prompt_text:
#                                 raise ValueError("ASR识别失败且未提供prompt文本，请手动输入prompt文本或重新上传清晰的音频")
#                     inference_method = cosyvoice.inference_zero_shot
#                     method_args = [text, prompt_text, prompt_speech]

#                 elif mode == '跨语种复刻':
#                     logger.info("Using inference_cross_lingual mode")
#                     prompt_speech = load_wav(prompt_wav, PROMPT_SR)
#                     inference_method = cosyvoice.inference_cross_lingual
#                     method_args = [text, prompt_speech]

#                 else:  # 自然语言控制
#                     logger.info("Using inference_instruct mode")
#                     ModelClass = get_model_class(model_path)
#                     prompt_speech = load_wav(prompt_wav, PROMPT_SR) if prompt_wav else None
#                     if ModelClass == CosyVoice2:
#                         inference_method = cosyvoice.inference_instruct2
#                         method_args = [text, instruct_text, prompt_speech]
#                     else:
#                         if 'sft' in model_dropdown.lower():
#                             raise ValueError("当前模型为SFT，不支持自然语言控制模式，请切换非SFT模型")
#                         inference_method = cosyvoice.inference_instruct
#                         method_args = [text, speaker, instruct_text]

#                 kwargs_infer = {"stream": stream, "speed": speed}
#                 logger.info(f"Inference method: {inference_method}, args: {method_args}, kwargs: {kwargs_infer}")
#                 output_iter = inference_method(*method_args, **kwargs_infer)

#                 # 处理输出
#                 if stream:
#                     # 流式输出处理
#                     tts_speeches = []
#                     for item in output_iter:
#                         wav = postprocess(item['tts_speech'])
#                         tts_speeches.append(wav)
                    
#                     if tts_speeches:
#                         final_audio = np.concatenate(tts_speeches, axis=0)
#                     else:
#                         final_audio = np.zeros(TARGET_SR)
#                 else:
#                     # 非流式输出处理
#                     tts_speeches = [item['tts_speech'] if isinstance(item['tts_speech'], torch.Tensor)
#                                     else torch.tensor(item['tts_speech']) for item in output_iter]
#                     if tts_speeches:
#                         audio_t = torch.concat(tts_speeches, dim=1)
#                         final_audio = audio_t.detach().cpu().numpy().flatten()
#                         final_audio = postprocess(final_audio)
#                     else:
#                         final_audio = np.zeros(TARGET_SR)

#             except Exception as e:
#                 logger.exception("推理出错")
#                 raise RuntimeError(f"生成失败: {e}")

#             # 保存音频文件
#             OUTPUT_DIR = Path("outputs") / "temp"
#             OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#             out_name = f"cosyvoice_{int(time.time()*1000)}.wav"
#             out_path = OUTPUT_DIR / out_name
#             logger.info(f"Saving audio to: {out_path}")
#             sf.write(str(out_path), final_audio, TARGET_SR)
            
#             duration_ms = int(len(final_audio) / TARGET_SR * 1000)
#             logger.info(f"Audio generation successful: {out_path} ({duration_ms}ms)")
            
#             return {"path": str(out_path), "duration_ms": duration_ms}
                
#         except Exception as e:
#             logger.error(f"CosyVoice synthesis error: {str(e)}")
#             raise

#     # =============================================================================
#     # 位置: CosyVoiceWorker.cleanup方法 (630-647)
#     # 作用: 资源清理与模型卸载（确保显存释放）
#     # 依赖: 调用: release_cosyvoice_model(模型释放) ; 被调用: orchestrator(任务调度)
#     # 概念: 资源管理(中级)、异常安全(中级)、显存回收(高级)
#     # 关键词: 资源清理, 模型卸载, 异常安全
#     # =============================================================================
#     async def cleanup(self):
#         """清理资源"""
#         try:
#             release_cosyvoice_model()
#         except Exception:
#             pass

# # =============================================================================
# # 位置: L3-功能层 - ASR识别函数 (362-379)
# # 作用: 语音转文本识别，支持提示音频内容提取
# # 依赖: 调用: AutoModel(语音识别模型) ; 被调用: synthesize(提示处理)
# # 概念: 语音转文本, 模型验证, 版本检查（难度：中级）
# # 关键词: 语音转文本, 模型验证, 版本检查
# # =============================================================================
# def asr_prompt_wav_recognition(prompt_wav_path):
#     # 检查ASR模块是否可用
#     if not ASR_AVAILABLE or AutoModel is None:
#         logger.warning("ASR module not available, skipping recognition")
#         return None
        
#     # 强化模型加载验证
#     model_path = os.path.join(MODELS_DIR, "SenseVoiceSmall")
#     if not os.path.exists(os.path.join(model_path, "model.pt")):
#         logger.critical(f"ASR模型未安装: {model_path}")
#         raise RuntimeError("请执行模型下载命令: python download_models.py --model=SenseVoiceSmall")
    
#     try:
#         # 新增模型版本检查
#         with open(os.path.join(model_path, "version.txt"), 'r') as f:
#             assert f.read().strip() == "1.0.0"
        
#         # 加载模型（保持原有逻辑）
#         asr = AutoModel(model=model_path)
        
#         # 执行语音识别
#         rec_result = asr.generate(prompt_wav_path)
        
#         # 提取识别结果文本
#         if rec_result and len(rec_result) > 0:
#             text = rec_result[0]["text"]
#             logger.info(f"ASR recognition result: {text}")
#             return text
#         else:
#             logger.warning("ASR recognition returned empty result")
#             return None
            
#     except Exception as e:
#         logger.error(f"ASR模型加载失败: {e}")
#         raise

# # =============================================================================
# # 位置: L3-功能层 - 音频后处理函数 (380-397)
# # 作用: 语音后处理模块（音量归一化/静音裁剪/尾音添加）
# # 依赖: 调用: librosa(音频处理)/numpy(数组操作) ; 被调用: synthesize(输出处理)
# # 概念: 动态范围压缩(中级)、零交叉检测(中级)、尾音平滑(中级)
# # 关键词: 音频标准化, 尾音平滑, 响度控制
# # =============================================================================
# def postprocess(speech_tensor, top_db=60, hop_length=220, win_length=440):
#     """对生成的音频进行后处理 - 基于app.py"""
#     import librosa
#     if isinstance(speech_tensor, torch.Tensor):
#         arr = speech_tensor.detach().cpu().squeeze().numpy()
#     else:
#         arr = np.asarray(speech_tensor).squeeze()
#     if arr.size == 0:
#         arr = np.zeros(int(TARGET_SR * 0.5))
#     trimmed, _ = librosa.effects.trim(arr, top_db=top_db, frame_length=win_length, hop_length=hop_length)
#     if trimmed.size == 0:
#         trimmed = arr
#     max_val = np.max(np.abs(trimmed)) if np.max(np.abs(trimmed)) > 0 else 1.0
#     if max_val > MAX_VAL:
#         trimmed = trimmed / max_val * MAX_VAL
#     tail = np.zeros(int(TARGET_SR * 0.2), dtype=trimmed.dtype)
#     final = np.concatenate([trimmed, tail], axis=0)
#     return final

import logging
import asyncio
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Callable
from functools import wraps
import traceback

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
matcha_path = str(Path(__file__).parent.parent / "third_party" / "Matcha-TTS")
if matcha_path not in sys.path:
    sys.path.insert(0, matcha_path)

import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CosyVoiceConfig:
    """配置类，控制装饰器和监控功能"""
    ENABLE_DETAILED_LOGGING = True  # 启用详细日志
    ENABLE_PERFORMANCE_MONITORING = True  # 启用性能监控
    ENABLE_MEMORY_MONITORING = True  # 启用内存监控
    PERFORMANCE_THRESHOLD_MS = {
        'validate_inputs': 100,
        'reset_feat_extractor': 500,
        'load_cosyvoice_model': 2000,
        'synthesize': 5000,
        'asr_prompt_wav_recognition': 3000,
    }
    
    @classmethod
    def should_log_execution(cls):
        return cls.ENABLE_DETAILED_LOGGING
    
    @classmethod
    def get_performance_threshold(cls, func_name: str) -> int:
        return cls.PERFORMANCE_THRESHOLD_MS.get(func_name, 1000)
    
    @classmethod
    def should_monitor_memory(cls):
        return cls.ENABLE_MEMORY_MONITORING and torch.cuda.is_available()


def log_execution(func: Callable) -> Callable:
    """记录函数执行的入口和出口，增强可观测性"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not CosyVoiceConfig.should_log_execution():
            return func(*args, **kwargs)
            
        func_name = func.__name__
        logger.debug(f"Entering {func_name}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Exiting {func_name} successfully")
            return result
        except Exception as e:
            logger.debug(f"Exiting {func_name} with error: {e}")
            raise
    return wrapper


def safe_exception(func: Callable) -> Callable:
    """安全的异常处理装饰器，提供详细的错误上下文"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        try:
            return func(*args, **kwargs)
        except (RuntimeError, ValueError, FileNotFoundError, AttributeError, NameError) as e:
            # 已知的特定异常，保持原有处理
            logger.warning(f"{func_name} encountered expected error: {type(e).__name__}: {e}")
            raise
        except Exception as e:
            # 未知异常，提供详细上下文
            logger.error(f"Unexpected error in {func_name}: {type(e).__name__}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"{func_name} failed: {e}") from e
    return wrapper


def memory_monitor(func: Callable) -> Callable:
    """内存监控装饰器，监控GPU内存使用"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not CosyVoiceConfig.should_monitor_memory():
            return func(*args, **kwargs)
            
        # 记录函数执行前的内存状态
        initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        try:
            result = func(*args, **kwargs)
            # 记录函数执行后的内存状态
            final_memory = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_increase = final_memory - initial_memory
            
            if memory_increase > 100:  # 内存增加超过100MB就警告
                logger.warning(f"{func.__name__} increased GPU memory by {memory_increase:.1f}MB "
                             f"(from {initial_memory:.1f}MB to {final_memory:.1f}MB)")
            else:
                logger.debug(f"{func.__name__} GPU memory usage: {final_memory:.1f}MB "
                           f"(increase: {memory_increase:.1f}MB)")
                
            return result
        except Exception as e:
            final_memory = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_increase = final_memory - initial_memory
            logger.error(f"{func.__name__} failed after GPU memory increase of "
                        f"{memory_increase:.1f}MB: {e}")
            raise
    return wrapper

def performance_monitor(threshold_ms: int = 1000) -> Callable:
    """性能监控装饰器，记录执行时间"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not CosyVoiceConfig.ENABLE_PERFORMANCE_MONITORING:
                return func(*args, **kwargs)
                
            # 使用函数特定的阈值，如果没有指定则使用默认值
            actual_threshold = CosyVoiceConfig.get_performance_threshold(func.__name__)
            if actual_threshold == 1000 and threshold_ms != 1000:  # 没有特定配置，使用传入的阈值
                actual_threshold = threshold_ms
                
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000
                
                if execution_time > actual_threshold:
                    logger.warning(f"{func.__name__} took {execution_time:.2f}ms (threshold: {actual_threshold}ms)")
                else:
                    logger.debug(f"{func.__name__} completed in {execution_time:.2f}ms")
                    
                return result
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                logger.error(f"{func.__name__} failed after {execution_time:.2f}ms: {e}")
                raise
        return wrapper
    return decorator

try:
    from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
    from cosyvoice.utils.common import set_all_random_seed
    from cosyvoice.utils.file_utils import load_wav
    
    try:
        import matcha.utils.audio as matcha_audio
    except ImportError as e:
        logger.warning(f"Failed to import matcha.utils.audio: {e}")
        matcha_audio = None
        
    COSYVOICE_AVAILABLE = True
    logger.debug("CosyVoice engine initialized successfully")
except ImportError as e:
    COSYVOICE_AVAILABLE = False
    logger.error(f"CosyVoice import error: {e}")
    CosyVoice = None
    CosyVoice2 = None
    matcha_audio = None

try:
    from funasr import AutoModel
    ASR_AVAILABLE = True
except ImportError as e:
    ASR_AVAILABLE = False
    logger.warning(f"ASR module import error: {e}")
    AutoModel = None

PROMPT_SR = 16000
TARGET_SR = 24000
MAX_PROMPT_SEC = 30
MAX_TEXT_LEN = 500
MAX_VAL = 0.8

inference_mode_list = ['预训练音色', '3s极速复刻', '跨语种复刻', '自然语言控制']
instruct_dict = {
    '预训练音色': '1. 选择预训练音色\n2. 点击生成音频按钮\n\n适用模型：CosyVoice-300M-SFT, CosyVoice-300M-Instruct',
    '3s极速复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 输入prompt文本\n3. 点击生成音频按钮\n\n适用模型：CosyVoice2-0.5B, CosyVoice-300M, CosyVoice-300M-SFT, CosyVoice-300M-Instruct, CosyVoice-300M-25Hz',
    '跨语种复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 点击生成音频按钮\n\n适用模型：CosyVoice2-0.5B, CosyVoice-300M, CosyVoice-300M-SFT, CosyVoice-300M-Instruct, CosyVoice-300M-25Hz', 
    '自然语言控制': '1. 选择预训练音色\n2. 输入instruct文本\n3. 点击生成音频按钮\n\n适用模型：CosyVoice2-0.5B'
}
import threading
_model_lock = threading.Lock()
_current_model = None
_current_model_path = None
_current_model_mode = None
_feat_ready = False

@log_execution
@safe_exception
def force_reset_matcha_window(device, win_length=1920):
    if matcha_audio is None:
        logger.warning("matcha_audio not imported, cannot reset hann_window")
        return

    try:
        device_str = str(device)
        hw = torch.hann_window(win_length).to(device)
        if hasattr(matcha_audio, "hann_window"):
            try:
                matcha_audio.hann_window[device_str] = hw
            except Exception:
                matcha_audio.hann_window = {device_str: hw}
        else:
            matcha_audio.hann_window = {device_str: hw}
        logger.debug(f"Matcha hann_window reset to size={win_length} @ {device}")
    except Exception as e:
        logger.exception(f"[WARN] 重置 matcha hann_window 失败: {e}")

@log_execution
@safe_exception
def reset_matcha_mel_basis(n_fft, num_mels, sampling_rate, fmin, fmax, device):
    if matcha_audio is None:
        logger.warning("matcha_audio not imported, cannot reset mel_basis")
        return
        
    try:
        key = f"{str(fmax)}_{str(device)}"
        if hasattr(matcha_audio, "mel_basis") and key in matcha_audio.mel_basis:
            del matcha_audio.mel_basis[key]
    except Exception as e:
        logger.warning(f"Failed to reset mel_basis: {e}")

@log_execution
@safe_exception
def get_model_device(model):
    try:
        import torch.nn as nn

        try:
            if hasattr(model, "device"):
                d = getattr(model, "device")
                if isinstance(d, torch.device):
                    return d
                else:
                    return torch.device(str(d))
        except Exception:
            pass

        candidates = ('hift', 'frontend', 'llm', 'flow', 'encoder', 'decoder', 'acoustic')
        for name in candidates:
            try:
                sub = getattr(model, name, None)
                if sub is None:
                    continue
                if isinstance(sub, nn.Module):
                    try:
                        params = list(sub.parameters())
                        if params:
                            return params[0].device
                    except (AttributeError, TypeError):
                        pass
            except (AttributeError, TypeError):
                pass

        try:
            for v in model.__dict__.values():
                if isinstance(v, nn.Module):
                    try:
                        params = list(v.parameters())
                        if params:
                            return params[0].device
                    except (AttributeError, TypeError):
                        pass
        except (AttributeError, TypeError):
            pass

        return torch.device("cpu")
    except Exception as e:
        logger.warning(f"Failed to get model device, fallback to CPU: {e}")
        return torch.device("cpu")

@log_execution
@safe_exception
@performance_monitor(threshold_ms=500)  # 特征提取重置超过500ms就警告
def reset_feat_extractor(model):
    global _feat_ready
    _feat_ready = False

    try:
        if hasattr(model, 'frontend') and hasattr(model.frontend, 'feat_extractor'):
            feat = model.frontend.feat_extractor

            try:
                if hasattr(feat, 'window'):
                    feat.window = None
            except (AttributeError, TypeError):
                pass
            try:
                if hasattr(feat, '_cached_window'):
                    feat._cached_window = None
            except (AttributeError, TypeError):
                pass

            win_length = None
            try:
                if hasattr(feat, 'win_length') and getattr(feat, 'win_length') is not None:
                    win_length = int(getattr(feat, 'win_length'))
                elif hasattr(feat, 'n_fft') and getattr(feat, 'n_fft') is not None:
                    win_length = int(getattr(feat, 'n_fft'))
                elif hasattr(feat, 'fft_size') and getattr(feat, 'fft_size') is not None:
                    win_length = int(getattr(feat, 'fft_size'))
            except (AttributeError, TypeError, ValueError):
                win_length = None

            if win_length is None:
                win_length = 1920

            device = get_model_device(model)

            force_reset_matcha_window(device, win_length=win_length)

            reset_matcha_mel_basis(
                n_fft=win_length,
                num_mels=80,
                sampling_rate=16000,
                fmin=0,
                fmax=8000,
                device=device
            )

            logger.debug("feat_extractor initialization completed, Matcha window and mel_basis reset")
            _feat_ready = True
        else:
            logger.debug("Model has no frontend/feat_extractor, skipping feat initialization")
            _feat_ready = True
    except (AttributeError, TypeError) as e:
        logger.warning(f"Feature extractor reset failed due to attribute/type error: {e}")
        _feat_ready = True
    except (AttributeError, TypeError) as e:
        logger.warning(f"Feature extractor reset failed due to attribute/type error: {e}")
        _feat_ready = True
    except Exception as e:
        logger.exception(f"reset_feat_extractor error: {e}")
        _feat_ready = True

@log_execution
@safe_exception
def get_model_class(model_path):
    # 原始逻辑完整保留
    if os.path.exists(os.path.join(str(model_path), "cosyvoice2.yaml")):
        logger.debug("Selected CosyVoice2 model class (cosyvoice2.yaml)")
        return CosyVoice2
    if os.path.exists(os.path.join(str(model_path), "cosyvoice.yaml")):
        logger.debug("Selected CosyVoice model class (cosyvoice.yaml)")
        return CosyVoice
    # 根据文件夹名后备判断
    base_name = os.path.basename(str(model_path)).lower()
    if "cosyvoice2" in base_name:
        logger.debug("Selected CosyVoice2 model class (folder name)")
        return CosyVoice2
    if "cosyvoice" in base_name:
        logger.debug("Selected CosyVoice model class (folder name)")
        return CosyVoice
    # Fallback
    raise ValueError(f"Unsupported model: {model_path}")


@log_execution
@safe_exception
@performance_monitor(threshold_ms=2000)  # 模型加载超过2秒就警告
@memory_monitor
def load_cosyvoice_model(model_path, mode=None, load_jit=False, load_trt=False, fp16=False):
    global _current_model, _current_model_path, _current_model_mode
    with _model_lock:
        if _current_model_path == model_path:
            if _current_model_mode != mode:
                _current_model_mode = mode
                reset_feat_extractor(_current_model)
            return _current_model

        if _current_model is not None:
            try:
                del _current_model
            except (NameError, AttributeError):
                pass
            _current_model = None
            _current_model_path = None
            _current_model_mode = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        ModelClass = get_model_class(model_path)
        model = ModelClass(model_path, load_jit=load_jit, load_trt=load_trt, fp16=fp16)
        reset_feat_extractor(model)

        _current_model = model
        _current_model_path = model_path
        _current_model_mode = mode
        return _current_model

@log_execution
@safe_exception
def release_cosyvoice_model():
    global _current_model, _current_model_path, _current_model_mode
    with _model_lock:
        if _current_model is not None:
            try:
                del _current_model
            except (NameError, AttributeError):
                pass
            _current_model = None
            _current_model_path = None
            _current_model_mode = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

@log_execution
@safe_exception
def get_sft_speakers(model_path):
    spk2info_path = os.path.join(model_path, "spk2info.pt")
    if os.path.exists(spk2info_path):
        info = torch.load(spk2info_path, map_location="cpu")
        return list(info.keys())
    return []

@log_execution
@safe_exception
@performance_monitor(threshold_ms=100)  # 输入验证超过100ms就警告
def validate_inputs(text, mode, prompt_wav, prompt_text, instruct_text, model_path):
    logger.debug(f"Validating inputs: text={text[:50]}..., mode={mode}, prompt_wav={prompt_wav}, prompt_text={(prompt_text[:50] if prompt_text else '')}, instruct_text={(instruct_text[:50] if instruct_text else '')}, model_path={model_path}")
    
    if not text or len(text) == 0:
        raise ValueError("请输入合成文本")
    if len(text) > MAX_TEXT_LEN:
        raise ValueError(f"文字过长，请限制在{MAX_TEXT_LEN}字以内")
    if mode not in inference_mode_list:
        raise ValueError("不支持的推理模式")

    if mode in ['3s极速复刻', '跨语种复刻'] and not prompt_wav:
        raise ValueError("请提供参考音频")
    if mode == '自然语言控制':
        if not instruct_text or instruct_text.strip() == "":
            raise ValueError("自然语言控制需要填写instruct文本")

    ModelClass = get_model_class(model_path)
    if ModelClass == CosyVoice2 and mode == '预训练音色':
        raise ValueError("CosyVoice2 model does not support pretrained voice mode")
    if mode == '自然语言控制' and 'sft' in os.path.basename(str(model_path)).lower():
        raise ValueError("Current model is SFT, natural language control mode not supported, please switch to non-SFT model")

    if not os.path.isdir(model_path):
        error_msg = f"模型目录无效: {model_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    if ModelClass == CosyVoice2:
        required_files = ['llm.pt', 'flow.pt', 'hift.pt', 'cosyvoice2.yaml']
    else:
        required_files = ['llm.pt', 'flow.pt', 'hift.pt', 'cosyvoice.yaml']
    
    for f in required_files:
        file_path = os.path.join(model_path, f)
        if not os.path.exists(file_path):
            error_msg = f"模型文件缺失: {f} at {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
    
    logger.debug("Input validation passed")
    return True, None

BASE_DIR = Path(__file__).parent.parent

MODELS_DIR = BASE_DIR / "models"

class CosyVoiceWorker:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        logger.info(f"CosyVoiceWorker initialized with models_dir: {models_dir}")

    @log_execution
    @safe_exception
    async def initialize(self):
        logger.info("CosyVoiceWorker initialized successfully")
        return

    @log_execution
    @performance_monitor(threshold_ms=5000)  # 合成超过5秒就警告
    @memory_monitor
    async def synthesize(self, text: str, **kwargs):
        global _feat_ready
        logger.info(f"Starting synthesis with text length: {len(text)}")
        try:
            mode = kwargs.get("mode", "预训练音色")
            speaker = kwargs.get("speaker", "中文女")
            prompt_wav = kwargs.get("prompt_wav_upload") or kwargs.get("prompt_wav_record")
            prompt_text = kwargs.get("prompt_text", "")
            instruct_text = kwargs.get("instruct_text", "")
            stream = kwargs.get("stream", False)
            speed = float(kwargs.get("speed", 1.0))
            seed = int(kwargs.get("seed", 0))
            model_dropdown = kwargs.get("model_dropdown") or "cosyvoice-300m-sft"
            
            logger.debug(f"Synthesizing with mode: {mode}, speaker: {speaker}, text length: {len(text)}")
            logger.debug(f"All kwargs: {kwargs}")
            
            if len(text) > MAX_TEXT_LEN:
                logger.warning(f"Text length {len(text)} exceeds recommended limit {MAX_TEXT_LEN}, truncating to {MAX_TEXT_LEN} characters")
                text = text[:MAX_TEXT_LEN]

            logger.debug(f"Processing prompt_wav: {prompt_wav}, type: {type(prompt_wav)}")
            if prompt_wav and hasattr(prompt_wav, "name"):
                prompt_wav = prompt_wav.name
                logger.debug(f"prompt_wav after name extraction: {prompt_wav}")
            if prompt_wav:
                prompt_wav = os.path.normpath(str(prompt_wav))
                logger.debug(f"prompt_wav after normpath: {prompt_wav}")
                if not os.path.exists(prompt_wav):
                    raise FileNotFoundError(f"Prompt audio file not found: {prompt_wav}")

            model_path = str(self.models_dir / model_dropdown)
            logger.debug(f"Model path: {model_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model directory not found: {model_path}")

            logger.debug("Validating inputs...")
            ok, err = validate_inputs(text, mode, prompt_wav, prompt_text, instruct_text, model_path)
            if not ok:
                raise ValueError(err)

            logger.debug("Loading model...")
            try:
                cosyvoice = load_cosyvoice_model(model_path, mode=mode)
            except (FileNotFoundError, ValueError) as e:
                raise
            except Exception as e:
                raise RuntimeError(f"Model loading failed: {e}")

            if not _feat_ready:
                raise RuntimeError("Waiting for window reset, please try again later (initializing feat_extractor / matcha window)")

            set_all_random_seed(seed if seed else random.randint(1, 100000000))

            logger.debug("Starting inference...")
            try:
                if mode == '预训练音色':
                    logger.debug("Using inference_sft mode")
                    inference_method = cosyvoice.inference_sft
                    method_args = [text, speaker]

                elif mode == '3s极速复刻':
                    logger.debug("Using inference_zero_shot mode")
                    prompt_speech = load_wav(prompt_wav, PROMPT_SR)
                    if not prompt_text:
                        prompt_text = asr_prompt_wav_recognition(prompt_wav)
                        if prompt_text:
                            logger.info(f"ASR recognition result: {prompt_text}")
                        else:
                            logger.warning("ASR recognition failed, please manually input prompt text")
                            if not prompt_text:
                                raise ValueError("ASR recognition failed and no prompt text provided, please manually input prompt text or upload clearer audio")
                    inference_method = cosyvoice.inference_zero_shot
                    method_args = [text, prompt_text, prompt_speech]

                elif mode == '跨语种复刻':
                    logger.debug("Using inference_cross_lingual mode")
                    prompt_speech = load_wav(prompt_wav, PROMPT_SR)
                    inference_method = cosyvoice.inference_cross_lingual
                    method_args = [text, prompt_speech]

                else:
                    logger.debug("Using inference_instruct mode")
                    ModelClass = get_model_class(model_path)
                    prompt_speech = load_wav(prompt_wav, PROMPT_SR) if prompt_wav else None
                    if ModelClass == CosyVoice2:
                        inference_method = cosyvoice.inference_instruct2
                        method_args = [text, instruct_text, prompt_speech]
                    else:
                        if 'sft' in model_dropdown.lower():
                            raise ValueError("当前模型为SFT，不支持自然语言控制模式，请切换非SFT模型")
                        inference_method = cosyvoice.inference_instruct
                        method_args = [text, speaker, instruct_text]

                kwargs_infer = {"stream": stream, "speed": speed}
                logger.debug(f"Inference method: {inference_method}, args: {method_args}, kwargs: {kwargs_infer}")
                output_iter = inference_method(*method_args, **kwargs_infer)

                if stream:
                    tts_speeches = []
                    for item in output_iter:
                        wav = postprocess(item['tts_speech'])
                        tts_speeches.append(wav)
                    
                    if tts_speeches:
                        final_audio = np.concatenate(tts_speeches, axis=0)
                    else:
                        final_audio = np.zeros(TARGET_SR)
                else:
                    tts_speeches = [item['tts_speech'] if isinstance(item['tts_speech'], torch.Tensor)
                                    else torch.tensor(item['tts_speech']) for item in output_iter]
                    if tts_speeches:
                        audio_t = torch.concat(tts_speeches, dim=1)
                        final_audio = audio_t.detach().cpu().numpy().flatten()
                        final_audio = postprocess(final_audio)
                    else:
                        final_audio = np.zeros(TARGET_SR)

            except (RuntimeError, ValueError, FileNotFoundError) as e:
                raise
            except Exception as e:
                logger.exception("Inference failed")
                raise RuntimeError(f"Generation failed: {e}")

            OUTPUT_DIR = Path("outputs") / "temp"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            out_name = f"cosyvoice_{int(time.time()*1000)}.wav"
            out_path = OUTPUT_DIR / out_name
            logger.debug(f"Saving audio to: {out_path}")
            sf.write(str(out_path), final_audio, TARGET_SR)
            
            duration_ms = int(len(final_audio) / TARGET_SR * 1000)
            logger.debug(f"Audio generation successful: {out_path} ({duration_ms}ms)")
            
            return {"path": str(out_path), "duration_ms": duration_ms}
                
        except (RuntimeError, ValueError, FileNotFoundError) as e:
            logger.error(f"CosyVoice synthesis error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during CosyVoice synthesis: {e}")
            raise RuntimeError(f"CosyVoice synthesis failed: {e}")

    @log_execution
    @safe_exception
    async def cleanup(self):
        logger.info("Starting CosyVoice cleanup")
        try:
            release_cosyvoice_model()
            logger.info("CosyVoice cleanup completed successfully")
        except (AttributeError, NameError) as e:
            logger.warning(f"CosyVoice cleanup warning: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during CosyVoice cleanup: {e}")

@log_execution
@safe_exception
@performance_monitor(threshold_ms=3000)  # ASR识别超过3秒就警告
def asr_prompt_wav_recognition(prompt_wav_path):
    if not ASR_AVAILABLE or AutoModel is None:
        logger.warning("ASR module not available, skipping recognition")
        return None
        
    model_path = os.path.join(MODELS_DIR, "SenseVoiceSmall")
    if not os.path.exists(os.path.join(model_path, "model.pt")):
        logger.critical(f"ASR model not installed: {model_path}")
        raise RuntimeError("Please run model download command: python download_models.py --model=SenseVoiceSmall")
    
    try:
        with open(os.path.join(model_path, "version.txt"), 'r') as f:
            assert f.read().strip() == "1.0.0"
        
        asr = AutoModel(model=model_path)
        
        rec_result = asr.generate(prompt_wav_path)
        
        if rec_result and len(rec_result) > 0:
            text = rec_result[0]["text"]
            logger.info(f"ASR recognition result: {text}")
            return text
        else:
            logger.warning("ASR recognition returned empty result")
            return None
            
    except Exception as e:
        logger.error(f"ASR model loading failed: {e}")
        raise

def postprocess(speech_tensor, top_db=60, hop_length=220, win_length=440):
    import librosa
    if isinstance(speech_tensor, torch.Tensor):
        arr = speech_tensor.detach().cpu().squeeze().numpy()
    else:
        arr = np.asarray(speech_tensor).squeeze()
    if arr.size == 0:
        arr = np.zeros(int(TARGET_SR * 0.5))
    trimmed, _ = librosa.effects.trim(arr, top_db=top_db, frame_length=win_length, hop_length=hop_length)
    if trimmed.size == 0:
        trimmed = arr
    max_val = np.max(np.abs(trimmed)) if np.max(np.abs(trimmed)) > 0 else 1.0
    if max_val > MAX_VAL:
        trimmed = trimmed / max_val * MAX_VAL
    tail = np.zeros(int(TARGET_SR * 0.2), dtype=trimmed.dtype)
    final = np.concatenate([trimmed, tail], axis=0)
    return final