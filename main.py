#!/usr/bin/env python3
"""
Main entry point for the multi-engine TTS orchestrator.
This script starts the orchestrator process which manages workers and provides the Gradio UI.
Supports command-line arguments for configuration.
"""
import argparse
import asyncio
import logging
import logging.config
import os
import sys
from pathlib import Path


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator.orchestrator import Orchestrator
from config import LOGGING_CONFIG

def setup_environment():
    """Setup environment variables and create necessary directories."""
    # Create required directories
    directories = [
        "models",
        "logs",
        "outputs/temp",
        "outputs/jobs",
        "speakers"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import gradio
        import fastapi
        import uvicorn
        import pydub
        print("✅ All core dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: python setup.py")
        return False

def setup_logging():
    """Configure logging for the application."""
    import os
    import sys
    
    # 设置环境变量以确保使用UTF-8编码
    if sys.platform.startswith('win'):
        os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # 修改LOGGING_CONFIG以使用UTF-8编码
    LOGGING_CONFIG['handlers']['file']['encoding'] = 'utf-8'
    logging.config.dictConfig(LOGGING_CONFIG)

def main(args=None):
    """Main entry point."""
    # If no args provided, parse command line arguments
    if args is None:
        parser = argparse.ArgumentParser(description="Launch Multi-Engine TTS Orchestrator")
        parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
        parser.add_argument("--port", type=int, default=7860, help="Port to run on")
        parser.add_argument("--workers", type=int, default=2, help="Number of TTS workers")
        parser.add_argument("--debug", action="store_true", help="Enable debug mode")
        parser.add_argument("--setup", action="store_true", help="Run setup before launching")
        
        args = parser.parse_args()
        
        # Run setup if requested
        if args.setup:
            print("🔄 Running setup...")
            os.system(f"{sys.executable} setup.py")
        
        # Set environment variables for configuration
        os.environ["GRADIO_SERVER_NAME"] = args.host
        os.environ["GRADIO_SERVER_PORT"] = str(args.port)
        os.environ["MAX_WORKERS"] = str(args.workers)
        
        if args.debug:
            os.environ["LOG_LEVEL"] = "DEBUG"
            os.environ["GRADIO_DEBUG"] = "True"
        
        print("🎙️  Multi-Engine TTS Orchestrator")
        print("=" * 50)
        print(f"   Host: {args.host}")
        print(f"   Port: {args.port}")
        print(f"   Workers: {args.workers}")
        print(f"   Debug: {args.debug}")
    
    # Setup environment
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Configure logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting Multi-Engine TTS Orchestrator...")
        orchestrator = Orchestrator()
        orchestrator.run()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()