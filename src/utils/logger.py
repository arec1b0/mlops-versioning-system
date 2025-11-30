"""
Logging Infrastructure
Provides structured logging with file rotation and colored console output.
SRP: Single responsibility for logging configuration.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional
import colorlog
from datetime import datetime

from src.utils.config import config


class LoggerFactory:
    """
    Factory for creating configured loggers.
    Implements consistent logging across the application.
    """
    
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name: str, log_file: Optional[str] = None) -> logging.Logger:
        """
        Get or create a logger with the given name.
        
        Args:
            name: Logger name (usually __name__)
            log_file: Optional specific log file path
            
        Returns:
            Configured logger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]
        
        logger = logging.getLogger(name)
        
        # Avoid duplicate handlers
        if logger.hasHandlers():
            return logger
        
        logger.setLevel(logging.DEBUG)
        
        # Console handler with colors
        console_handler = cls._create_console_handler()
        logger.addHandler(console_handler)
        
        # File handler with rotation
        file_handler = cls._create_file_handler(log_file)
        logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        cls._loggers[name] = logger
        return logger
    
    @classmethod
    def _create_console_handler(cls) -> logging.Handler:
        """Create colored console handler."""
        console_handler = colorlog.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s%(reset)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        
        console_handler.setFormatter(formatter)
        return console_handler
    
    @classmethod
    def _create_file_handler(cls, log_file: Optional[str] = None) -> logging.Handler:
        """Create rotating file handler."""
        if log_file is None:
            log_dir = Path(config.get('paths.logs'))
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"mlops_{datetime.now().strftime('%Y%m%d')}.log"
        
        max_bytes = config.get('logging.max_bytes', 10485760)  # 10MB
        backup_count = config.get('logging.backup_count', 5)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        return file_handler
    
    @classmethod
    def shutdown(cls):
        """Shutdown all loggers and handlers."""
        for logger in cls._loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        cls._loggers.clear()


# Convenience function
def get_logger(name: str) -> logging.Logger:
    """Get logger instance."""
    return LoggerFactory.get_logger(name)