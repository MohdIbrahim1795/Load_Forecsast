# logger_config.py
import logging
from typing import Optional
import sys

def setup_logger(
    name: str = __name__,
    log_file: str = 'application.log',
    level: int = logging.INFO
) -> logging.Logger:
    """
    Configure and return a logger with file and console handlers
    
    Args:
        name: Logger name (usually __name__)
        log_file: Path to log file
        level: Logging level (e.g., logging.INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding handlers multiple times
    if logger.handlers:
        return logger

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger