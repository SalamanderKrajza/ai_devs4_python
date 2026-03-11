import logging
import sys
from pathlib import Path

def setup_logger(name: str, log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """
    Create a logger with console and optional file output.
    
    Args:
        name: The name of the logger.
        log_file: Optional path to a log file. If provided, logs will be written here.
        level: The logging level (default: logging.INFO).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding handlers multiple times if setup_logger is called again
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
