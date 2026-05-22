import logging
import os
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    log_file_path: str,
    logger_name: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
    file_mode: str = "w",
) -> logging.Logger:

    log_path = Path(log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if logger_name is None:
        logger_name = str(log_path.resolve())

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(
        filename=log_path,
        mode=file_mode,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info(f"Logger initialized. Log file: {log_path}")

    return logger


def close_logger(logger: logging.Logger) -> None:

    handlers = logger.handlers[:]

    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)