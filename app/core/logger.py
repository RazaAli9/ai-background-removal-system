import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.config import settings

# Create logs directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "app.log"

# Formatter
formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# File Handler (rotates after 5 MB)
file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=5 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)

file_handler.setFormatter(formatter)

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Logger
logger = logging.getLogger("bg_remover")

logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))

logger.handlers.clear()

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.propagate = False