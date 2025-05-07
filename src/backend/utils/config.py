"""
Configuration and logging setup for backend environment.

Loads environment variables, sets global configs, and configures logging.
"""

import logging
import logging.config
import os
from dotenv import load_dotenv

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")
load_dotenv(dotenv_path)

# API Keys and connection strings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Logging configuration
LOG_DIR = os.getenv(
    "LOG_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "logs"),
)
os.makedirs(LOG_DIR, exist_ok=True)


def setup_logging():
    """Configure logging for the backend."""
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "standard",
                "filename": os.path.join(LOG_DIR, "backend_errors.log"),
                "maxBytes": 10 * 1024 * 1024,  # 10MB
                "backupCount": 5,
            },
        },
        "root": {
            "handlers": ["console", "file"],
            "level": "INFO",
        },
    }
    logging.config.dictConfig(log_config)


setup_logging()

if GEMINI_API_KEY is None:
    logger = logging.getLogger(__name__)
    logger.warning("GEMINI_API_KEY is not set in the environment.")
