import logging
import os

from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

# Reset log every run (should be changed later)
if not logging.getLogger().handlers:
    log_path = ".log"
    with open(log_path, "w") as f:
        f.write("")
    # Create a log file if it doesn't exist
    logging.basicConfig(
        filename=log_path,  # The log file path
        level=logging.INFO,  # Log level; change to DEBUG for more detailed output
        format="%(asctime)s - %(levelname)s - %(message)s",  # Format for each log message
    )


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or None
NEO4J_URI = os.getenv("NEO4J_URI") or None
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME") or None
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") or None

if GEMINI_API_KEY is None:
    logging.error("GEMINI_API_KEY is not set in the environment variables.")
if NEO4J_URI is None:
    logging.error("NEO4J_URI is not set in the environment variables.")
if NEO4J_USERNAME is None:
    logging.error("NEO4J_USERNAME is not set in the environment variables.")
if NEO4J_PASSWORD is None:
    logging.error("NEO4J_PASSWORD is not set in the environment variables.")
