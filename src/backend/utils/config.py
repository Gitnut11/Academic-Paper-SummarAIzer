import os

from dotenv import load_dotenv

load_dotenv(dotenv_path="../../.env")
# load_dotenv()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or None
NEO4J_URI = os.getenv("NEO4J_URI") or None
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME") or None
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") or None
