import logging
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from prompt import SAFETY_PROMPT

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

CLIENT = genai.Client(api_key=GEMINI_API_KEY)
generate_content_config = types.GenerateContentConfig(
    response_mime_type="text/plain",
)


def safety_check(question: str):
    contents = [
        types.Content(
            parts=[types.Part.from_text(text=SAFETY_PROMPT.format(question=question))],
            role="user",
        )
    ]

    response = CLIENT.models.generate_content(
        model="gemini-2.0-flash", contents=contents, config=generate_content_config
    )

    safe = not "unsafe" in response.text.strip().lower()
    if not safe:
        logging.warning(f"Unsafe question detected: {question}")
    return safe
