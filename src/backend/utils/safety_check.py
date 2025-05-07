"""
Module for content safety checking using Gemini API.
"""

import logging

from google import genai
from google.genai import types

from utils.config import GEMINI_API_KEY
from utils.prompt import SAFETY_PROMPT

# Set up Gemini client and model
CLIENT = genai.Client(api_key=GEMINI_API_KEY)
generate_content_config = types.GenerateContentConfig(
    response_mime_type="text/plain",
)
model = "gemini-2.0-flash"


def safety_check(question: str) -> bool:
    """
    Perform a safety check on a user's question using Gemini.

    Args:
        question (str): The question or prompt to check.

    Returns:
        bool: True if safe, False otherwise.
    """
    contents = [
        types.Content(
            parts=[types.Part.from_text(text=SAFETY_PROMPT.format(question=question))],
            role="user",
        )
    ]

    response = CLIENT.models.generate_content(
        model=model, contents=contents, config=generate_content_config
    )

    is_safe = "unsafe" not in response.text.strip().lower()

    if not is_safe:
        logging.warning(f"Unsafe question detected: {question}")

    return is_safe
