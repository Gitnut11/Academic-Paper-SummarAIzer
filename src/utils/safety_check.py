import logging

from google import genai
from google.genai import types

from utils.config import GEMINI_API_KEY
from utils.prompt import SAFETY_PROMPT

CLIENT = genai.Client(api_key=GEMINI_API_KEY)
generate_content_config = types.GenerateContentConfig(
    response_mime_type="text/plain",
)
model = "gemini-2.0-flash"


def safety_check(question: str):
    contents = [
        types.Content(
            parts=[types.Part.from_text(text=SAFETY_PROMPT.format(question=question))],
            role="user",
        )
    ]

    response = CLIENT.models.generate_content(
        model=model, contents=contents, config=generate_content_config
    )

    safe = not "unsafe" in response.text.strip().lower()
    if not safe:
        logging.warning(f"Unsafe question detected: {question}")
    return safe
