
from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL

client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options={"timeout": 120000}
)


def generate_text(
    prompt,
    system_instruction=None,
    temperature=0.2
):
    config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system_instruction
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=config
        )

        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        return response.text

    except Exception as error:
        raise RuntimeError(
            f"Gemini API request failed: {error}"
        ) from error
