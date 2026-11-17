
from openai import OpenAI
from google import genai
from google.genai import types

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL
)


def generate_text(
    prompt,
    system_instruction=None,
    temperature=0.2,
    provider="gemini",
    model=None
):
    if provider == "gemini":

        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"timeout": 120000}
        )

        response = client.models.generate_content(
            model=model or GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_instruction
            )
        )
        
        print("GEMINI ACTUAL MODEL:", response.model_version)

        content = response.text

        elif provider == "openrouter":

        import time

        client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            timeout=120.0,
            max_retries=0
        )

        messages = []

        if system_instruction:
            messages.append({
                "role": "system",
                "content": system_instruction
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        content = None

        # Retry once if OpenRouter returns an empty response.
        for attempt in range(2):
            response = client.chat.completions.create(
                model=model or OPENROUTER_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=4096
            )

            choice = response.choices[0] if response.choices else None

            print("OPENROUTER MODEL:", response.model)
            print("FINISH REASON:", choice.finish_reason if choice else None)
            print("USAGE:", response.usage)

            content = (
                choice.message.content
                if choice and choice.message
                else None
            )

            if content and content.strip():
                break

            print(f"OpenRouter returned empty content. Attempt {attempt + 1}/2")

            if attempt == 0:
                time.sleep(2)

        if not content or not content.strip():
            raise RuntimeError(
                "OpenRouter returned empty content after 2 attempts. "
                "Check deployment logs for model and finish reason."
            )

    else:
        raise ValueError(f"Unsupported provider: {provider}")

    if not content:
        raise RuntimeError(f"{provider} returned an empty response.")

    return content
