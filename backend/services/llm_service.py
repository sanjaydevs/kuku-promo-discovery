
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

        response = client.chat.completions.create(
            model=model or OPENROUTER_MODEL,
            messages=messages,
            temperature=temperature
        )

        print("OPENROUTER ACTUAL MODEL:", response.model)
        
        content = response.choices[0].message.content

    else:
        raise ValueError(f"Unsupported provider: {provider}")

    if not content:
        raise RuntimeError(f"{provider} returned an empty response.")

    return content
