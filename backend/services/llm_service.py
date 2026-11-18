
import csv
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from openai import OpenAI
from google import genai
from google.genai import types

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL
)


# ============================================================
# USAGE LOGGING
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[1]

USAGE_DIR = BACKEND_DIR / "data" / "usage"
USAGE_FILE = USAGE_DIR / "llm_usage.csv"

USAGE_FIELDS = [
    "request_id",
    "timestamp",
    "provider",
    "model",
    "operation",
    "attempt",
    "input_tokens",
    "output_tokens",
    "total_tokens",
]


def save_usage(
    provider,
    model,
    operation,
    attempt,
    input_tokens,
    output_tokens,
    total_tokens
):
    """Append individual API request usage to a CSV file."""

    USAGE_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = USAGE_FILE.exists()

    with USAGE_FILE.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=USAGE_FIELDS
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "request_id": str(uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "model": model,
            "operation": operation,
            "attempt": attempt,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        })


# ============================================================
# LLM GENERATION
# ============================================================

def generate_text(
    prompt,
    system_instruction=None,
    temperature=0.2,
    provider="gemini",
    model=None,
    operation="unspecified"
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

        actual_model = response.model_version
        content = response.text

        usage = response.usage_metadata

        if usage:

            save_usage(
                provider="gemini",
                model=actual_model,
                operation=operation,
                attempt=1,
                input_tokens=usage.prompt_token_count,
                output_tokens=usage.candidates_token_count,
                total_tokens=usage.total_token_count
            )

        else:
            print("Gemini usage metadata unavailable.")

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

        for attempt in range(2):

            response = client.chat.completions.create(
                model=model or OPENROUTER_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=4096
            )

            choice = response.choices[0] if response.choices else None

            actual_model = response.model
            usage = response.usage

            if usage:

                save_usage(
                    provider="openrouter",
                    model=actual_model,
                    operation=operation,
                    attempt=attempt + 1,
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens
                )

            content = (
                choice.message.content
                if choice and choice.message
                else None
            )

            if content and content.strip():
                break

            print(
                f"OpenRouter returned empty content. "
                f"Attempt {attempt + 1}/2"
            )

            if attempt == 0:
                time.sleep(2)

        if not content or not content.strip():
            raise RuntimeError(
                "OpenRouter returned empty content after 2 attempts."
            )

    else:
        raise ValueError(f"Unsupported provider: {provider}")

    if not content:
        raise RuntimeError(f"{provider} returned an empty response.")

    return content
