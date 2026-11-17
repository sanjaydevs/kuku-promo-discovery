import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env")


def get_setting(name, default=None):
    # Local development: read from .env or environment variables.
    value = os.getenv(name)

    if value:
        return value

    # Cloud deployment: read from Streamlit Secrets.
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


GEMINI_API_KEY = get_setting("GEMINI_API_KEY")
GEMINI_MODEL = get_setting(
    "GEMINI_MODEL",
    "gemini-3.1-flash-lite",
)

OPENROUTER_API_KEY = get_setting("OPENROUTER_API_KEY")
OPENROUTER_MODEL = get_setting(
    "OPENROUTER_MODEL",
    "openrouter/free",
)