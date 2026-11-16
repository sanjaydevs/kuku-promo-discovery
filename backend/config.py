import os
from pathlib import Path
from dotenv import load_dotenv


# Locate the backend directory
BACKEND_DIR = Path(__file__).resolve().parent

# Load environment variables from backend/.env
load_dotenv(BACKEND_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing. "
        "Add it to backend/.env"
    )
