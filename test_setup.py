"""
Phase 1 sanity check.
Confirms: Python version, venv activation, and .env loading all work.
"""

import sys
import os
from dotenv import load_dotenv

def main() -> None:
    print("Python executable in use:", sys.executable)
    print("Python version:", sys.version)

    load_dotenv()  # reads .env into environment variables
    api_key = os.getenv("GEMINI_API_KEY")

    if api_key and api_key != "your_api_key_here":
        print("GEMINI_API_KEY loaded successfully (value hidden).")
    else:
        print("GEMINI_API_KEY not set yet — that's fine for now, "
              "just make sure to add it before Phase 3 (LLM integration).")

if __name__ == "__main__":
    main()