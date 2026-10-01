"""
Text cleaning: normalize whitespace and formatting before chunking.

Kept intentionally conservative — we clean up noise without
altering clinical meaning (e.g., we never strip numbers, units,
or punctuation that could change a dosage or measurement).
"""

import re


def clean_text(text: str) -> str:
    """
    Normalizes whitespace and removes common extraction artifacts.
    """
    if not text:
        return ""

    # Normalize different newline styles to \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse 3+ consecutive newlines into 2 (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse repeated spaces/tabs into a single space
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Strip trailing whitespace on each line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Remove leading/trailing whitespace of the whole document
    return text.strip()