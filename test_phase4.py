"""
Phase 4 sanity check.
Confirms: PDF, TXT, and DOCX loaders all work correctly.
"""

from pathlib import Path

from app.rag.loaders import load_document, DocumentLoadError
from app.utils.logger import get_logger

logger = get_logger(__name__)

TEST_FILES = [
    "data/documents/sample_notes.txt",
    "data/documents/sample_report.pdf",
    "data/documents/sample_report.docx",
]


def main() -> None:
    for file_path in TEST_FILES:
        if not Path(file_path).exists():
            logger.warning(f"Skipping (not found yet): {file_path}")
            continue

        try:
            doc = load_document(file_path)
            print(f"\n--- {doc.source_filename} ---")
            print(f"Type: {doc.document_type}")
            print(f"Pages: {doc.page_count}")
            print(f"Chars: {doc.char_count}")
            print("Preview:", doc.full_text[:200].replace("\n", " "), "...")
        except DocumentLoadError as e:
            logger.error(f"Failed to load {file_path}: {e}")


if __name__ == "__main__":
    main()