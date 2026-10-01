"""
Chunking: splits cleaned document text into overlapping chunks
suitable for embedding and retrieval.
"""

from pydantic import BaseModel

from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TextChunk(BaseModel):
    chunk_index: int
    text: str
    char_count: int


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[TextChunk]:
    """
    Splits text into overlapping chunks using a sliding window over
    characters, breaking on whitespace where possible to avoid
    cutting words in half.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    if not text:
        return []

    chunks: list[TextChunk] = []
    start = 0
    text_length = len(text)
    index = 0

    while start < text_length:
        end = start + chunk_size

        if end < text_length:
            # Try to break at the last whitespace before `end`
            # so we don't cut a word in half.
            last_space = text.rfind(" ", start, end)
            if last_space != -1 and last_space > start:
                end = last_space

        chunk_str = text[start:end].strip()

        if chunk_str:
            chunks.append(
                TextChunk(
                    chunk_index=index,
                    text=chunk_str,
                    char_count=len(chunk_str),
                )
            )
            index += 1

        # Move the window forward, stepping back by the overlap amount
        next_start = end - chunk_overlap
        # Safety: always make forward progress
        start = next_start if next_start > start else end

    logger.info(f"Chunked text into {len(chunks)} chunks "
                f"(chunk_size={chunk_size}, overlap={chunk_overlap})")
    return chunks