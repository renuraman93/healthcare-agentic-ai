"""
Unit tests for text cleaning and chunking. No API calls.
"""

from app.rag.chunking import chunk_text
from app.rag.cleaning import clean_text


def test_clean_text_collapses_excess_whitespace():
    dirty = "Line one.\n\n\n\nLine two.   Extra   spaces."
    result = clean_text(dirty)
    assert "\n\n\n" not in result
    assert "   " not in result


def test_clean_text_preserves_clinical_values():
    text = "Blood pressure: 145/90 mmHg. Metformin 500 mg twice daily."
    result = clean_text(text)
    assert "145/90" in result
    assert "500 mg" in result


def test_clean_text_empty_input():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_chunk_text_respects_size_and_overlap():
    text = "word " * 500  # long enough to require multiple chunks
    chunks = chunk_text(text, chunk_size=800, chunk_overlap=150)
    assert len(chunks) > 1
    for c in chunks:
        assert c.char_count <= 800 + 1  # small tolerance for whitespace break


def test_chunk_text_does_not_split_words():
    text = "Metformin " * 200
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    for c in chunks:
        assert not c.text.startswith("etformin")  # would indicate a mid-word cut
        assert not c.text.endswith("Metform")


def test_chunk_text_empty_input_returns_no_chunks():
    assert chunk_text("") == []


def test_chunk_text_rejects_overlap_ge_size():
    import pytest
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=100, chunk_overlap=100)