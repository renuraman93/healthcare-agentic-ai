"""
Phase 5 sanity check.
Runs the FULL RAG pipeline end-to-end:
Load -> Clean -> Chunk -> Embed -> Store -> Retrieve
"""

from app.rag.loaders import load_document
from app.rag.cleaning import clean_text
from app.rag.chunking import chunk_text
from app.rag.vector_store import VectorStore
from app.rag.retriever import Retriever
from app.utils.logger import get_logger

logger = get_logger(__name__)

TEST_FILE = "data/documents/sample_notes.txt"


def main() -> None:
    # 1. Load
    doc = load_document(TEST_FILE)
    print(f"\nLoaded: {doc.source_filename} ({doc.char_count} chars)")

    # 2. Clean
    cleaned = clean_text(doc.full_text)
    print(f"Cleaned: {len(cleaned)} chars")

    # 3. Chunk
    chunks = chunk_text(cleaned)
    print(f"Chunked into: {len(chunks)} chunks")
    for c in chunks:
        print(f"  Chunk {c.chunk_index}: {c.char_count} chars — "
              f"'{c.text[:60]}...'")

    # 4. Store (embeds + persists to ChromaDB)
    store = VectorStore()
    chunk_texts = [c.text for c in chunks]
    store.add_chunks(
        chunks=chunk_texts,
        source_filename=doc.source_filename,
        document_type=doc.document_type,
    )
    print(f"\nTotal chunks now in vector store: {store.count()}")
    print(f"Indexed sources: {store.list_sources()}")

    # 5. Retrieve
    retriever = Retriever()
    query = "What medications is the patient taking?"
    results = retriever.retrieve(query, top_k=2)

    print(f"\n--- Retrieval results for query: '{query}' ---")
    for r in results:
        print(f"\nSource: {r.source} (chunk {r.chunk_index}, "
              f"distance {r.relevance_distance:.4f})")
        print(r.text)

    context = retriever.build_context(results)
    print("\n--- Built Context (would be sent to LLM) ---")
    print(context)


if __name__ == "__main__":
    main()