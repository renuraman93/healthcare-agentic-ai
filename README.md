# Healthcare Agentic AI Chatbot

An educational, portfolio-grade multi-agent AI system that answers questions
about uploaded healthcare documents using Retrieval-Augmented Generation (RAG),
LangGraph orchestration, and Google Gemini.

**⚠️ This is a demo project, not medical advice.** It does not diagnose
conditions and cannot replace a qualified healthcare professional.

## Overview

The system routes user questions to one of several specialized agents —
document search, clinical data extraction, clinical reasoning, summarization,
or general health information — using a rule-based supervisor with an
optional LLM fallback. Every document-grounded answer includes source
citations; every response carries a safety disclaimer; emergency language
is detected and flagged.

## Business Problem / Objectives

Healthcare information is scattered across documents (clinical notes,
reports) and hard to query naturally. This project demonstrates how a
multi-agent RAG system can:
- Answer natural-language questions grounded in uploaded documents
- Extract structured clinical data (conditions, medications, vitals) from
  free text
- Summarize documents at different levels of detail
- Distinguish retrieved facts from general explanation, and clearly flag
  when information isn't available — rather than inventing an answer

## Architecture
                USER
                 |
          STREAMLIT UI
                 |
             FASTAPI
                 |
       LANGGRAPH SUPERVISOR
                 |
  +-------+------+------+-------+
  |       |             |       |
Document Clinical Summarization General
Search Extraction Healthcare
| | | |
+---RAG/ChromaDB------+ |
| |
+------------+----------------+
|
Response Agent
(sources, disclaimer,
emergency banner)
|
USER


**RAG pipeline:**
Documents (PDF/TXT/DOCX) -> Load -> Clean -> Chunk -> Embed (Gemini)
-> ChromaDB -> Retrieve top-K -> Build context -> LLM -> Answer

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| LLM | Google Gemini |
| Embeddings | Gemini (gemini-embedding-001) |
| Agent orchestration | LangGraph |
| Vector store | ChromaDB (local, persistent) |
| Backend | FastAPI |
| Frontend | Streamlit |
| Testing | pytest |
| Containerization | Docker, docker-compose |

## Project Structure
healthcare-agentic-ai/
├── app/
│ ├── main.py # FastAPI entry point
│ ├── api/ # chat, documents, health routers
│ ├── agents/ # supervisor, document, clinical,
│ │ # reasoning, summarization, general,
│ │ # response agents
│ ├── graph/ # LangGraph state, workflow, memory
│ ├── rag/ # loaders, cleaning, chunking,
│ │ # embeddings, vector_store, retriever
│ ├── llm/ # LLMProvider interface + Gemini impl
│ ├── models/ # Pydantic schemas (internal + API)
│ ├── services/ # document_service (ingestion orchestration)
│ ├── config/ # settings (pydantic-settings)
│ └── utils/ # logger
├── frontend/
│ ├── streamlit_app.py
│ ├── api_client.py
│ └── Dockerfile
├── tests/ # pytest unit, API, integration tests
├── data/documents/ # uploaded documents (gitignored)
├── vectorstore/ # ChromaDB persistent storage (gitignored)
├── Dockerfile.api
├── docker-compose.yml
├── requirements.txt
└── .env.example

## Installation

```powershell
git clone https://github.com/renuraman93/healthcare-agentic-ai.git
cd healthcare-agentic-ai
python -m venv myenv
.\myenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Environment Variables

Copy `.env.example` to `.env` and fill in your own Gemini API key
(get one at https://aistudio.google.com/app/apikey):

```env
GEMINI_API_KEY=your_api_key_here
LLM_PROVIDER=gemini
LLM_MODEL_NAME=gemini-2.5-flash
EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL_NAME=gemini-embedding-001
VECTOR_STORE_DIR=./vectorstore
VECTOR_COLLECTION_NAME=healthcare_docs
CHUNK_SIZE=800
CHUNK_OVERLAP=150
RETRIEVAL_TOP_K=4
RELEVANCE_MAX_DISTANCE=0.6
USE_LLM_ROUTER=false
APP_ENV=development
LOG_LEVEL=INFO
MAX_UPLOAD_SIZE_MB=10
ALLOWED_UPLOAD_EXTENSIONS=.pdf,.txt,.docx
```

**Note on API quota:** the free Gemini tier has a daily request limit on
some models. Tests are designed to use a `FakeLLM` wherever possible to
avoid consuming quota — see Testing below.

## How to Run

**Terminal 1 — API:**
```powershell
uvicorn app.main:app --reload
```

**Terminal 2 — Frontend:**
```powershell
streamlit run frontend/streamlit_app.py
```

Then open:
- API: http://127.0.0.1:8000
- Swagger docs: http://127.0.0.1:8000/docs
- Streamlit UI: http://localhost:8501

### Docker (alternative)

`Dockerfile.api` (backend) and `frontend/Dockerfile` (frontend) containerize
the API and Streamlit as separate services via `docker-compose.yml`, with
persistent volume mounts for the vector store and uploaded documents:

```powershell
docker-compose up --build
```

> **Status note:** the Docker configuration was written and reviewed but
> not fully verified end-to-end in this environment due to local setup
> constraints. The `uvicorn`/`streamlit` local run path above is fully
> tested and is the primary supported way to run this project.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check, returns indexed chunk count |
| POST | `/chat` | Main conversational endpoint |
| POST | `/documents/upload` | Upload a PDF/TXT/DOCX file |
| POST | `/documents/index` | Index a previously uploaded file into the vector store |
| GET | `/documents` | List indexed documents |

Example `/chat` request:
```json
{"message": "What medications is the patient taking?", "session_id": "demo"}
```
Response:
```json
{
  "answer": "FROM THE DOCUMENTS:\n- ...",
  "sources": [{"source": "sample_notes.txt", "chunk_index": 0, "distance": 0.257}],
  "agent": "reasoning_agent",
  "emergency_flag": false
}
```

## RAG Pipeline

Documents are loaded (PDF via `pypdf`, DOCX via `python-docx`, TXT directly),
cleaned (whitespace normalization only — clinical values like "145/90" and
"500 mg" are never altered), chunked (800 chars, 150 overlap, breaking on
whitespace to avoid splitting words), embedded via Gemini, and stored in
ChromaDB with cosine similarity. Retrieval applies a distance threshold
(`RELEVANCE_MAX_DISTANCE`) to decide whether retrieved content is actually
relevant enough to answer from, rather than always returning the nearest
match regardless of quality.

## Multi-Agent Architecture

A rule-based Supervisor Agent routes each question by keyword pattern
(extraction, summarization, document, or general) before falling back to
an optional LLM classifier — this keeps routing fast and free in the common
case. LangGraph orchestrates the resulting flow as a graph with conditional
edges, maintaining per-session conversation memory for follow-up questions.
Every node that calls an LLM is wrapped so a failure (e.g., quota exhaustion)
produces a graceful fallback message rather than a crash, with the failure
logged by exception type only — never by message content, to avoid logging
sensitive input.

## Testing

```powershell
pytest -v                                    # unit + API tests (no API quota used)
$env:RUN_INTEGRATION = "1"
pytest tests/test_workflow_integration.py -v  # real Gemini calls, uses quota
```

39 of 40 tests pass without any API calls, using a `FakeLLM` and a fake
embedder with an isolated temp ChromaDB instance. Integration tests
exercise the real Gemini API and are run manually/sparingly given the
free-tier daily quota.

## Safety Considerations

- Every response includes a disclaimer: this is not medical advice
- Regex-based emergency detection (deliberately over-sensitive — false
  positives are preferred over missed genuine emergencies) prepends a
  banner directing users to real emergency services
- Document-grounded answers never invent facts not present in retrieved
  context; the Reasoning Agent explicitly separates "FROM THE DOCUMENTS"
  from "EXPLANATION" and states "Nothing relevant found" when appropriate
- No question, answer, or document content is ever written to logs —
  only metadata (request ID, route, exception type, latency)
- File uploads are validated by extension and size before processing;
  filenames are sanitized against path traversal
- No real patient data is used anywhere — all sample documents are
  synthetic

## Future Enhancements

- OCR support for scanned/image-only PDFs
- True batch embedding calls (currently loops per chunk)
- LLM-based query rewriting for follow-up questions (currently rule-based)
- Authentication and per-user session isolation for multi-user deployment
- Migration path to PostgreSQL + pgvector for production scale
- Rate limiting
- CI/CD via GitHub Actions running the pytest suite on every push
- Separate, smaller `requirements.txt` for the frontend Docker image

## Author

Renu Prasath — built as a portfolio project demonstrating agentic AI,
RAG, and multi-agent orchestration patterns for AI/ML engineering roles.