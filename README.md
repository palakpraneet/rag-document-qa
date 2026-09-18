# RAG Document Q&A

A Retrieval-Augmented Generation (RAG) system that answers questions grounded in a document, instead of relying on the LLM's own (potentially outdated or hallucinated) knowledge.

Implements **two retrieval approaches** side by side for comparison:

1. **Manual** — embeddings and similarity search built from scratch using cosine similarity
2. **ChromaDB** — a real vector database handling embedding, storage, and similarity search internally

## Architecture

Document → Chunking → Embeddings → Vector Store →
User Question → Embed Question → Similarity Search → Top-K Chunks →
LLM (Gemini) generates answer using ONLY retrieved chunks

## Manual vs ChromaDB

| | Manual | ChromaDB |
| Similarity metric : Cosine similarity (higher = more similar) | Distance (lower = more similar) |
| Persistence : Recomputed every run | Persisted to disk |
| Embedding model : Gemini's `gemini-embedding-001` explicitly | Chroma's own default model |

Interestingly, the two approaches sometimes retrieve **different top chunks** for the same question — a reminder that retrieval quality depends on the embedding model used, not just the algorithm.

## Tech stack

- Google Gemini API (`google-genai`) — for generation and embeddings
- ChromaDB — vector database
- NumPy — manual cosine similarity calculations

## Setup

```bash
git clone https://github.com/palakpraneet/rag-document-qa.git
cd rag-document-qa
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` (see `.env.example`):
GEMINI_API_KEY=your_api_key_here

Run:

```bash
python rag.py
```

## Possible extensions

- Chunk by paragraph/semantic boundaries instead of by line, for larger real-world documents
- Add source citations (which chunk/section the answer came from)
- Support PDF/Word document ingestion
