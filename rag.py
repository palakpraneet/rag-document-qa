"""
Document Q&A using RAG (Retrieval-Augmented Generation).

Two retrieval approaches are implemented for comparison:
1. Manual - embeddings computed via Gemini API, similarity via cosine similarity (built from scratch)
2. ChromaDB - a real vector database handling embedding + storage + search internally

Both feed retrieved context into Gemini for a grounded answer.
"""

import os
import json
import numpy as np
import chromadb
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


def load_and_chunk(filepath):
    """Splits a text file into chunks, one per non-empty line."""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]


def generate_answer(question, context_chunks):
    """Given retrieved chunks, asks Gemini to answer using ONLY that context."""
    context = "\n".join(context_chunks)
    prompt = f"""Answer the question using ONLY the information below.
If the information doesn't contain the answer, say "I don't have that information."

Context:
{context}

Question: {question}"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )
    return response.text


# ---------- Approach 1: Manual (embeddings + cosine similarity) ----------

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def embed_chunks(chunks):
    result = client.models.embed_content(model="gemini-embedding-001", contents=chunks)
    return [e.values for e in result.embeddings]


def find_top_k_chunks_manual(question, chunks, chunk_embeddings, k=3):
    question_embedding = client.models.embed_content(
        model="gemini-embedding-001", contents=question
    ).embeddings[0].values

    similarities = [cosine_similarity(question_embedding, emb) for emb in chunk_embeddings]
    top_indices = np.argsort(similarities)[::-1][:k]
    return [chunks[i] for i in top_indices]


def answer_with_manual_rag(question, chunks, chunk_embeddings, k=3):
    top_chunks = find_top_k_chunks_manual(question, chunks, chunk_embeddings, k=k)
    answer = generate_answer(question, top_chunks)
    return answer, top_chunks


# ---------- Approach 2: ChromaDB ----------

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="policy_docs")


def store_chunks_in_chroma(chunks):
    # Clear existing entries to avoid duplicate IDs on repeated runs
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
    collection.add(documents=chunks, ids=[f"chunk_{i}" for i in range(len(chunks))])


def answer_with_chroma_rag(question, k=3):
    results = collection.query(query_texts=[question], n_results=k)
    top_chunks = results["documents"][0]
    answer = generate_answer(question, top_chunks)
    return answer, top_chunks


# ---------- Demo ----------

if __name__ == "__main__":
    chunks = load_and_chunk("policy.txt")
    question = "Tell me about all leave-related policies including maternity and sick leave"

    print("=== Manual RAG (cosine similarity) ===")
    chunk_embeddings = embed_chunks(chunks)
    answer, retrieved = answer_with_manual_rag(question, chunks, chunk_embeddings)
    print("Retrieved:")
    for c in retrieved:
        print(" -", c)
    print(f"\nAnswer: {answer}\n")

    print("=== ChromaDB RAG ===")
    store_chunks_in_chroma(chunks)
    answer, retrieved = answer_with_chroma_rag(question)
    print("Retrieved:")
    for c in retrieved:
        print(" -", c)
    print(f"\nAnswer: {answer}")