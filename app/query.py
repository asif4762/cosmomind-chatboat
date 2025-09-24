import os
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv

from app.utils.ollama_client import OllamaClient

load_dotenv()

STORE_DIR = Path("store")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
TOP_K = int(os.getenv("TOP_K", 5))


def load_index_and_chunks():
    index = faiss.read_index(str(STORE_DIR / "index.faiss"))
    chunks: list[Dict] = []
    with open(STORE_DIR / "chunks.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return index, chunks


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def retrieve(question: str, client: OllamaClient, k: int = TOP_K) -> Tuple[List[Dict], List[int]]:
    """Hybrid-lite retrieval: FAISS dense recall -> keyword re-rank."""
    # 1) embed query
    qv = np.array([client.embed(question, EMBED_MODEL)], dtype="float32")
    qv = l2_normalize(qv)

    # 2) search many candidates
    index, chunks = load_index_and_chunks()
    CAND_MULT = 6
    sims, idxs = index.search(qv, k * CAND_MULT)
    idxs = idxs[0].tolist()
    faiss_scores = sims[0].tolist()

    # 3) simple keyword score
    tokens = re.findall(r"\w+", question.lower())
    def kw_score(text: str):
        t = text.lower()
        return sum(t.count(tok) for tok in tokens if len(tok) > 2)

    candidates = []
    for i, s in zip(idxs, faiss_scores):
        rec = chunks[i]
        score = 0.75 * float(s) + 0.25 * float(kw_score(rec["text"]))  # weight blend
        candidates.append((score, rec, i))

    candidates.sort(key=lambda x: x[0], reverse=True)
    top = candidates[:k]
    items = [rec for _, rec, _ in top]
    final_idxs = [i for _, _, i in top]
    return items, final_idxs


def make_prompt(question: str, retrieved: List[Dict]) -> Tuple[list[dict], list[dict]]:
    # Build numbered context blocks and rich sources (with snippet + path)
    ctx_lines = []
    sources = []
    for i, rec in enumerate(retrieved, start=1):
        meta = rec["meta"]
        text = (rec["text"] or "").strip()
        src_label = f"[{i}] {meta['doc']} p.{meta['page']}"
        ctx_lines.append(f"{src_label}:\n{text}")
        sources.append({
            "n": i,
            "doc": meta["doc"],
            "page": meta["page"],
            "path": meta.get("path", ""),
            "snippet": (text[:320] + ("…" if len(text) > 320 else "")),
        })

    system = '''You are a helpful assistant. Use ONLY the provided context to answer.
If the needed info is not present in the context, respond EXACTLY with: "I don't know from these PDFs."
Do not include outside knowledge or guesses.'''

    user = (
        f"Question: {question}\n\n"
        f"Context blocks:\n\n" + "\n\n".join(ctx_lines)
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return messages, sources


def ask(question: str, k: int = TOP_K) -> dict:
    client = OllamaClient()
    retrieved, idxs = retrieve(question, client, k=k)
    messages, sources = make_prompt(question, retrieved)
    answer = client.chat(model=LLM_MODEL, messages=messages)
    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What are the key ideas across these PDFs?"
    out = ask(q)
    print("\n=== Answer ===\n")
    print(out["answer"]) 
    print("\nSources:")
    for s in out["sources"]:
        print(f"[{s['n']}] {s['doc']} p.{s['page']}")
