import os
from typing import List, Dict, Tuple

from app.utils.ollama_client import OllamaClient
from app.query import retrieve, make_prompt, TOP_K

LLM_MODELS = [m.strip() for m in os.getenv("LLM_MODELS", "llama3.1:8b").split(",") if m.strip()]
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama3.1:8b")

def ask_with_model(question: str, model: str, k: int = TOP_K) -> dict:
    client = OllamaClient()
    retrieved, _ = retrieve(question, client, k=k)
    messages, sources = make_prompt(question, retrieved)
    answer = client.chat(model=model, messages=messages)
    return {"answer": answer, "sources": sources, "model": model}

def route_model(question: str, models: List[str] = None) -> str:
    models = models or LLM_MODELS
    q = question.lower()
    long_q = len(question) > 220
    complex_terms = any(t in q for t in ["compare","contrast","trade-off","why","how","analyz","synthesize","across","multiple","summarize thoroughly"])
    if (long_q or complex_terms) and len(models) >= 2:
        return models[1]
    return models[0]

def ask_router(question: str, k: int = TOP_K, models: List[str] = None) -> dict:
    model = route_model(question, models)
    return ask_with_model(question, model, k)

def ask_consensus(question: str, k: int = TOP_K, models: List[str] = None, judge_model: str = None) -> dict:
    models = models or LLM_MODELS[:3]
    judge_model = judge_model or JUDGE_MODEL
    client = OllamaClient()
    retrieved, _ = retrieve(question, client, k=k)
    messages, sources = make_prompt(question, retrieved)

    candidates: List[Dict] = []
    for m in models:
        ans = client.chat(model=m, messages=messages)
        candidates.append({"model": m, "answer": ans})

    cand_lines = []
    for i, c in enumerate(candidates, start=1):
        cand_lines.append(f"[Candidate {i} — {c['model']}]\n{c['answer']}")
    judge_system = '''You are a strict judge. Use ONLY the provided context to produce a final answer.
If the context lacks the answer, respond EXACTLY: "I don't know from these PDFs."'''
    judge_user = (
        messages[1]["content"] + "\n\n"
        "Candidate answers to consider (choose or synthesize one final answer using only the context):\n\n"
        + "\n\n".join(cand_lines)
    )
    final_answer = client.chat(model=judge_model, messages=[
        {"role": "system", "content": judge_system},
        {"role": "user", "content": judge_user},
    ])

    return {"answer": final_answer, "sources": sources, "candidates": candidates, "judge_model": judge_model}
