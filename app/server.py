from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.query import ask as ask_single
from app.query_multi import ask_router, ask_consensus

app = FastAPI(title="PDF QA — OCR + Incremental + Multi-Model (v4)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    mode: Optional[str] = None          # off | router | consensus
    models: Optional[List[str]] = None  # e.g., ["llama3.1:8b","gemma2:9b","mistral:7b"]
    judge_model: Optional[str] = None

class AskResponse(BaseModel):
    answer: str
    sources: list[dict]

@app.post("/ask", response_model=AskResponse)
async def ask_api(req: AskRequest):
    k = req.top_k or 5
    mode = (req.mode or "off").lower()
    if mode == "router":
        res = ask_router(req.question, k=k, models=req.models)
        return AskResponse(answer=res["answer"], sources=res["sources"])
    if mode == "consensus":
        res = ask_consensus(req.question, k=k, models=req.models, judge_model=req.judge_model)
        return AskResponse(answer=res["answer"], sources=res["sources"])
    # default single-model behavior (uses LLM_MODEL from env)
    res = ask_single(req.question, k=k)
    return AskResponse(answer=res["answer"], sources=res["sources"])
