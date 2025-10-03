# PDF-QA with Ollama (v4) — NASA Space Apps Challenge 🚀

![Project Screenshot](https://i.ibb.co.com/4Rq2t8ns/Screenshot-2025-10-03-at-3-34-16-PM.png)

This repository presents **PDF-QA with Ollama (Llama 3.1:8B)** — a local Retrieval-Augmented Generation (RAG) system designed for efficient question answering, summarization, and document exploration.  
It was developed as part of the **NASA Space Apps Challenge**, focusing on bridging AI with space and Earth science data.

---

## 🔹 Features

- **OCR Support**  
  Handles both native and scanned PDFs using `pytesseract` or `ocrmypdf`.

- **Incremental Ingest**  
  Add new PDFs without re-processing the entire dataset.

- **Multi-Model Support**  
  - **Router Mode** → Heuristics automatically select the most suitable model per query.  
  - **Consensus Mode** → Multiple models provide answers, then a judge fuses them into a final response.  

- **Streamlit UI**  
  Simple, user-friendly interface for exploration and Q&A.

- **Summarizer**  
  Summarize:
  - Entire **websites** (via `requests` + `beautifulsoup4`).
  - Any **pasted text** directly inside the app.

---

## 🔹 Environment Setup

### `.env` Configuration
```bash
LLM_MODEL=llama3.1:8b
LLM_MODELS=llama3.1:8b,gemma2:9b,mistral:7b
ENSEMBLE_MODE=off     # or router / consensus
JUDGE_MODEL=llama3.1:8b
```

### Model Pull
```bash
ollama pull llama3.1:8b
ollama pull gemma2:9b
ollama pull mistral:7b
ollama pull nomic-embed-text
```

---

## 🔹 Usage

### Ingest PDFs
```bash
# Full ingest
python -m app.ingest

# Incremental ingest
python -m app.incremental_ingest
```

### Run Q&A
```bash
# CLI
python -m app.cli

# API
uvicorn app.server:app --reload --port 8000

# Streamlit UI
streamlit run app/ui_streamlit.py
```

---

## 🔹 API Endpoints

### Ask a Question
```json
POST /ask
{
  "question": "Explain satellite NDVI readings",
  "top_k": 8,
  "mode": "router",
  "models": ["llama3.1:8b","gemma2:9b","mistral:7b"],
  "judge_model": "llama3.1:8b"
}
```

### Summarize Content
```json
POST /summarize
{
  "text": "NASA Terra satellite data analysis",
  "mode": "website|text",
  "url": "https://example.com"
}
```

---

## 🔹 Notes

- **Consensus mode** ensures reliability by combining multiple model outputs.  
- **Router mode** is fast but heuristic-based.  
- **Summarizer** is ideal for condensing large reports or web content.  
- If a model isn’t installed locally, remove it from `LLM_MODELS` in `.env`.

---

## 🔹 Project Relevance to NASA Space Apps 🌍

This project demonstrates how **AI + RAG** can empower researchers, students, and innovators to **explore large NASA datasets** (e.g., Terra’s MISR, MODIS, CERES, and ASTER instruments) with ease.  
By combining **multi-model reasoning** and **summarization**, users can extract insights faster — from **satellite NDVI vegetation data** to **climate science reports**.

---

✨ *Built for the NASA Space Apps Challenge, to make space and Earth data more accessible with AI.*  
