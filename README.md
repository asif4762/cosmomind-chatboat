```markdown
# PDF‑QA with Ollama (Llama 3.1:8B) — OCR + UI + Incremental + Multi‑Model (v4)

Local RAG over your PDFs with:
- **OCR** (auto/pytesseract/ocrmypdf)
- **Streamlit UI**
- **Incremental ingest** (add-only)
- **Multi‑model** support: *router* and *consensus ensemble*
- **Text Summarizer** (summarize content directly from websites or pasted text)

## New in v4 (multi-model & summarizer)
- **Router** (`ENSEMBLE_MODE=router`): heuristics pick a model per question.
- **Consensus** (`ENSEMBLE_MODE=consensus`): ask 2–3 models, then a **judge** fuses a final answer using only context.
- **Summarizer**: A new feature for summarizing the content of websites or pasted text. Utilize Ollama’s capabilities to create a concise summary.

### Configure models in `.env`
```

LLM_MODEL=llama3.1:8b
LLM_MODELS=llama3.1:8b,gemma2:9b,mistral:7b
ENSEMBLE_MODE=off     # or router / consensus
JUDGE_MODEL=llama3.1:8b

````
Pull them with:
```bash
ollama pull llama3.1:8b
ollama pull gemma2:9b
ollama pull mistral:7b
ollama pull nomic-embed-text
````

### Summarizer Setup

To summarize text or websites, you can now use the **Summarizer** feature. It can summarize either:

1. **Website content** by providing a URL.
2. **Pasted text** directly in the app.

### Commands

```bash
# Ingest (full / incremental)
python -m app.ingest
python -m app.incremental_ingest

# Ask (CLI / API / UI)
python -m app.cli
uvicorn app.server:app --reload --port 8000
streamlit run app/ui_streamlit.py
```

### Server API additions

`POST /ask` now accepts optional fields:

```json
{
  "question": "string",
  "top_k": 8,
  "mode": "off|router|consensus",
  "models": ["llama3.1:8b","gemma2:9b","mistral:7b"],
  "judge_model": "llama3.1:8b"
}
```

For **Summarizer**, use `POST /summarize`:

```json
{
  "text": "string",
  "mode": "website|text",
  "url": "https://example.com"  # Optional: provide the URL if mode is 'website'
}
```

## Notes

* Router is simple heuristics; tune in `app/query_multi.py`.
* Consensus asks each model with the **same retrieved context**, then a judge writes the final answer with citations. If context lacks the info, it must return `I don't know from these PDFs.`
* If a model isn't pulled, Ollama will error; just remove it from `LLM_MODELS`.
* **Summarizer** requires the `requests` and `beautifulsoup4` libraries for fetching and parsing websites.

### Summarizer Features

* **Summarize Websites**: Provide a URL, and the summarizer will generate a short, concise summary of the content.
* **Summarize Text**: Paste text, and the summarizer will condense it into a summary for easy digestion.
