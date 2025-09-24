import os
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
from pypdf import PdfReader
from dotenv import load_dotenv
from tqdm import tqdm

from app.utils.ollama_client import OllamaClient
from app.utils.hash_utils import make_uid
from app.ocr import page_needs_ocr, ocr_with_pytesseract, ocrmypdf_available, ocr_with_ocrmypdf

load_dotenv()

DATA_DIR = Path("data")
STORE_DIR = Path("store")
STORE_DIR.mkdir(exist_ok=True)

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1200))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
OCR_MODE = os.getenv("OCR_MODE", "auto").lower()
OCR_LANGS = os.getenv("OCR_LANGS", "eng")

def extract_pdf_text_native(pdf_path: Path) -> List[Dict]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append({"page": i, "text": text})
    return pages

def extract_pdf_text_with_ocr(pdf_path: Path) -> List[Dict]:
    mode = OCR_MODE
    if mode == "off":
        return extract_pdf_text_native(pdf_path)

    if mode == "ocrmypdf":
        if not ocrmypdf_available():
            print("WARN: OCR_MODE=ocrmypdf but 'ocrmypdf' not found. Falling back to 'pytesseract'.")
            mode = "pytesseract"
        else:
            try:
                searchable_pdf = ocr_with_ocrmypdf(pdf_path, lang=OCR_LANGS)
                return extract_pdf_text_native(searchable_pdf)
            except Exception as e:
                print(f"WARN: ocrmypdf failed ({e}). Falling back to 'pytesseract'.")
                mode = "pytesseract"

    if mode == "pytesseract":
        ocr_texts = ocr_with_pytesseract(pdf_path, dpi=300, lang=OCR_LANGS)
        return [{"page": i+1, "text": t} for i, t in enumerate(ocr_texts)]

    native_pages = extract_pdf_text_native(pdf_path)
    pages_texts = [p["text"] for p in native_pages]
    need_idx = [i for i, txt in enumerate(pages_texts) if page_needs_ocr(txt)]
    if need_idx:
        ocr_pages = ocr_with_pytesseract(pdf_path, dpi=300, lang=OCR_LANGS)
        for i in need_idx:
            pages_texts[i] = ocr_pages[i] if i < len(ocr_pages) else pages_texts[i]
    return [{"page": i+1, "text": t} for i, t in enumerate(pages_texts)]

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = text.replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.splitlines())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            soft = text[end:end+200]
            bump_nl = soft.find("\n")
            bump_dot = soft.find(". ")
            bumps = [b for b in [bump_nl, bump_dot] if b != -1]
            if bumps:
                end += min(bumps) + 1
                chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks

def build_corpus() -> tuple[list[str], list[dict]]:
    texts: list[str] = []
    metas: list[dict] = []

    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No PDFs found in ./data. Please add files and retry.")

    for pdf in pdfs:
        pages = extract_pdf_text_with_ocr(pdf)
        for p in pages:
            page_num = p["page"]
            page_text = p["text"]
            if not page_text:
                continue
            for chunk in chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP):
                meta = {"doc": pdf.name, "path": str(pdf), "page": page_num}
                texts.append(chunk)
                metas.append(meta)
    if not texts:
        raise SystemExit("No extractable text found in PDFs (even after OCR).")
    return texts, metas

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    import numpy as np
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms

def main():
    import numpy as np, faiss
    client = OllamaClient()
    if not client.is_alive():
        raise SystemExit("Ollama is not reachable. Ensure it's running and OLLAMA_HOST is correct.")

    print(f"Using OCR mode: {OCR_MODE} (langs={OCR_LANGS})")
    print("Building corpus from PDFs...")
    texts, metas = build_corpus()

    print(f"Embedding {len(texts)} chunks with '{EMBED_MODEL}' via Ollama...")
    from tqdm import tqdm
    vecs_list = []
    for t in tqdm(texts):
        vecs_list.append(client.embed(t, EMBED_MODEL))
    vecs = np.array(vecs_list, dtype="float32")
    vecs = l2_normalize(vecs)

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim) 
    index.add(vecs)

    faiss.write_index(index, str(STORE_DIR / "index.faiss"))

    with open(STORE_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for meta, text in zip(metas, texts):
            uid = make_uid(meta, text)
            rec = {"uid": uid, "meta": meta, "text": text}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    manifest = {
        "embedding_model": EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "vector_count": int(vecs.shape[0]),
        "vector_dim": int(vecs.shape[1]),
        "ocr_mode": OCR_MODE,
        "ocr_langs": OCR_LANGS,
    }
    with open(STORE_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n✅ Ingestion complete. Index saved to ./store")

if __name__ == "__main__":
    main()
