import os
import json
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import faiss
from dotenv import load_dotenv
from tqdm import tqdm

from app.utils.ollama_client import OllamaClient
from app.utils.hash_utils import make_uid
from app.ingest import build_corpus, l2_normalize, STORE_DIR, EMBED_MODEL

load_dotenv()

def load_existing_uids_and_index():
    index_path = STORE_DIR / "index.faiss"
    chunks_path = STORE_DIR / "chunks.jsonl"
    have_index = index_path.exists() and chunks_path.exists()
    uids: Set[str] = set()
    if chunks_path.exists():
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if "uid" in obj:
                    uids.add(obj["uid"])
                else:
                    uids.add(make_uid(obj["meta"], obj["text"]))
    index = faiss.read_index(str(index_path)) if have_index else None
    return index, uids

def main():
    client = OllamaClient()
    if not client.is_alive():
        raise SystemExit("Ollama is not reachable. Ensure it's running and OLLAMA_HOST is correct.")

    index, existing_uids = load_existing_uids_and_index()
    texts, metas = build_corpus()

    new_items: List[Dict] = []
    for meta, text in zip(metas, texts):
        uid = make_uid(meta, text)
        if uid not in existing_uids:
            new_items.append({"uid": uid, "meta": meta, "text": text})

    if not new_items:
        print("No new chunks detected. Nothing to do.")
        return

    print(f"Embedding {len(new_items)} NEW chunks...")
    vecs_list = []
    for rec in tqdm(new_items):
        vecs_list.append(client.embed(rec["text"], EMBED_MODEL))
    vecs = np.array(vecs_list, dtype="float32")
    vecs = l2_normalize(vecs)

    if index is None:
        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    faiss.write_index(index, str(STORE_DIR / "index.faiss"))

    with open(STORE_DIR / "chunks.jsonl", "a", encoding="utf-8") as f:
        for rec in new_items:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    manifest_path = STORE_DIR / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    manifest["vector_count"] = int(index.ntotal)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("✅ Incremental ingest complete. Index updated.")

if __name__ == "__main__":
    main()
