import hashlib
from typing import Dict

def make_uid(meta: Dict, text: str) -> str:
    doc = meta.get("doc", "")
    page = str(meta.get("page", ""))
    payload = f"{doc}|{page}|{len(text)}|{text}".encode("utf-8", errors="ignore")
    return hashlib.sha1(payload).hexdigest()
