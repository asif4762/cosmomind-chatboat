import os
import requests
from typing import List, Dict, Any

class OllamaClient:
    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        if self.base_url.endswith("/"):
            self.base_url = self.base_url[:-1]

    def embed(self, text: str, model: str) -> list[float]:
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": model, "prompt": text}
        r = requests.post(url, json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        return data["embedding"]

    def chat(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        url = f"{self.base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": int(os.getenv("NUM_CTX", "8192"))},
        }
        r = requests.post(url, json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            msg = data.get("message") or {}
            content = msg.get("content")
            if content:
                return content
        return data.get("response", "")

    def is_alive(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=10)
            return r.ok
        except Exception:
            return False
