#!/usr/bin/env python3
import shutil
from pathlib import Path

STORE = Path("store")
if STORE.exists():
    shutil.rmtree(STORE)
STORE.mkdir(parents=True, exist_ok=True)
print("Store reset. You can now run a fresh full ingest: python -m app.ingest")
