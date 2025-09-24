import os
from app.query import ask as ask_single
from app.query_multi import ask_router, ask_consensus

MODE = os.getenv("ENSEMBLE_MODE", "off").lower()

BANNER = f"""PDF QA (v4) — Mode: {MODE.upper()}
Type your questions. Ctrl+C to exit.
"""

def main():
    print(BANNER)
    while True:
        try:
            q = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not q.strip():
            continue
        if MODE == "router":
            res = ask_router(q)
        elif MODE == "consensus":
            res = ask_consensus(q)
        else:
            res = ask_single(q)
        print("\nAssistant:\n" + res["answer"])
if __name__ == "__main__":
    main()
