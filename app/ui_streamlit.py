import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import time
import re
import streamlit as st

from app.query import ask as ask_single
from app.query_multi import ask_router, ask_consensus, LLM_MODELS, JUDGE_MODEL
from app.ingest import main as ingest_main, DATA_DIR
from app.incremental_ingest import main as incr_main

ICON_PATH = Path(__file__).resolve().parents[2] / "untitled folder 4" / "Cosmo-Minds-Nasa-Space-App-Challenge" / "public" / "como.png"
page_icon = str(ICON_PATH) if ICON_PATH.exists() else None

st.set_page_config(page_title="CosmoMinds Chatbot", page_icon=page_icon)

st.title("Ask any question about Terra — hope I can help you")
# st.caption("OCR + incremental ingest + multi-model router/consensus")

def highlight_snippet(snippet: str, query: str) -> str:
    terms = [t for t in re.findall(r"\w+", (query or "")) if len(t) > 2]
    if not terms:
        return snippet
    def repl(m):
        return f"<mark>{m.group(0)}</mark>"
    out = snippet
    for t in sorted(set(terms), key=len, reverse=True):
        out = re.sub(rf"(?i)\b{re.escape(t)}\b", repl, out)
    return out

def source_card(src: dict, query: str):
    doc = src.get("doc", "")
    page = src.get("page", "")
    path = src.get("path", "")
    snip = src.get("snippet", "")
    snip_html = highlight_snippet(snip, query)

    card_html = f"""
    <div style="border:1px solid #eaeaea; border-radius:14px; padding:12px 14px; margin:10px 0; box-shadow:0 1px 3px rgba(0,0,0,0.05);">
      <div style="display:flex; justify-content:space-between; gap:12px; align-items:center;">
        <div style="font-weight:600;">📄 {doc}
          <span style="margin-left:8px; font-size:12px; background:#eef; color:#223; padding:2px 8px; border-radius:999px;">p.{page}</span>
        </div>
      </div>
      <div style="margin-top:8px; font-size:13px; line-height:1.5; color:#222;">
        {snip_html}
      </div>
      <div style="margin-top:10px; font-size:12px; color:#666;">
        <code style="background:#f6f6f6; padding:2px 6px; border-radius:6px;">{path}</code>
      </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

# with st.sidebar:
#     st.header("Ingestion")
#     st.write("Upload PDFs to add them into the `data/` folder.")
#     uploaded = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
#     if uploaded:
#         for f in uploaded:
#             path = DATA_DIR / f.name
#             with open(path, "wb") as out:
#                 out.write(f.read())
#         st.success(f"Saved {len(uploaded)} file(s) to data/")

#     if st.button("🔁 Full Rebuild Index"):
#         with st.spinner("Full ingest..."):
#             start = time.time()
#             ingest_main()
#             st.success(f"Done in {time.time()-start:.1f}s")

#     if st.button("➕ Incremental Ingest (Add Only)"):
#         with st.spinner("Embedding only NEW chunks..."):
#             start = time.time()
#             incr_main()
#             st.success(f"Done in {time.time()-start:.1f}s")

# st.header("Ask a question")
q = st.text_input("Your question", placeholder="e.g., Summarize procurement rules across the PDFs.")
top_k = st.slider("Top-K passages", min_value=3, max_value=12, value=int(os.getenv("TOP_K", "5")), step=1)

# mode = st.radio("Mode", options=["off", "router", "consensus"], index=0, horizontal=True)
# models_default = [m for m in LLM_MODELS][:3]
# models_sel = st.multiselect("Models (for router/consensus)", options=LLM_MODELS, default=models_default)

# if mode == "consensus":
#     judge_default = JUDGE_MODEL if JUDGE_MODEL in LLM_MODELS else (LLM_MODELS[0] if LLM_MODELS else "")
#     judge_sel = st.selectbox("Judge model (consensus)", options=LLM_MODELS, index=(LLM_MODELS.index(judge_default) if judge_default in LLM_MODELS else 0))
# else:
#     judge_sel = None

if st.button("Ask") and q.strip():
    with st.spinner("Thinking..."):
        try:
            # if mode == "router":
            #     res = ask_router(q, k=top_k, models=models_sel or None)
            # elif mode == "consensus":
            #     res = ask_consensus(q, k=top_k, models=models_sel or None, judge_model=judge_sel)
            # else:
                res = ask_single(q, k=top_k)
        except Exception as e:
            st.error(f"Backend error: {e}")
            raise

    st.subheader("Answer")
    st.write(res.get("answer", "(no answer)"))

    # if res.get("sources"):
    #     st.subheader("Sources")
    #     for src in res["sources"]:
    #         source_card(src, q)
