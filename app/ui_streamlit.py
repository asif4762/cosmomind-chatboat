import sys
from pathlib import Path
import os
import time
import re
import streamlit as st

# Add the root directory to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the query and other modules after modifying sys.path
from app.query import ask as ask_single
from app.query_multi import ask_router, ask_consensus, LLM_MODELS, JUDGE_MODEL
from app.ingest import main as ingest_main, DATA_DIR
from app.incremental_ingest import main as incr_main
from app.summarizer import display_summary  # Import the summarizer logic

# page_icon = str(ICON_PATH) if ICON_PATH.exists() else None

st.set_page_config(page_title="CosmoMinds Chatbot", page_icon="data:image/jpeg;base64,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SP1iNOcRHwb45Xd8V/Dl3NYXH5fG9OR5YsZaw6L5lVwUlgWIHUprHYkep9xqecDgNwHhbMOaVW+78RZ5PUG1fbWKqqQV9T8i71sD5j6DcpngfFvJYClcPMtpRjs1/I9W/chHBAP3A7zBzviHOznV8zJsyCm+gMQETfr0ooAUn7CTxz1nhd8drawB/wB3jf6LJ/8A0Gnf8AeMeOyeLwHzsvFrysIlNX31JYLErNQtAYjfVW/r6bY/SUMviLOGH+CGVaMQhh+G2PL0X6yNa/tEmcuWdLOcp5F7/CbnDn87zeUSem1KPKB32pVytY17HpUE/cmZvg3/AJ28Uf66n/2MiUnw3OZmGztiZFmO1gCuayAWAOwDsfczLxvibkMay+zHy7arMpg97oRu1+ot1N29du5/UyTpTzhYvC6fgx/knN/7Ryv7gldfA7/P2L/ocn/4TI5xvifkMZbVx8y2lbnay0IRp3I0WPb1mlxfI34tq3Y1rU2qGC2IfmAI0R/CdbJ5+0m8cL64bla6fGvLUWEKcvHwxUT72V46N0D7lWY/7si/OfCbkbectsUVnDyMt8l8k2IDXW9vmOhT1LDbAaGj27iVdn8nkX3nIuuey9ihNxP7TaABTsehAUfwnVzfG3L345xruQvell6GQsNuutFXcDqYEeuyd+8njtHS74ntbXx1xHvt4SmuxKrLcnISu2x2REchOkllBK99dx76nSxMzxVh5GNRdjY/KUMKxZlVEV2IS2mDMWG+kd99HffrvcobmvEOdmhBl5VmQKyxQWEEKW9daHvoTqYXxD5umoVV8jd0KNAOKrGA+nU6k/zk8c4xwb4yl/8AhCYWLXnYdlSouRfTc2SF0CyqwFVjAe5/aDfv0a9pVM2M/Nuvte6+17rXO3ssYs7fTufYfymvNKxiMM7TmSIidIREQEREoRESBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQP/Z")

# Login page simulation
# def login_page():
#     st.title("Welcome to CosmoMinds Chatbot!")
#     if st.button("Login"):
#         st.session_state.logged_in = True  # Set session state for login
#         st.success("You are now logged in!")
#         st.experimental_rerun()  # Rerun the app to show the next page

# Main content page for summarizer and question-answering
def main_page():
    st.title("Ask any question about Terra — hope I can help you")

    # Summarizer and model logic
    tab1, tab2 = st.tabs(["Ask a Question", "Summarize Text"])

    with tab1:
        st.header("Ask a Question")
        q = st.text_input("Your question", placeholder="e.g., Summarize procurement rules across the PDFs.")
        top_k = st.slider("Top-K passages", min_value=3, max_value=12, value=int(os.getenv("TOP_K", "5")), step=1)

        mode = st.radio("Mode", options=["off", "router", "consensus"], index=0, horizontal=True)
        models_default = [m for m in LLM_MODELS][:3]
        models_sel = st.multiselect("Models (for router/consensus)", options=LLM_MODELS, default=models_default)

        if mode == "consensus":
            judge_default = JUDGE_MODEL if JUDGE_MODEL in LLM_MODELS else (LLM_MODELS[0] if LLM_MODELS else "")
            judge_sel = st.selectbox("Judge model (consensus)", options=LLM_MODELS, index=(LLM_MODELS.index(judge_default) if judge_default in LLM_MODELS else 0))
        else:
            judge_sel = None

        if st.button("Ask") and q.strip():
            with st.spinner("Thinking..."):
                try:
                    if mode == "router":
                        res = ask_router(q, k=top_k, models=models_sel or None)
                    elif mode == "consensus":
                        res = ask_consensus(q, k=top_k, models=models_sel or None, judge_model=judge_sel)
                    else:
                        res = ask_single(q, k=top_k)
                except Exception as e:
                    st.error(f"Backend error: {e}")
                    raise

            st.subheader("Answer")
            st.write(res.get("answer", "(no answer)"))

    with tab2:
        st.header("Summarize Text")
        summarize_input = st.text_area("Paste the text to summarize", height=200)
        
        if st.button("Summarize") and summarize_input.strip():
            with st.spinner("Summarizing..."):
                summarized_content = display_summary(summarize_input)
                st.subheader("Summary")
                st.write(summarized_content)
main_page()
