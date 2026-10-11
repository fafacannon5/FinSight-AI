"""Minimal demo chatbot (M11, owner J). Run: streamlit run app/streamlit_app.py

Plan:
1. Load cached FiQA embeddings + corpus; build the best index from E1/E2 (cache with st.cache_resource).
2. Text input → encode query → search top-k → if should_refuse(top score, tau): show REFUSAL_TEXT.
3. Else build_prompt + generate; show the answer and an expander per cited document.
Keep it minimal: the demo is for the video, the research is in the experiments.
"""
import streamlit as st

st.title("FinSight-AI — financial QA with RAG")
st.info("Demo not implemented yet (milestone M11).")
