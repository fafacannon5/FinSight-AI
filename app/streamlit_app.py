"""Minimal demo chatbot (M11, owner J). Run: streamlit run app/streamlit_app.py

Final config A (wiki/decisions/final_configs.md): MiniLM 384-d, exact TorchKNN over the full FiQA corpus,
top-k = 5, refuse if the top-1 cosine < τ = 0.60, otherwise answer with the fixed LLM and cite doc ids.
Same pipeline as the E3 experiments: docs cut to DOC_CHARS characters, prompt from rag.build_prompt.
Needs data/fiqa_corpus_minilm.npy (created by scripts/run_experiment.py) and GROQ_API_KEY in .env.
"""
import sys
import time
from pathlib import Path

import streamlit as st

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from finsight.data import load_beir  # noqa: E402
from finsight.embed import MODEL_NAME, load_or_encode  # noqa: E402
from finsight.rag import REFUSAL_TEXT, build_prompt, cited_ids, generate_with_usage, should_refuse  # noqa: E402
from finsight.retrieval import TorchKNN  # noqa: E402

FINAL_K = 5        # E3 run 001
FINAL_TAU = 0.60   # E3 run 002 (config A)
DOC_CHARS = 1000   # same cut as scripts/e3_generate.py
EXAMPLES = [       # FiQA dev questions + off-topic ones; top-1 cosine in brackets (config A)
    "Can gold prices vary between two places or country at the same time?",  # 0.747 → answers
    "Borrow from 401k for down payment on rental property?",                  # 0.660 → answers
    "Can stock brokerage firms fail?",                                        # 0.578 → refused, though answerable
    "What is a good recipe for sourdough bread?",                             # 0.448 → refused
    "Who won the 2018 FIFA World Cup?",                                       # 0.279 → refused
]


@st.cache_resource(show_spinner="Loading FiQA corpus and index (first run only)…")
def load_system():
    """Corpus dict, doc ids in row order, the encoder and a TorchKNN index over all 57,638 docs."""
    from sentence_transformers import SentenceTransformer

    corpus, _, _ = load_beir("fiqa", "dev")  # the corpus is the same for every split; dev avoids the test queries
    doc_ids = sorted(corpus)
    emb = load_or_encode(str(REPO / "data" / "fiqa_corpus_minilm.npy"), [corpus[d] for d in doc_ids], ids=doc_ids)
    # CPU on purpose: on Apple MPS every query paid ~0.7 s inside Streamlit; on CPU it is ~10 ms
    return corpus, doc_ids, SentenceTransformer(MODEL_NAME, device="cpu"), TorchKNN().fit(emb)


def retrieve(question: str, k: int):
    """[(doc_id, cosine), ...] best first, plus encode and search times in ms (search = what E1/E2 measure)."""
    corpus, doc_ids, model, index = load_system()
    t0 = time.perf_counter()
    q = model.encode([question], normalize_embeddings=True, convert_to_numpy=True,  # same as embed.encode
                     show_progress_bar=False)
    t1 = time.perf_counter()
    scores, idx = index.search(q.astype("float32"), k)
    t2 = time.perf_counter()
    hits = [(doc_ids[j], float(s)) for j, s in zip(idx[0], scores[0])]
    return hits, (t1 - t0) * 1000, (t2 - t1) * 1000


st.set_page_config(page_title="FinSight-AI", page_icon="💬")
st.title("FinSight-AI — financial QA with RAG")
st.caption("Answers come only from FiQA forum posts and cite them. If retrieval finds nothing close enough, it refuses.")

with st.sidebar:
    st.header("Settings")
    st.markdown(f"Final config A: MiniLM 384-d, exact KNN, **k = {FINAL_K}**, **τ = {FINAL_TAU:.2f}**  \n"
                "LLM: `openai/gpt-oss-120b` (Groq), temperature 0")
    k = st.slider("top-k documents given to the LLM", 1, 10, FINAL_K)
    tau = st.slider("refusal threshold τ (top-1 cosine)", 0.30, 0.90, FINAL_TAU, 0.01)
    if (k, tau) != (FINAL_K, FINAL_TAU):
        st.warning("Changed from the final config — for exploration only.")

with st.expander("Try an example", expanded=True):
    for ex in EXAMPLES:
        if st.button(ex):  # sets the text box below, then Streamlit reruns the script
            st.session_state["question"] = ex
question = st.text_input("Your question", key="question", placeholder="e.g. Should I pay off debt or invest?")

if question:
    corpus = load_system()[0]
    hits, encode_ms, search_ms = retrieve(question, k)
    top1 = hits[0][1]
    st.caption(f"Top-1 cosine **{top1:.3f}** vs τ {tau:.2f} · encode {encode_ms:.0f} ms · search {search_ms:.1f} ms")

    if should_refuse(top1, tau):
        st.warning(REFUSAL_TEXT)
        st.caption("Refused before calling the LLM: no document is similar enough to the question.")
        cited = []
    else:
        docs = [(d, corpus[d][:DOC_CHARS]) for d, _ in hits]
        with st.spinner("Generating answer…"):
            t0 = time.perf_counter()
            try:
                answer, usage = generate_with_usage(build_prompt(question, docs), max_tokens=400)
            except Exception as e:  # missing key, network, rate limit: show it instead of a stack trace
                st.error(f"LLM call failed: {e}")
                st.stop()
            llm_s = time.perf_counter() - t0
        st.markdown(answer)
        cited = cited_ids(answer, [d for d, _ in hits])
        st.caption(f"LLM {llm_s:.1f} s · {usage.get('total_tokens', '?')} tokens · cites {len(cited)} of {k} documents")

    st.subheader("Retrieved documents")
    for rank, (d, s) in enumerate(hits, 1):
        mark = " · cited ✅" if d in cited else ""
        with st.expander(f"#{rank} [{d}] cosine {s:.3f}{mark}"):
            st.write(corpus[d][:DOC_CHARS] + ("…" if len(corpus[d]) > DOC_CHARS else ""))
