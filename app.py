"""
CS4241 — Introduction to Artificial Intelligence (2026)
RAG Chatbot: Ghana Elections + 2025 Budget Statement
Academic City University

Parts covered:
  A  Data Engineering & Preparation
  B  Custom Retrieval System (FAISS + hybrid search)
  C  Prompt Engineering (hallucination control, context window management)
  D  Full RAG Pipeline (with per-stage logging)
  E  Critical Evaluation (RAG vs pure LLM comparison tab)
  F  Architecture displayed in sidebar
  G  Innovation — Conversation Memory

Student: Maame Yaa Adumaba Appiah | Index: 10022200171
Course : CS4241 - Introduction to Artificial Intelligence | ACity 2026
"""
import os
import sys
import streamlit as st

# ── path fix ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from rag.index_builder import build_index, CACHE_FILE
from rag.pipeline import run_rag, run_pure_llm
from rag.memory import ConversationMemory

# ── constants ────────────────────────────────────────────────────────────────
DATA_DIR     = os.path.join(BASE_DIR, "data")
ELECTION_CSV = os.path.join(DATA_DIR, "Ghana_Election_Result.csv")
BUDGET_PDF   = os.path.join(DATA_DIR, "2025-Budget-Statement-and-Economic-Policy_v4.pdf")
DEFAULT_TOP_K = 5

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ACity RAG Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── clean minimal styling (no background image) ──────────────────────────────
st.markdown("""
<style>
    .stButton > button {
        background-color: #006400;
        color: white;
        border: none;
        border-radius: 6px;
    }
    .stButton > button:hover {
        background-color: #004d00;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/"
        "Academic_City_University_College_logo.png/220px-Academic_City_University_College_logo.png",
        use_container_width=True,
    )
    st.title("ACity RAG Assistant")
    st.caption("CS4241 · Intro to AI · 2026")
    st.caption("**Maame Yaa Adumaba Appiah**")
    st.caption("Index: 10022200171")

    st.divider()

    # Auto-load from Streamlit Cloud secrets if available
    _secret_key = ""
    try:
        _secret_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        pass

    api_key = st.text_input(
        "🔑 Anthropic API Key",
        value=_secret_key,
        type="password",
        help="Paste your Claude API key. On Streamlit Cloud set ANTHROPIC_API_KEY in Secrets.",
    )
    model = st.selectbox("🤖 Model", [
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-6",
        "claude-opus-4-7",
    ])
    top_k    = st.slider("📚 Top-K documents", 1, 10, DEFAULT_TOP_K)
    use_hybrid = st.toggle("🔀 Hybrid search (keyword + vector)", value=True)
    rebuild    = st.button("🔄 Rebuild Index")

    st.divider()
    st.subheader("🏗️ Architecture")
    st.markdown("""
```
User Query
    │
    ▼
[Query Expansion]
    │
    ▼
[FAISS Vector Store] ◄── Embeddings
    │                     (MiniLM-L6-v2)
    ▼
[Hybrid Re-rank]
    │
    ▼
[Context Selection]
    │
    ▼
[Prompt Builder] ◄── Conversation Memory
    │
    ▼
[Claude LLM]
    │
    ▼
Response + Logs
```
**Data sources:**
- 🗳️ Ghana Election Results CSV
- 📊 2025 Budget Statement PDF
""")

# ── session state ─────────────────────────────────────────────────────────────
if "memory"   not in st.session_state: st.session_state.memory   = ConversationMemory()
if "store"    not in st.session_state: st.session_state.store    = None
if "messages" not in st.session_state: st.session_state.messages = []

# ── index loader (no blocking spinner at top level) ───────────────────────────
@st.cache_resource
def load_store(force: bool = False):
    return build_index(ELECTION_CSV, BUDGET_PDF, force_rebuild=force)

# ── tabs — rendered IMMEDIATELY before any loading ───────────────────────────
tab_chat, tab_eval, tab_logs = st.tabs(["💬 Chat", "⚖️ RAG vs LLM", "📋 Logs"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.header("💬 Chat with Ghana Data")
    st.caption("Ask questions about Ghana Presidential Elections (1992–2020) or the 2025 Budget Statement.")

    # ── Load index INSIDE the tab so the UI always renders first ─────────────
    if rebuild:
        st.cache_resource.clear()
        st.session_state.store = None

    if st.session_state.store is None:
        with st.status("⏳ Loading knowledge index …", expanded=True) as status:
            try:
                st.write("📂 Reading data files …")
                st.session_state.store = load_store(force=rebuild)
                st.write("✅ Index loaded — 768 chunks ready!")
                status.update(label="✅ Knowledge index ready!", state="complete", expanded=False)
            except Exception as e:
                status.update(label="❌ Index failed to load", state="error")
                st.error(f"Error: {e}")
                st.stop()

    store = st.session_state.store

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask about elections or the 2025 budget …")

    if query:
        if not api_key:
            st.warning("⚠️ Please enter your Anthropic API key in the sidebar.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Retrieving and generating …"):
                result = run_rag(
                    query=query,
                    store=store,
                    api_key=api_key,
                    top_k=top_k,
                    use_hybrid=use_hybrid,
                    conversation_history=st.session_state.memory.get_recent(),
                    model=model,
                )
            st.markdown(result["response"])

            with st.expander("📄 Retrieved Chunks & Similarity Scores"):
                for i, chunk in enumerate(result["retrieved"], 1):
                    st.markdown(
                        f"**Chunk {i}** | Source: `{chunk['source']}` | "
                        f"Score: `{chunk['score']:.4f}`"
                    )
                    st.text(chunk["text"][:400] + ("…" if len(chunk["text"]) > 400 else ""))
                    st.divider()

            with st.expander("🔧 Final Prompt Sent to LLM"):
                st.code(result["prompt"], language="text")

            st.caption(
                f"⚡ Retrieval: {result['retrieval_ms']} ms | "
                f"🧠 Generation: {result['generation_ms']} ms"
            )

        st.session_state.memory.add_turn(query, result["response"])
        st.session_state.messages.append({"role": "assistant", "content": result["response"]})

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — PART E: RAG vs Pure LLM
# ════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.header("⚖️ Part E: RAG vs Pure LLM Evaluation")
    st.info(
        "Compare grounded (RAG) vs ungrounded (pure LLM) responses. "
        "This demonstrates hallucination control and factual accuracy improvements."
    )

    adversarial_examples = [
        "Who won the 2020 election in Volta Region and by how much?",
        "What is Ghana's projected GDP growth in the 2025 budget?",
        "Tell me about the NDC performance across all regions in 2016",
        "What revenue targets did the government set for 2025?",
        "Which party won the Ashanti Region in 2024?",           # adversarial — 2024 not in data
        "What did Nana Akufo-Addo say about education in the budget speech?",  # misleading
    ]

    eval_query  = st.selectbox("Choose an example query:", [""] + adversarial_examples)
    custom_q    = st.text_input("Or enter a custom query:")
    final_q     = custom_q.strip() or eval_query

    if st.button("🔍 Compare") and final_q:
        if not api_key:
            st.warning("⚠️ Please enter your Anthropic API key in the sidebar.")
            st.stop()
        if st.session_state.store is None:
            st.warning("⚠️ Please visit the Chat tab first to load the index.")
            st.stop()

        col_rag, col_llm = st.columns(2)
        store = st.session_state.store

        with col_rag:
            st.subheader("✅ RAG Response")
            with st.spinner("Running RAG pipeline …"):
                rag_result = run_rag(final_q, store, api_key, top_k=top_k,
                                     use_hybrid=use_hybrid, model=model)
            st.success(rag_result["response"])
            st.caption(f"Retrieved {len(rag_result['retrieved'])} chunks | "
                       f"Retrieval: {rag_result['retrieval_ms']} ms")
            with st.expander("📄 Retrieved evidence"):
                for c in rag_result["retrieved"]:
                    st.markdown(f"**{c['source']}** (score={c['score']:.3f})")
                    st.text(c["text"][:300])

        with col_llm:
            st.subheader("⚠️ Pure LLM (No Retrieval)")
            with st.spinner("Calling LLM directly …"):
                llm_response = run_pure_llm(final_q, api_key, model=model)
            st.warning(llm_response)
            st.caption("No document retrieval — answer from model training data only")

        st.divider()
        st.subheader("📊 Comparison Table")
        st.markdown("""
| Criterion | ✅ RAG | ⚠️ Pure LLM |
|-----------|--------|-------------|
| **Factual grounding** | Cites retrieved documents | Relies on training memory |
| **Hallucination risk** | Low — constrained to context | Higher — may confabulate |
| **Coverage** | Limited to indexed documents | Broad but unverifiable |
| **Transparency** | Shows retrieved chunks & scores | Opaque reasoning |
| **Year specificity** | Correct (hybrid boost) | May generalise incorrectly |
""")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — LOGS
# ════════════════════════════════════════════════════════════════════════════
with tab_logs:
    st.header("📋 Pipeline Logs")
    st.caption("Every query is logged with retrieval scores, prompt size, and latencies.")

    log_path = os.path.join(BASE_DIR, "logs", "pipeline_log.jsonl")
    if os.path.exists(log_path):
        import json
        entries = []
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        if entries:
            st.write(f"**{len(entries)} logged queries**")
            for entry in reversed(entries[-20:]):
                with st.expander(f"[{entry['timestamp'][:19]}]  {entry['query'][:80]}"):
                    st.json(entry)
        else:
            st.info("No logs yet — ask a question in the Chat tab first.")
    else:
        st.info("No log file found — ask a question in the Chat tab first.")
