"""
app.py — StudyMind AI Streamlit Application.

Entry point: `streamlit run app.py`

Session state keys used across reruns:
    messages        : list[dict]   — chat history
    selected_model  : str          — Ollama model chosen in sidebar
    chunk_size      : int          — chunking parameter (can be overridden)
    last_sources    : list[dict]   — context chunks from the last RAG call
    generated_content: str         — content for the PDF exporter tab
    generated_title  : str         — title for the exported PDF
"""

import time
import logging
import tempfile
from pathlib import Path

import streamlit as st
import config
import vector_store as vs
import rag_engine as rag
import pdf_processor as pp
import pdf_exporter as pe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be the very first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — modern dark-themed design
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Base ───────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── App background ─────────────────────────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(20, 20, 40, 0.95) !important;
    border-right: 1px solid rgba(100, 100, 200, 0.2);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #c8b8ff;
}

/* ── Cards ──────────────────────────────────────────────────────────────── */
.studymind-card {
    background: rgba(30, 30, 60, 0.7);
    border: 1px solid rgba(100, 100, 200, 0.25);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin: 0.6rem 0;
    backdrop-filter: blur(10px);
}

/* ── Status badge ───────────────────────────────────────────────────────── */
.status-online  { color: #4ade80; font-weight: 600; }
.status-offline { color: #f87171; font-weight: 600; }
.status-warn    { color: #facc15; font-weight: 600; }

/* ── Chat bubbles ───────────────────────────────────────────────────────── */
.chat-user {
    background: linear-gradient(135deg, #3730a3, #4f46e5);
    color: #fff;
    border-radius: 18px 18px 4px 18px;
    padding: 0.8rem 1.1rem;
    margin: 0.4rem 0 0.4rem 20%;
    line-height: 1.5;
    box-shadow: 0 2px 12px rgba(79,70,229,0.3);
}
.chat-assistant {
    background: rgba(30, 30, 60, 0.85);
    color: #e2e8f0;
    border: 1px solid rgba(100, 100, 200, 0.25);
    border-radius: 18px 18px 18px 4px;
    padding: 0.8rem 1.1rem;
    margin: 0.4rem 20% 0.4rem 0;
    line-height: 1.6;
    box-shadow: 0 2px 12px rgba(0,0,0,0.25);
}

/* ── Tabs ───────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(20, 20, 40, 0.8) !important;
    border-radius: 10px;
    gap: 4px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px;
    color: #94a3b8;
    font-weight: 500;
    font-size: 0.92rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
}

/* ── Quick-tool buttons ─────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #312e81, #4c1d95) !important;
    color: #e0e7ff !important;
    border: 1px solid rgba(139, 92, 246, 0.4) !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    border-color: rgba(167, 139, 250, 0.6) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.35) !important;
}

/* ── Expanders (source citations) ───────────────────────────────────────── */
details > summary {
    color: #a78bfa;
    font-size: 0.85rem;
    cursor: pointer;
}
.streamlit-expanderContent {
    background: rgba(15, 15, 30, 0.6) !important;
    border: 1px solid rgba(100, 100, 200, 0.2) !important;
    border-radius: 8px;
    font-size: 0.83rem;
    color: #94a3b8 !important;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Input fields ───────────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: rgba(20, 20, 45, 0.9) !important;
    border: 1px solid rgba(100, 100, 200, 0.35) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.2) !important;
}

/* ── File uploader ──────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: rgba(25, 25, 50, 0.7) !important;
    border: 2px dashed rgba(124, 58, 237, 0.4) !important;
    border-radius: 10px !important;
}

/* ── Warning / info boxes ───────────────────────────────────────────────── */
.stAlert {
    border-radius: 8px !important;
}

/* ── Metric cards ───────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: rgba(30, 30, 60, 0.7) !important;
    border: 1px solid rgba(100, 100, 200, 0.2) !important;
    border-radius: 10px;
    padding: 0.5rem;
}

/* ── Scrollbar ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: rgba(15, 15, 30, 0.5); }
::-webkit-scrollbar-thumb { background: rgba(124, 58, 237, 0.5); border-radius: 3px; }

/* ── Hero title ─────────────────────────────────────────────────────────── */
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 0.3rem;
}
.hero-subtitle {
    color: #64748b;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
}

/* ── Source chip ────────────────────────────────────────────────────────── */
.source-chip {
    display: inline-block;
    background: rgba(79, 70, 229, 0.2);
    border: 1px solid rgba(79, 70, 229, 0.4);
    color: #a5b4fc;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.78rem;
    margin: 2px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _init_state():
    """Set default session state values on the very first run."""
    defaults = {
        "messages":          [],       # chat history
        "selected_model":    config.DEFAULT_MODEL,
        "chunk_size":        config.CHUNK_SIZE,
        "last_sources":      [],
        "generated_content": "",
        "generated_title":   "Study Notes",
        "ollama_ok":         None,     # None = not checked yet
        "ollama_models":     [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Ollama health check (cached for 30 s to avoid hammering the local server)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def _cached_health_check():
    return rag.check_ollama_health()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        # ── Branding ──────────────────────────────────────────────────────
        st.markdown(
            f"<div class='hero-title' style='font-size:1.6rem'>{config.APP_ICON} {config.APP_TITLE}</div>"
            f"<div class='hero-subtitle'>Local AI Study Assistant — v{config.APP_VERSION}</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Ollama status ──────────────────────────────────────────────────
        st.markdown("#### 🔌 System Status")
        ok, detail = _cached_health_check()
        st.session_state["ollama_ok"] = ok

        if ok:
            chunk_count = vs.get_chunk_count()
            st.markdown(
                f"<span class='status-online'>🟢 Ollama Connected</span><br>"
                f"<span class='status-online'>📚 {chunk_count} Chunks Indexed</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<span class='status-offline'>🔴 Ollama Offline</span>",
                unsafe_allow_html=True,
            )
            st.error(
                f"**Ollama not reachable.**\n\n"
                f"{detail}\n\n"
                "**Fix:** Open a terminal and run:\n```\nollama serve\n```"
            )

        st.divider()

        # ── Settings ───────────────────────────────────────────────────────
        st.markdown("#### ⚙️ Settings")

        # Model selector — populated from live Ollama instance
        models = rag.list_local_models() if ok else [config.DEFAULT_MODEL]
        models = [m for m in models if "embed" not in m.lower()]
        st.session_state["ollama_models"] = models
        selected = st.selectbox(
            "LLM Model",
            options=models,
            index=models.index(config.DEFAULT_MODEL) if config.DEFAULT_MODEL in models else 0,
            help="Models must be pulled with `ollama pull <name>`",
        )
        st.session_state["selected_model"] = selected

        st.session_state["chunk_size"] = st.slider(
            "Chunk Size (chars)",
            min_value=300,
            max_value=2000,
            value=st.session_state["chunk_size"],
            step=100,
            help="Larger chunks preserve more context per retrieval hit.",
        )

        st.divider()

        # ── PDF Upload ─────────────────────────────────────────────────────
        st.markdown("#### 📂 Upload PDFs")
        uploaded_files = st.file_uploader(
            "Drop PDF files here",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files and st.button("⚡ Index Uploaded PDFs", use_container_width=True):
            _index_uploaded_files(uploaded_files)

        # ── Indexed sources ────────────────────────────────────────────────
        sources = vs.get_indexed_sources()
        if sources:
            st.divider()
            st.markdown(f"#### 📚 Indexed Documents ({len(sources)})")
            for src in sources:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"📄 `{src}`")
                with col2:
                    if st.button("🗑", key=f"del_{src}", help=f"Remove {src}"):
                        removed = vs.delete_source(src)
                        st.success(f"Removed {removed} chunks for {src}.")
                        st.cache_data.clear()
                        st.rerun()

        # ── Danger zone ────────────────────────────────────────────────────
        st.divider()
        with st.expander("⚠️ Danger Zone"):
            if st.button("🗑 Reset Entire Vector DB", use_container_width=True):
                vs.reset_collection()
                st.warning("Vector database wiped.")
                st.cache_data.clear()
                st.rerun()


def _index_uploaded_files(files):
    """Save uploaded files to a temp dir, process, and index each one."""
    progress = st.progress(0, text="Starting indexing…")
    total_chunks = 0
    errors = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, f in enumerate(files):
            progress.progress((i) / len(files), text=f"Processing {f.name}…")
            tmp_path = Path(tmpdir) / f.name

            # Write bytes to disk so pdf_processor can open the file
            tmp_path.write_bytes(f.read())

            try:
                chunks, warning = pp.process_pdf(str(tmp_path))
                if warning:
                    st.warning(f"**{f.name}:** {warning}")
                if chunks:
                    added = vs.add_chunks(chunks)
                    total_chunks += added
            except Exception as e:
                errors.append(f"{f.name}: {e}")
                logger.error("Failed to process %s: %s", f.name, e)

    progress.progress(1.0, text="Done!")
    time.sleep(0.5)
    progress.empty()

    if total_chunks > 0:
        st.success(f"✅ Indexed **{total_chunks}** new chunks from {len(files)} file(s).")
    if not errors:
        pass
    else:
        for err in errors:
            st.error(f"❌ {err}")

    # Bust the cached health-check so the chunk count updates
    st.cache_data.clear()
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Source citations widget (reused across tabs)
# ─────────────────────────────────────────────────────────────────────────────

def render_source_citations(sources: list[dict], label: str = "🔍 Retrieved Context Sources"):
    """Render a collapsible accordion showing all source chunks."""
    if not sources:
        return

    with st.expander(label, expanded=False):
        for i, src in enumerate(sources, 1):
            badge_color = "#f87171" if src.get("low_conf") else "#4ade80"
            conf_label = "Low confidence" if src.get("low_conf") else "Good match"

            st.markdown(
                f"**Source {i}** — "
                f"<span class='source-chip'>📄 {src['source']}</span> "
                f"<span class='source-chip'>📖 Page {src['page']}</span> "
                f"<span style='color:{badge_color}; font-size:0.8rem;'>● {conf_label} "
                f"(dist={src['distance']})</span>",
                unsafe_allow_html=True,
            )
            st.code(src["text"][:600] + ("…" if len(src["text"]) > 600 else ""), language="")
            if i < len(sources):
                st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Chat with Documents
# ─────────────────────────────────────────────────────────────────────────────

def render_chat_tab():
    st.markdown("<div class='hero-title'>💬 Chat with Documents</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-subtitle'>Ask anything about your indexed PDFs. "
        "All answers are grounded in your documents — no hallucinations.</div>",
        unsafe_allow_html=True,
    )

    # Guard: Ollama must be online
    if not st.session_state.get("ollama_ok"):
        st.error("🔴 Ollama is offline. Start it with `ollama serve` and refresh.")
        return

    chunk_count = vs.get_chunk_count()
    if chunk_count == 0:
        st.info("📭 No documents indexed yet. Upload PDFs via the sidebar to get started.")
        return

    # ── Chat history ───────────────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["messages"]:
            css_class = "chat-user" if msg["role"] == "user" else "chat-assistant"
            icon = "🧑‍🎓" if msg["role"] == "user" else "🧠"
            st.markdown(
                f"<div class='{css_class}'>{icon} {msg['content']}</div>",
                unsafe_allow_html=True,
            )
            # Show sources beneath each assistant message
            if msg["role"] == "assistant" and msg.get("sources"):
                render_source_citations(msg["sources"])

    # ── Input row ──────────────────────────────────────────────────────────
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query = st.text_input(
            "Ask a question",
            placeholder="e.g. What are the main causes of the French Revolution?",
            label_visibility="collapsed",
            key="chat_input",
        )
    with col_btn:
        send = st.button("Send ➤", use_container_width=True)

    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

    # ── Process query ──────────────────────────────────────────────────────
    if send and query.strip():
        # Add user message to history
        st.session_state["messages"].append({"role": "user", "content": query})

        model = st.session_state["selected_model"]

        # Display streaming response
        with st.spinner("🧠 Thinking…"):
            stream_gen, sources, low_conf = rag.answer_query(
                query, model=model, stream=True
            )

        # Stream the response into a placeholder
        response_placeholder = st.empty()
        full_response = ""
        for token in stream_gen:
            full_response += token
            response_placeholder.markdown(
                f"<div class='chat-assistant'>🧠 {full_response}▌</div>",
                unsafe_allow_html=True,
            )
        # Final render without cursor
        response_placeholder.markdown(
            f"<div class='chat-assistant'>🧠 {full_response}</div>",
            unsafe_allow_html=True,
        )

        # Low-confidence warning
        if low_conf:
            st.warning(
                "⚠️ The retrieved context had low similarity to your question. "
                "The answer above may be less reliable. Try rephrasing."
            )

        # Persist to session history
        st.session_state["messages"].append({
            "role":    "assistant",
            "content": full_response,
            "sources": sources,
        })
        st.session_state["last_sources"] = sources

        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Quick Tools
# ─────────────────────────────────────────────────────────────────────────────

def render_quick_tools_tab():
    st.markdown("<div class='hero-title'>⚡ Quick Study Tools</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-subtitle'>One click to generate study materials from your indexed documents.</div>",
        unsafe_allow_html=True,
    )

    if not st.session_state.get("ollama_ok"):
        st.error("🔴 Ollama is offline.")
        return

    if vs.get_chunk_count() == 0:
        st.info("📭 No documents indexed yet.")
        return

    model = st.session_state["selected_model"]

    # ── Three action buttons ───────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='studymind-card'>", unsafe_allow_html=True)
        st.markdown("### 📝 Document Summary")
        st.caption("AI synthesises a structured overview of all indexed material.")
        if st.button("Generate Summary", key="btn_summary", use_container_width=True):
            with st.spinner("✍️ Generating summary…"):
                result, sources = rag.generate_summary(model=model)
            st.session_state["generated_content"] = result
            st.session_state["generated_title"]   = "Document Summary"
            st.session_state["quick_result"]      = ("summary", result, sources)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='studymind-card'>", unsafe_allow_html=True)
        st.markdown("### 🃏 Flashcards")
        st.caption("Creates 15 Q&A pairs to help you memorise key concepts.")
        if st.button("Create Flashcards", key="btn_flash", use_container_width=True):
            with st.spinner("🃏 Generating flashcards…"):
                result, sources = rag.generate_flashcards(model=model)
            st.session_state["generated_content"] = result
            st.session_state["generated_title"]   = "Study Flashcards"
            st.session_state["quick_result"]      = ("flashcards", result, sources)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='studymind-card'>", unsafe_allow_html=True)
        st.markdown("### 📋 Practice Quiz")
        st.caption("Builds a 10-question MCQ exam with an answer key.")
        if st.button("Build Quiz", key="btn_quiz", use_container_width=True):
            with st.spinner("📋 Generating quiz…"):
                result, sources = rag.generate_quiz(model=model)
            st.session_state["generated_content"] = result
            st.session_state["generated_title"]   = "Practice Quiz"
            st.session_state["quick_result"]      = ("quiz", result, sources)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Display last generated result ──────────────────────────────────────
    if "quick_result" in st.session_state:
        kind, result, sources = st.session_state["quick_result"]
        st.divider()
        kind_labels = {
            "summary":    "📝 Generated Summary",
            "flashcards": "🃏 Generated Flashcards",
            "quiz":       "📋 Generated Quiz",
        }
        st.markdown(f"### {kind_labels.get(kind, 'Result')}")
        st.markdown(result)

        render_source_citations(sources)

        # Quick path to PDF export
        st.info(
            "💡 Head to the **📄 PDF Exporter** tab to download this as a styled PDF.",
            icon="📄",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — PDF Exporter
# ─────────────────────────────────────────────────────────────────────────────

def render_pdf_exporter_tab():
    st.markdown("<div class='hero-title'>📄 PDF Exporter</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-subtitle'>Preview Markdown study notes and export them as a styled, downloadable PDF.</div>",
        unsafe_allow_html=True,
    )

    # ── Content source ─────────────────────────────────────────────────────
    col_title, col_clear = st.columns([4, 1])
    with col_title:
        doc_title = st.text_input(
            "Document title",
            value=st.session_state.get("generated_title", "Study Notes"),
            placeholder="e.g. Chapter 5 Review",
        )
    with col_clear:
        st.markdown("<div style='margin-top:1.8rem'>", unsafe_allow_html=True)
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state["generated_content"] = ""
        st.markdown("</div>", unsafe_allow_html=True)

    content = st.text_area(
        "Markdown content",
        value=st.session_state.get("generated_content", ""),
        height=380,
        placeholder=(
            "Paste or generate Markdown content here.\n\n"
            "Use the ⚡ Quick Tools tab to auto-generate a Summary, Flashcards, or Quiz, "
            "then come back here to export it."
        ),
        help="Supports # headings, **bold**, - bullets, and numbered lists.",
    )
    # Keep session state in sync
    st.session_state["generated_content"] = content

    # ── Preview ────────────────────────────────────────────────────────────
    if content.strip():
        st.divider()
        with st.expander("👁 Markdown Preview", expanded=True):
            st.markdown(content)

        # ── Export ────────────────────────────────────────────────────────
        st.divider()
        if st.button("🖨 Generate PDF", use_container_width=False):
            with st.spinner("Rendering PDF…"):
                try:
                    pdf_bytes = pe.markdown_to_pdf_bytes(content, doc_title=doc_title)
                    st.success("✅ PDF ready! Click below to download.")
                    st.download_button(
                        label="📥 Download PDF",
                        data=pdf_bytes,
                        file_name=f"{doc_title.replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        use_container_width=False,
                    )
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")
                    logger.error("PDF export error: %s", e)
    else:
        st.info(
            "📭 No content yet. Generate notes in the **⚡ Quick Tools** tab "
            "or paste your own Markdown above."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────────────────────

def main():
    render_sidebar()

    # ── App header ─────────────────────────────────────────────────────────
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(
            f"<div class='hero-title'>{config.APP_ICON} {config.APP_TITLE}</div>"
            "<div class='hero-subtitle'>Your local AI study partner — 100% offline, 100% private.</div>",
            unsafe_allow_html=True,
        )
    with c2:
        # Live stats
        chunk_count = vs.get_chunk_count()
        source_count = len(vs.get_indexed_sources())
        st.metric("📚 Chunks", chunk_count)
        st.metric("📄 Documents", source_count)

    st.divider()

    # ── Tabs ───────────────────────────────────────────────────────────────
    tab_chat, tab_tools, tab_export = st.tabs([
        "💬 Chat with Documents",
        "⚡ Quick Tools",
        "📄 PDF Exporter",
    ])

    with tab_chat:
        render_chat_tab()

    with tab_tools:
        render_quick_tools_tab()

    with tab_export:
        render_pdf_exporter_tab()


if __name__ == "__main__":
    main()
