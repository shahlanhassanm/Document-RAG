"""
Advanced Local RAG System — Streamlit UI
Qwen3 + ChromaDB + Multi-Stage Retrieval
100% Local · Zero Cloud · Maximum Privacy
"""

import streamlit as st
import tempfile
import os
from core.document_processor import process_file, process_batch
from core.vector_store import build_vector_store, clear_store, store_exists
from core.rag_chain import ask
from core.memory import store_exchange, clear_memory, memory_exists
from core.llm import check_model, get_available_models
from core.logger import get_logger

log = get_logger("app")

st.set_page_config(
    page_title="Qwen RAG",
    layout="wide",
    page_icon="⬡",
    initial_sidebar_state="expanded",
)

# Disable telemetry
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

# ─── Premium Dark CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

    /* Hide Streamlit chrome */
    #MainMenu, footer, header,
    .stDeployButton,
    div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"] { display: none !important; }

    /* Background */
    .stApp { background: #080808 !important; }

    /* Typography */
    .stApp, .stApp * {
        font-family: 'Inter', -apple-system, sans-serif !important;
    }
    .stMarkdown p, .stMarkdown li {
        color: #a0a0a0;
        font-weight: 300;
        font-size: 0.9rem;
        line-height: 1.75;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #e0e0e0 !important;
        font-weight: 500 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0a0a0a !important;
        border-right: 1px solid #151515 !important;
    }

    /* Buttons */
    .stButton > button {
        background: #111 !important;
        border: 1px solid #1e1e1e !important;
        color: #888 !important;
        font-weight: 400 !important;
        font-size: 0.8rem !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
        letter-spacing: 0.5px !important;
    }
    .stButton > button:hover {
        border-color: #3a3a3a !important;
        color: #ddd !important;
        background: #161616 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.5) !important;
    }

    /* Chat */
    .stChatMessage {
        background: rgba(255,255,255,0.015) !important;
        border: 1px solid #141414 !important;
        border-radius: 12px !important;
        transition: border-color 0.3s ease;
    }
    .stChatMessage:hover { border-color: #1e1e1e !important; }

    .stChatInput > div {
        border: 1px solid #1e1e1e !important;
        border-radius: 12px !important;
        background: #0e0e0e !important;
        transition: border-color 0.3s ease !important;
    }
    .stChatInput > div:focus-within { border-color: #3a3a3a !important; }
    .stChatInput textarea { color: #ddd !important; font-size: 0.9rem !important; }
    .stChatInput textarea::placeholder { color: #555 !important; }

    /* File uploader */
    section[data-testid="stFileUploader"] {
        border: 1px dashed #1a1a1a !important;
        border-radius: 10px;
        transition: border-color 0.3s ease;
    }
    section[data-testid="stFileUploader"]:hover { border-color: #333 !important; }

    /* Expanders */
    div[data-testid="stExpander"] {
        background: rgba(255,255,255,0.015) !important;
        border: 1px solid #141414 !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }
    div[data-testid="stExpander"] summary {
        color: #e0e0e0 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stExpander"] summary:hover {
        color: #fff !important;
        background: transparent !important;
    }
    div[data-testid="stExpander"] svg {
        fill: #888 !important;
    }
    /* Scrollbar */
    ::-webkit-scrollbar { width: 3px; }
    ::-webkit-scrollbar-track { background: #080808; }
    ::-webkit-scrollbar-thumb { background: #1a1a1a; border-radius: 3px; }

    /* Selection */
    ::selection { background: #2a2a2a; color: white; }

    /* ─── Custom Components ─── */
    .app-header {
        text-align: center;
        padding: 2rem 0 0.5rem;
    }
    .app-logo {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.6rem;
        font-weight: 600;
        letter-spacing: 6px;
        text-transform: uppercase;
        color: #3a3a3a;
        margin-bottom: 0.8rem;
    }
    .app-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #e8e8e8;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .app-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        color: #404040;
        font-weight: 300;
        margin-top: 0.5rem;
        letter-spacing: 0.3px;
    }
    .thin-rule {
        height: 1px;
        background: linear-gradient(90deg, transparent, #1a1a1a, transparent);
        border: none;
        margin: 1.5rem 0;
    }
    .privacy-banner {
        text-align: center;
        padding: 6px 12px;
        background: rgba(34, 197, 94, 0.06);
        border: 1px solid rgba(34, 197, 94, 0.12);
        border-radius: 8px;
        margin: 0.5rem auto 1rem;
        max-width: 320px;
    }
    .privacy-text {
        font-family: 'Inter', sans-serif;
        font-size: 0.68rem;
        color: #2d8a56;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        font-weight: 500;
    }
    .stat-row {
        display: flex;
        justify-content: center;
        gap: 2rem;
        padding: 0.8rem 0;
    }
    .stat-item { text-align: center; }
    .stat-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.6rem;
        color: #3a3a3a;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }
    .stat-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.85rem;
        color: #777;
        margin-top: 2px;
    }
    .section-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.6rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #2a2a2a;
        margin-bottom: 0.5rem;
    }
    .status-dot {
        display: inline-block;
        width: 5px; height: 5px;
        border-radius: 50%;
        margin-right: 5px;
        position: relative; top: -1px;
    }
    .dot-on { background: #2d8a56; box-shadow: 0 0 6px rgba(45,138,86,0.3); }
    .dot-off { background: #222; }
    .pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.65rem;
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.3px;
        margin: 1px 0;
    }
    .pill-on {
        background: rgba(255,255,255,0.03);
        border: 1px solid #1e1e1e;
        color: #666;
    }
    .pill-off {
        background: transparent;
        border: 1px solid #141414;
        color: #2a2a2a;
    }
    .pipeline-box {
        background: rgba(255,255,255,0.02);
        border: 1px solid #141414;
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 8px;
        font-size: 0.75rem;
        color: #555;
        font-family: 'Inter', sans-serif;
    }
    .pipeline-step {
        display: inline-block;
        padding: 2px 6px;
        background: rgba(255,255,255,0.03);
        border-radius: 4px;
        font-size: 0.65rem;
        color: #555;
        margin: 1px 2px;
    }
    .source-tag {
        display: inline-block;
        padding: 2px 8px;
        background: rgba(45,138,86,0.08);
        border: 1px solid rgba(45,138,86,0.15);
        border-radius: 6px;
        font-size: 0.7rem;
        color: #2d8a56;
        margin: 2px 3px;
        font-family: 'Inter', sans-serif;
    }

    /* Thinking indicator */
    .thinking-indicator {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 0;
    }
    .thinking-dots { display: flex; gap: 4px; }
    .thinking-dots span {
        width: 5px; height: 5px;
        border-radius: 50%;
        background: #444;
        animation: pulse-dot 1.4s ease-in-out infinite;
    }
    .thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
    .thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes pulse-dot {
        0%, 80%, 100% { opacity: 0.2; transform: scale(0.8); }
        40% { opacity: 1; transform: scale(1.2); }
    }
    .thinking-text {
        font-family: 'Inter', sans-serif;
        font-size: 0.78rem;
        color: #444;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

THINKING_HTML = """
<div class="thinking-indicator">
    <div class="thinking-dots"><span></span><span></span><span></span></div>
    <span class="thinking-text">Processing through pipeline...</span>
</div>
"""

# ─── Sidebar ───
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <span style="font-family:'Space Grotesk',sans-serif; font-size:1.1rem; font-weight:700;
                     color:#ccc; letter-spacing: 2px;">⬡ QWEN RAG</span>
        <br>
        <span style="font-family:'Inter',sans-serif; font-size:0.5rem; color:#333;
                     letter-spacing:4px; text-transform:uppercase;">Advanced Local Pipeline</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="thin-rule"></div>', unsafe_allow_html=True)

    # ─── Models Status ───
    st.markdown('<p class="section-label">Models</p>', unsafe_allow_html=True)
    
    models_to_check = ["qwen3:8b", "qwen3-embedding:0.6b"]
    all_ready = True
    for model in models_to_check:
        if check_model(model):
            st.markdown(f'<span class="pill pill-on"><span class="status-dot dot-on"></span>{model}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="pill pill-off"><span class="status-dot dot-off"></span>{model}</span>', unsafe_allow_html=True)
            all_ready = False
    
    if not all_ready:
        st.warning("⚠️ Pull missing models with `ollama pull <model>`")

    st.markdown('<div class="thin-rule"></div>', unsafe_allow_html=True)

    # ─── Settings ───
    st.markdown('<p class="section-label">Settings</p>', unsafe_allow_html=True)
    
    # Lock model to Qwen3:8b
    st.session_state.model = "qwen3:8b"
    
    # Context Size
    ctx_options = [4096, 8192, 16384, 32768, 65536, 131072]
    ctx_labels = ["4K", "8K (Default)", "16K", "32K", "64K", "128K"]
    selected_ctx_idx = st.selectbox(
        "Context Window Size",
        options=range(len(ctx_options)),
        format_func=lambda x: ctx_labels[x],
        index=1, # Default to 8K to save memory
        help="Lower values are faster but can't read as much context at once."
    )
    st.session_state.num_ctx = ctx_options[selected_ctx_idx]
    
    # HyDE Toggle
    if "use_hyde" not in st.session_state:
        st.session_state.use_hyde = True
    use_hyde = st.toggle("Enable HyDE (Slower)", value=st.session_state.use_hyde, help="Doubles response time but improves answer quality.")
    st.session_state.use_hyde = use_hyde
    st.markdown('<div class="thin-rule"></div>', unsafe_allow_html=True)

    # ─── Document Upload ───
    st.markdown('<p class="section-label">Documents</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "txt", "docx", "pptx", "xlsx", "xls", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        clear_btn = st.button("Clear All", use_container_width=True)
    with col2:
        count = len(st.session_state.get("current_files", []))
        chunk_count = st.session_state.get("chunk_count", 0)
        st.markdown(f"""
        <div style="text-align:center; padding:4px 0;">
            <span style="font-family:'Space Grotesk',sans-serif; font-size:1rem; color:#666;">{count}</span>
            <span style="font-family:'Inter',sans-serif; font-size:0.55rem; color:#333; letter-spacing:1px;"> files</span>
        </div>
        """, unsafe_allow_html=True)

    if clear_btn:
        clear_store()
        clear_memory()
        if "current_files" in st.session_state:
            del st.session_state.current_files
        if "chunk_count" in st.session_state:
            del st.session_state.chunk_count
        if "messages" in st.session_state:
            del st.session_state.messages
        st.success("Cleared.")
        st.rerun()

    st.markdown('<div class="thin-rule"></div>', unsafe_allow_html=True)

    # ─── Pipeline Info ───
    st.markdown('<p class="section-label">Pipeline</p>', unsafe_allow_html=True)
    pipeline_stages = ["HyDE", "Dense MMR", "BM25", "RRF", "FlashRank", "Reorder", "Generate"]
    if not st.session_state.get("use_hyde", True):
        pipeline_stages.remove("HyDE")
    stage_html = " → ".join([f'<span class="pipeline-step">{s}</span>' for s in pipeline_stages])
    st.markdown(f'<div style="line-height:2;">{stage_html}</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="thin-rule"></div>', unsafe_allow_html=True)
    
    # ─── Stats ───
    st.markdown('<p class="section-label">Stats</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.72rem; color:#444; font-family:'Inter',sans-serif; line-height:2;">
        Chunks: <span style="color:#666;">{st.session_state.get('chunk_count', 0)}</span><br>
        Messages: <span style="color:#666;">{len(st.session_state.get('messages', []))}</span><br>
        Memory: <span style="color:#666;">{'Active' if memory_exists() else 'Empty'}</span>
    </div>
    """, unsafe_allow_html=True)


# ─── Main Header ───
st.markdown("""
<div class="app-header">
    <div class="app-logo">⬡ Local AI</div>
    <h1 class="app-title">Qwen RAG</h1>
    <p class="app-subtitle">Advanced retrieval-augmented generation · Fully local · Zero cloud</p>
</div>
<div class="privacy-banner">
    <span class="privacy-text">🔒 100% Local — No data leaves your machine</span>
</div>
""", unsafe_allow_html=True)

# ─── Status Row ───
doc_count = len(st.session_state.get("current_files", []))
chunk_count = st.session_state.get("chunk_count", 0)
msg_count = len(st.session_state.get("messages", []))
mode_label = "RAG" if doc_count > 0 else "Chat"

st.markdown(f"""
<div class="stat-row">
    <div class="stat-item">
        <div class="stat-label">Mode</div>
        <div class="stat-value">{mode_label}</div>
    </div>
    <div class="stat-item">
        <div class="stat-label">Documents</div>
        <div class="stat-value">{doc_count}</div>
    </div>
    <div class="stat-item">
        <div class="stat-label">Chunks</div>
        <div class="stat-value">{chunk_count}</div>
    </div>
    <div class="stat-item">
        <div class="stat-label">Messages</div>
        <div class="stat-value">{msg_count}</div>
    </div>
    <div class="stat-item">
        <div class="stat-label">Status</div>
        <div class="stat-value"><span class="status-dot {"dot-on" if doc_count > 0 else "dot-off"}"></span>{"Active" if doc_count > 0 else "Idle"}</div>
    </div>
</div>
<div class="thin-rule"></div>
""", unsafe_allow_html=True)


# ─── File Processing ───
if uploaded_files:
    file_names = [f.name for f in uploaded_files]
    if "current_files" not in st.session_state or st.session_state.current_files != file_names:
        with st.spinner(f"Processing {len(file_names)} file(s)..."):
            log.info(f"Processing batch of {len(file_names)} files")
            
            file_pairs = []
            temp_paths = []
            
            for uf in uploaded_files:
                suffix = f".{uf.name.split('.')[-1]}"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uf.getvalue())
                    temp_paths.append(tmp.name)
                    file_pairs.append((tmp.name, uf.name))
            
            try:
                all_docs = process_batch(file_pairs)
                
                if all_docs:
                    _, num_chunks = build_vector_store(all_docs)
                    st.session_state.current_files = file_names
                    st.session_state.chunk_count = num_chunks
                    log.info(f"Batch complete: {len(file_names)} files, {num_chunks} chunks")
                    st.success(f"✓ {len(file_names)} file(s) → {num_chunks} chunks embedded")
                    st.rerun()
                else:
                    st.error("No text could be extracted from the uploaded files.")
            except Exception as e:
                log.error(f"File processing error: {e}", exc_info=True)
                st.error(f"Error processing files: {e}")
            finally:
                for tp in temp_paths:
                    try:
                        os.remove(tp)
                    except Exception:
                        pass


# ─── Chat ───
st.markdown('<p class="section-label">Conversation</p>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            sources_html = " ".join([
                f'<span class="source-tag">{s["file"]}'
                + (f' p.{s["page"]}' if "page" in s else '')
                + (f' slide {s["slide"]}' if "slide" in s else '')
                + (f' {s["sheet"]}' if "sheet" in s else '')
                + '</span>'
                for s in message["sources"]
            ])
            # Deduplicate
            seen = set()
            unique_sources = []
            for s in message["sources"]:
                key = str(s)
                if key not in seen:
                    seen.add(key)
                    unique_sources.append(s)
            
            sources_html = " ".join([
                f'<span class="source-tag">{s["file"]}'
                + (f' p.{s["page"]}' if "page" in s else '')
                + (f' slide {s["slide"]}' if "slide" in s else '')
                + (f' {s["sheet"]}' if "sheet" in s else '')
                + '</span>'
                for s in unique_sources
            ])
            st.markdown(f'<div style="margin-top:6px;">{sources_html}</div>', unsafe_allow_html=True)

user_query = st.chat_input("Ask anything about your documents...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        st_placeholder = st.empty()
        st_placeholder.markdown(THINKING_HTML, unsafe_allow_html=True)

        try:
            log.info(f"User query: '{user_query[:100]}...'")
            result = ask(
                question=user_query,
                chat_history=st.session_state.messages[:-1],  # exclude current
                use_hyde=st.session_state.get("use_hyde", True),
                num_ctx=st.session_state.get("num_ctx", 8192)
            )

            answer = result["answer"]
            sources = result.get("sources", [])
            pipeline = result.get("pipeline", {})

            # Display answer
            st_placeholder.markdown(answer.replace("$", r"\$"))

            # Display sources
            if sources:
                seen = set()
                unique_sources = []
                for s in sources:
                    key = str(s)
                    if key not in seen:
                        seen.add(key)
                        unique_sources.append(s)
                
                sources_html = " ".join([
                    f'<span class="source-tag">{s["file"]}'
                    + (f' p.{s["page"]}' if "page" in s else '')
                    + (f' slide {s["slide"]}' if "slide" in s else '')
                    + (f' {s["sheet"]}' if "sheet" in s else '')
                    + '</span>'
                    for s in unique_sources
                ])
                st.markdown(f'<div style="margin-top:6px;">{sources_html}</div>', unsafe_allow_html=True)

            # Pipeline stats (expandable)
            if pipeline and pipeline.get("mode") == "rag":
                with st.expander("Pipeline details"):
                    st.markdown(f"""
                    - **Candidates found**: {pipeline.get('candidates_found', 'N/A')}
                    - **After reranking**: {pipeline.get('after_rerank', 'N/A')}
                    - **Context chunks**: {pipeline.get('context_chunks', 'N/A')}
                    - **Memory recalls**: {pipeline.get('memory_recalls', 'N/A')}
                    """)

            # Store in session
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

            # Store in persistent memory
            store_exchange(user_query, answer)
            log.info("Exchange stored in memory")

        except Exception as e:
            log.error(f"Query error: {e}", exc_info=True)
            st_placeholder.empty()
            st.error(f"Error: {e}")
