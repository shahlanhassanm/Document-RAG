"""
Advanced Local RAG System — Streamlit UI
Qwen3 + Milvus + Multi-Stage Retrieval
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

# Initialize state from persistent store if UI was refreshed
if store_exists() and "current_files" not in st.session_state:
    try:
        from core.vector_store import _load_bm25
        _, chunks = _load_bm25()
        sources = set(doc.metadata.get("source", "Unknown") for doc in chunks)
        st.session_state.current_files = list(sources)
        st.session_state.chunk_count = len(chunks)
        log.info(f"Loaded {len(sources)} files and {len(chunks)} chunks from persistent store into UI state.")
    except Exception as e:
        log.error(f"Failed to load persistent store into UI state: {e}")

# ─── Premium Glassmorphic CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Outfit:wght@300;400;500;700&display=swap');

    /* Hide Streamlit chrome */
    #MainMenu, footer, header,
    .stDeployButton,
    div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"] { display: none !important; }

    /* Animated Mesh Gradient Background */
    .stApp {
        background: linear-gradient(-45deg, #020617, #1e1b4b, #0f172a, #312e81) !important;
        background-size: 400% 400% !important;
        animation: gradientBG 15s ease infinite !important;
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Global Typography */
    .stApp, .stApp * {
        font-family: 'Inter', -apple-system, sans-serif !important;
    }
    .stApp span.material-symbols-rounded, 
    .stApp span.material-icons,
    .stApp span[data-testid="stIconMaterial"] {
        font-family: "Material Symbols Rounded", "Material Icons" !important;
    }
    .stMarkdown p, .stMarkdown li {
        color: #cbd5e1;
        font-weight: 300;
        font-size: 0.95rem;
        line-height: 1.75;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Outfit', sans-serif !important;
        color: #f8fafc !important;
        font-weight: 500 !important;
    }

    /* Sidebar (Frosted Glass) */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.4) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    /* Buttons (Glass & Glow) */
    .stButton > button {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #e2e8f0 !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        border-radius: 12px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }
    .stButton > button:hover {
        border-color: rgba(139, 92, 246, 0.5) !important;
        color: #fff !important;
        background: rgba(139, 92, 246, 0.15) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 25px -5px rgba(139, 92, 246, 0.4) !important;
    }

    /* Chat Messages (Floating Glass Bubbles) */
    .stChatMessage {
        padding: 1.2rem !important;
        background: transparent !important;
    }
    .stChatMessage[data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.02) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 20px !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2) !important;
    }
    div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] p {
        font-size: 0.98rem !important;
        line-height: 1.65 !important;
        color: #e2e8f0;
    }

    /* Chat Input */
    .stChatInput > div {
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        background: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(20px) !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
    }
    .stChatInput > div:focus-within { 
        border-color: #8b5cf6 !important; 
        box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.3), 0 8px 32px rgba(0,0,0,0.4) !important;
    }
    .stChatInput textarea { color: #f8fafc !important; font-size: 0.95rem !important; }
    .stChatInput textarea::placeholder { color: #64748b !important; }

    /* File uploader */
    section[data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.02) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px dashed rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    section[data-testid="stFileUploader"]:hover { 
        border-color: #8b5cf6 !important; 
        background: rgba(139, 92, 246, 0.05) !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background: transparent !important;
        border: none !important;
        border-radius: 4px !important;
        box-shadow: none !important;
        margin-top: 0px !important;
    }
    div[data-testid="stExpander"] summary {
        color: #94a3b8 !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.85rem !important;
        padding-left: 0 !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
    }
    div[data-testid="stExpander"] summary:hover {
        color: #c4b5fd !important;
        background: transparent !important;
    }
    div[data-testid="stExpander"] summary svg,
    div[data-testid="stExpander"] summary span.material-symbols-rounded,
    div[data-testid="stExpander"] summary [data-testid="stIconMaterial"] {
        display: none !important;
    }
    div[data-testid="stExpander"] summary::before {
        content: "▶" !important;
        font-size: 0.75rem !important;
        color: #8b5cf6 !important;
        transition: transform 0.2s ease !important;
    }
    div[data-testid="stExpander"][open] summary::before {
        transform: rotate(90deg) !important;
    }
    div[data-testid="stExpanderDetails"] {
        background: rgba(255, 255, 255, 0.02) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px !important;
        padding: 12px !important;
        box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.05) !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: rgba(0,0,0,0.1); }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 5px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(139, 92, 246, 0.5); }

    /* Selection */
    ::selection { background: rgba(139, 92, 246, 0.4); color: white; }

    /* ─── Custom Components ─── */
    .app-header {
        text-align: center;
        padding: 3rem 0 1rem;
    }
    .app-logo {
        font-family: 'Outfit', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 8px;
        text-transform: uppercase;
        background: linear-gradient(to right, #38bdf8, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.8rem;
    }
    .app-title {
        font-family: 'Outfit', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        color: #fff;
        margin: 0;
        letter-spacing: -1px;
        text-shadow: 0 4px 20px rgba(139, 92, 246, 0.3);
    }
    .app-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: #94a3b8;
        font-weight: 400;
        margin-top: 0.8rem;
        letter-spacing: 0.5px;
    }
    .thin-rule {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        border: none;
        margin: 2rem 0;
    }
    .privacy-banner {
        text-align: center;
        padding: 6px 16px;
        background: rgba(16, 185, 129, 0.1);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 20px;
        margin: 1rem auto;
        display: inline-block;
    }
    .privacy-text {
        font-family: 'Outfit', sans-serif;
        font-size: 0.7rem;
        color: #34d399;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        font-weight: 600;
    }
    .stat-row {
        display: flex;
        justify-content: center;
        gap: 3rem;
        padding: 1rem 0;
        background: rgba(255,255,255,0.01);
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.03);
    }
    .stat-item { text-align: center; }
    .stat-label {
        font-family: 'Outfit', sans-serif;
        font-size: 0.65rem;
        color: #64748b;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-weight: 500;
    }
    .stat-value {
        font-family: 'Outfit', sans-serif;
        font-size: 1rem;
        color: #e2e8f0;
        margin-top: 6px;
        font-weight: 600;
        text-shadow: 0 2px 10px rgba(255,255,255,0.1);
    }
    .section-label {
        font-family: 'Outfit', sans-serif;
        font-size: 0.65rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.8rem;
        font-weight: 600;
    }
    .status-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-right: 8px;
        position: relative; top: -1px;
    }
    .dot-on { background: #34d399; box-shadow: 0 0 12px rgba(52, 211, 153, 0.6); }
    .dot-off { background: #334155; }
    .pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-family: 'Outfit', sans-serif;
        font-weight: 500;
        letter-spacing: 0.5px;
        margin: 4px 0;
        backdrop-filter: blur(8px);
    }
    .pill-on {
        background: rgba(139, 92, 246, 0.1);
        border: 1px solid rgba(139, 92, 246, 0.3);
        color: #c4b5fd;
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.1);
    }
    .pill-off {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        color: #64748b;
    }
    .pipeline-box {
        background: rgba(255,255,255,0.02);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 12px 16px;
        margin-top: 10px;
        font-size: 0.8rem;
        color: #cbd5e1;
        font-family: 'Inter', sans-serif;
    }
    .pipeline-step {
        display: inline-block;
        padding: 3px 8px;
        background: rgba(255,255,255,0.05);
        border-radius: 6px;
        font-size: 0.7rem;
        color: #e2e8f0;
        margin: 3px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .pipeline-step:not(:last-child)::after {
        content: "→";
        margin-left: 10px;
        margin-right: -2px;
        color: #64748b;
        font-family: system-ui, -apple-system, sans-serif;
    }
    .source-tag {
        display: inline-block;
        padding: 3px 10px;
        background: rgba(56, 189, 248, 0.1);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 8px;
        font-size: 0.75rem;
        color: #7dd3fc;
        margin: 3px 4px;
        font-family: 'Outfit', sans-serif;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .source-tag:hover {
        background: rgba(56, 189, 248, 0.2);
        transform: translateY(-1px);
    }

    /* Thinking indicator */
    .thinking-indicator {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 0;
    }
    .thinking-dots { display: flex; gap: 6px; }
    .thinking-dots span {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: #a78bfa;
        animation: pulse-dot 1.4s ease-in-out infinite;
    }
    .thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
    .thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes pulse-dot {
        0%, 80%, 100% { opacity: 0.3; transform: scale(0.8); }
        40% { opacity: 1; transform: scale(1.3); box-shadow: 0 0 10px rgba(167, 139, 250, 0.6); }
    }
    .thinking-text {
        font-family: 'Outfit', sans-serif;
        font-size: 0.85rem;
        color: #c4b5fd;
        letter-spacing: 0.5px;
        font-weight: 500;
        animation: pulse-text 2s ease-in-out infinite;
    }
    @keyframes pulse-text {
        0%, 100% { opacity: 0.7; }
        50% { opacity: 1; }
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
                     color:#f8fafc; letter-spacing: 2px;">⬡ QWEN RAG</span>
        <br>
        <span style="font-family:'Inter',sans-serif; font-size:0.55rem; color:#0ea5e9;
                     letter-spacing:4px; text-transform:uppercase;">Advanced Local Pipeline</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="thin-rule"></div>', unsafe_allow_html=True)

    # ─── Models Status ───
    st.markdown('<p class="section-label">Models</p>', unsafe_allow_html=True)
    
    models_to_check = ["qwen3:8b", "qwen3:1.7b", "qwen3-embedding:0.6b", "nomic-embed-text"]
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

    # Pipeline Mode selector
    def on_pipeline_mode_change():
        # User swapped modes, which means embedding model swapped, so we must clear the DB.
        old_mode = st.session_state.get("pipeline_mode", "accurate")
        new_mode = st.session_state.candidate_mode
        if old_mode != new_mode:
            clear_store()
            if "chunk_count" in st.session_state:
                st.session_state.chunk_count = 0
            if "current_files" in st.session_state:
                st.session_state.current_files = []
            if "processed_files" in st.session_state:
                st.session_state.processed_files = set()
            st.toast("Database cleared due to embedding model change.", icon="🧹")

    mode_options = {
        "accurate": "🎯 Accurate (qwen3:8b + qwen3-embed, ~4m)",
        "fast": "🚀 Fast (qwen3:1.7b + nomic-embed, ~30s)"
    }
    
    selected_mode = st.selectbox(
        "Pipeline Mode",
        options=list(mode_options.keys()),
        format_func=lambda x: mode_options[x],
        index=0 if st.session_state.get("pipeline_mode", "accurate") == "accurate" else 1,
        key="candidate_mode",
        on_change=on_pipeline_mode_change,
        help="Fast uses qwen3:1.7b & nomic-embed. Accurate uses qwen3:8b & qwen3-embedding."
    )
    st.session_state.pipeline_mode = selected_mode
    
    # Map mode to actual models
    if selected_mode == "fast":
        st.session_state.model = "qwen3:1.7b"
        st.session_state.embedding_model = "nomic-embed-text"
    else:
        st.session_state.model = "qwen3:8b"
        st.session_state.embedding_model = "qwen3-embedding:0.6b"
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="thin-rule"></div>', unsafe_allow_html=True)

    # ─── Document Upload ───
    st.markdown('<p class="section-label">Documents</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "txt", "md", "docx", "pptx", "xlsx", "xls", "csv"],
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
            <span style="font-family:'Space Grotesk',sans-serif; font-size:1rem; color:#f8fafc;">{count}</span>
            <span style="font-family:'Inter',sans-serif; font-size:0.55rem; color:#0ea5e9; letter-spacing:1px;"> files</span>
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
    stage_html = "".join([f'<span class="pipeline-step">{s}</span>' for s in pipeline_stages])
    st.markdown(f'<div style="line-height:2;">{stage_html}</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="thin-rule"></div>', unsafe_allow_html=True)
    
    # ─── Stats ───
    st.markdown('<p class="section-label">Stats</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.72rem; color:#737373; font-family:'Inter',sans-serif; line-height:2;">
        Chunks: <span style="color:#a3a3a3;">{st.session_state.get('chunk_count', 0)}</span><br>
        Messages: <span style="color:#a3a3a3;">{len(st.session_state.get('messages', []))}</span><br>
        Memory: <span style="color:#a3a3a3;">{'Active' if memory_exists() else 'Empty'}</span>
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
                    _, num_chunks = build_vector_store(all_docs, st.session_state.get("embedding_model", "qwen3-embedding:0.6b"))
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
    # Use emojis instead of material icon names to prevent broken text rendering
    avatar = "🧑‍💻" if message["role"] == "user" else "🧠"
    with st.chat_message(message["role"], avatar=avatar):
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

    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar="🧠"):
        st_placeholder = st.empty()
        st_placeholder.markdown(THINKING_HTML, unsafe_allow_html=True)

        try:
            log.info(f"User query: '{user_query[:100]}...'")
            
            # Start generator
            result_gen = ask(
                question=user_query,
                chat_history=st.session_state.messages[:-1],  # exclude current
                num_ctx=st.session_state.get("num_ctx", 8192),
                model_name=st.session_state.get("model", "qwen3:8b"),
                embedding_model_name=st.session_state.get("embedding_model", "qwen3-embedding:0.6b")
            )
            
            answer = ""
            sources = []
            pipeline = {}
            
            # Process stream
            for chunk in result_gen:
                if isinstance(chunk, dict) and chunk.get("is_meta"):
                    # Final metadata block
                    answer = chunk["full_answer"]
                    sources = chunk.get("sources", [])
                    pipeline = chunk.get("pipeline", {})
                elif isinstance(chunk, str):
                    # Text token streaming
                    answer += chunk
                    # Real-time UI update
                    st_placeholder.markdown(answer.replace("$", r"\$") + "▌")
                elif isinstance(chunk, dict) and "answer" in chunk:
                    # Fallback for the non-streaming error messages (e.g. no documents)
                    answer = chunk["answer"]
                    sources = chunk.get("sources", [])
                    pipeline = chunk.get("pipeline", {})
                    st_placeholder.markdown(answer.replace("$", r"\$"))

            # Final clean display without cursor
            st_placeholder.markdown(answer.replace("$", r"\$"))

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
