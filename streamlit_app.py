# streamlit_app.py

import streamlit as st
import requests
import time

# ── Configuration ──────────────────────────────────────────────────────────────
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="QAD Support Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f0f2f6; }

    /* Chat message bubbles */
    .user-message {
        background-color: #0066cc;
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        word-wrap: break-word;
    }
    .assistant-message {
        background-color: white;
        color: #1a1a1a;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 80%;
        border: 1px solid #e0e0e0;
        word-wrap: break-word;
    }
    .confidence-badge {
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 10px;
        margin-top: 4px;
        display: inline-block;
    }
    .high   { background-color: #e6ffe6; color: #006600; }
    .medium { background-color: #fff3e6; color: #cc6600; }
    .low    { background-color: #ffe6e6; color: #cc0000; }

    /* Upload status boxes */
    .success-box {
        background-color: #e6ffe6;
        border: 1px solid #00cc00;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #ffe6e6;
        border: 1px solid #cc0000;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .duplicate-box {
        background-color: #fff3e6;
        border: 1px solid #ff9900;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }

    /* Source citations */
    .source-item {
        font-size: 12px;
        color: #666;
        padding: 3px 0;
    }

    /* Sidebar styling */
    .sidebar-header {
        font-size: 18px;
        font-weight: bold;
        color: #0066cc;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state initialization ───────────────────────────────────────────────
def initialize_session():
    """
    Auto-create a session ID on first load.
    This mimics how ChatGPT handles sessions —
    user never sees or manages the session ID.
    """
    if "session_id" not in st.session_state:
        try:
            response = requests.post(f"{API_BASE_URL}/session", timeout=5)
            if response.status_code == 200:
                st.session_state.session_id = response.json()["session_id"]
            else:
                st.session_state.session_id = None
        except Exception:
            st.session_state.session_id = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "pending_escalation" not in st.session_state:
        st.session_state.pending_escalation = False

    if "upload_results" not in st.session_state:
        st.session_state.upload_results = []


# ── API helper functions ───────────────────────────────────────────────────────
def send_chat_message(query: str) -> dict:
    """Send a chat message to the API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={
                "query":      query,
                "session_id": st.session_state.session_id,
            },
            timeout=120,  # LLM can take up to 2 minutes on CPU
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "answer":          f"API error: {response.status_code}",
                "confidence":      0.0,
                "confidence_band": "low",
                "sources":         [],
                "should_escalate": False,
                "status":          "error",
            }
    except requests.exceptions.Timeout:
        return {
            "answer":          "Request timed out. The LLM is taking too long. Please try again.",
            "confidence":      0.0,
            "confidence_band": "low",
            "sources":         [],
            "should_escalate": False,
            "status":          "error",
        }
    except Exception as e:
        return {
            "answer":          f"Connection error: {str(e)}",
            "confidence":      0.0,
            "confidence_band": "low",
            "sources":         [],
            "should_escalate": False,
            "status":          "error",
        }


def upload_file(file) -> dict:
    """Upload a file to the API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/upload",
            files={"file": (file.name, file.getvalue(), "application/octet-stream")},
            timeout=300,  # Large PDFs can take up to 5 minutes
        )
        return response.json()
    except requests.exceptions.Timeout:
        return {
            "status":  "error",
            "message": "Upload timed out. The file may be too large or ingestion is taking too long.",
        }
    except Exception as e:
        return {
            "status":  "error",
            "message": f"Upload failed: {str(e)}",
        }


def get_collection_stats() -> list:
    """Fetch ChromaDB collection stats."""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()["collections"]
    except Exception:
        pass
    return []


def get_documents() -> list:
    """Fetch all ingested documents."""
    try:
        response = requests.get(f"{API_BASE_URL}/documents", timeout=5)
        if response.status_code == 200:
            return response.json()["documents"]
    except Exception:
        pass
    return []


def new_conversation():
    """Start a fresh conversation session."""
    try:
        response = requests.post(f"{API_BASE_URL}/session", timeout=5)
        if response.status_code == 200:
            st.session_state.session_id        = response.json()["session_id"]
            st.session_state.messages          = []
            st.session_state.pending_escalation = False
    except Exception:
        pass


# ── Page renderers ─────────────────────────────────────────────────────────────
def render_chat_page():
    """Render the main chat interface."""
    st.title("🤖 QAD Support Assistant")
    st.caption(
        f"Session: `{st.session_state.session_id[:8]}...`"
        if st.session_state.session_id
        else "⚠️ No active session — is the API running?"
    )

    # New conversation button
    if st.button("🔄 New Conversation", use_container_width=False):
        new_conversation()
        st.rerun()

    st.divider()

    # ── Chat history ───────────────────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
            <div style='text-align:center; color:#888; padding:40px;'>
                👋 Hello! I'm your QAD Support Assistant.<br><br>
                Ask me about QAD errors, processes, or system issues.<br>
                I'll search the knowledge base and provide answers with sources.
            </div>
            """, unsafe_allow_html=True)

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f"<div class='user-message'>👤 {msg['content']}</div>",
                    unsafe_allow_html=True,
                )
            else:
                # Assistant message with confidence badge
                confidence_band = msg.get("confidence_band", "low")
                confidence_pct  = int(msg.get("confidence", 0) * 100)

                st.markdown(
                    f"<div class='assistant-message'>🤖 {msg['content']}</div>",
                    unsafe_allow_html=True,
                )

                # Confidence badge
                st.markdown(
                    f"<span class='confidence-badge {confidence_band}'>"
                    f"Confidence: {confidence_pct}% — {confidence_band.upper()}"
                    f"</span>",
                    unsafe_allow_html=True,
                )

                # Sources (collapsed by default)
                if msg.get("sources"):
                    with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                        for source in msg["sources"]:
                            st.markdown(
                                f"<div class='source-item'>{source}</div>",
                                unsafe_allow_html=True,
                            )

    # ── Chat input ─────────────────────────────────────────────────────────────
    st.divider()

    # Show escalation notice if pending
    if st.session_state.pending_escalation:
        st.warning(
            "⚠️ Waiting for your response. Reply **yes** to create a "
            "ServiceNow incident or **no** to cancel."
        )

    user_input = st.chat_input(
        placeholder="Ask about a QAD error, process, or type 'escalate' to raise a ticket..."
    )

    if user_input:
        # Add user message to history
        st.session_state.messages.append({
            "role":    "user",
            "content": user_input,
        })

        # Send to API with spinner
        with st.spinner("Searching knowledge base..."):
            result = send_chat_message(user_input)

        # Update session ID if new session was created
        if result.get("session_id"):
            st.session_state.session_id = result["session_id"]

        # Track escalation state
        st.session_state.pending_escalation = result.get("should_escalate", False)

        # Add assistant response to history
        st.session_state.messages.append({
            "role":            "assistant",
            "content":         result.get("answer", "No response received."),
            "confidence":      result.get("confidence", 0.0),
            "confidence_band": result.get("confidence_band", "low"),
            "sources":         result.get("sources", []),
            "status":          result.get("status", "unknown"),
        })

        st.rerun()


def render_upload_page():
    """Render the document upload interface."""
    st.title("📄 Document Upload")
    st.markdown("Upload PDF or Excel files to add them to the knowledge base.")

    # ── Upload form ────────────────────────────────────────────────────────────
    st.subheader("Upload New Document")

    uploaded_file = st.file_uploader(
        label="Choose a file",
        type=["pdf", "xlsx", "xls"],
        help="Supported formats: PDF, Excel (.xlsx, .xls). Maximum size: 100MB.",
    )

    if uploaded_file:
        # Show file info
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.info(
            f"📎 **{uploaded_file.name}** "
            f"({file_size_mb:.1f} MB) — "
            f"ready to upload"
        )

        if st.button("⬆️ Upload and Ingest", type="primary", use_container_width=True):
            with st.spinner(
                f"Uploading and ingesting **{uploaded_file.name}**... "
                f"This may take several minutes for large PDFs."
            ):
                start_time = time.time()
                result     = upload_file(uploaded_file)
                elapsed    = time.time() - start_time

            # Store result in session state
            st.session_state.upload_results.insert(0, {
                "filename": uploaded_file.name,
                "result":   result,
                "elapsed":  elapsed,
            })

            # Show result
            if result.get("status") == "success":
                st.markdown(f"""
                <div class='success-box'>
                    <strong>✅ Successfully Ingested</strong><br>
                    File: <code>{result.get('filename')}</code><br>
                    Doc ID: <code>{result.get('doc_id', 'N/A')}</code><br>
                    Chunks created: <strong>{result.get('chunk_count', 0)}</strong><br>
                    Time taken: {elapsed:.1f} seconds
                </div>
                """, unsafe_allow_html=True)

            elif result.get("status") == "duplicate":
                st.markdown(f"""
                <div class='duplicate-box'>
                    <strong>⚠️ Duplicate File Detected</strong><br>
                    {result.get('message', '')}
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div class='error-box'>
                    <strong>❌ Upload Failed</strong><br>
                    {result.get('message', result.get('detail', 'Unknown error'))}
                </div>
                """, unsafe_allow_html=True)

            st.rerun()

    # ── Upload history ─────────────────────────────────────────────────────────
    if st.session_state.upload_results:
        st.divider()
        st.subheader("Upload History (this session)")

        for item in st.session_state.upload_results:
            r      = item["result"]
            status = r.get("status", "error")
            icon   = {"success": "✅", "duplicate": "⚠️", "error": "❌"}.get(status, "❓")

            with st.expander(f"{icon} {item['filename']} — {status.upper()}"):
                st.json(r)

    # ── Ingested documents ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("All Ingested Documents")

    if st.button("🔄 Refresh", use_container_width=False):
        st.rerun()

    docs = get_documents()
    if not docs:
        st.info("No documents ingested yet.")
    else:
        for doc in docs:
            status_icon = {"ready": "✅", "processing": "⏳", "failed": "❌"}.get(
                doc.get("status"), "❓"
            )
            with st.expander(
                f"{status_icon} {doc['filename']} "
                f"| {doc['source_type'].upper()} "
                f"| v{doc['version']} "
                f"| {doc['chunk_count']} chunks"
            ):
                st.write(f"**Doc ID:** `{doc['doc_id']}`")
                st.write(f"**Status:** {doc['status']}")
                st.write(f"**Ingested at:** {doc['ingested_at']}")
                if doc.get("notes"):
                    st.write(f"**Notes:** {doc['notes']}")


def render_stats_page():
    """Render system stats."""
    st.title("📊 System Stats")

    if st.button("🔄 Refresh Stats"):
        st.rerun()

    # Collection stats
    st.subheader("ChromaDB Collections")
    stats = get_collection_stats()

    if stats:
        col1, col2, col3 = st.columns(3)
        icons = {"pdf_chunks": "📄", "excel_errors": "📊", "servicenow_tickets": "🎫"}

        for i, stat in enumerate(stats):
            col = [col1, col2, col3][i % 3]
            with col:
                st.metric(
                    label=f"{icons.get(stat['collection'], '📁')} {stat['collection']}",
                    value=f"{stat['count']} chunks",
                )
    else:
        st.warning("Could not fetch stats — is the API running?")

    # API health
    st.divider()
    st.subheader("API Health")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            st.success("✅ API is healthy and running.")
        else:
            st.error(f"❌ API returned status {response.status_code}")
    except Exception as e:
        st.error(f"❌ Cannot connect to API: {e}")


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("<div class='sidebar-header'>🤖 QAD Support</div>",
                    unsafe_allow_html=True)
        st.divider()

        page = st.radio(
            "Navigation",
            options=["💬 Chat", "📄 Upload Documents", "📊 Stats"],
            index=0,
        )

        st.divider()

        # Session info
        if st.session_state.get("session_id"):
            st.caption(f"Session: `{st.session_state.session_id[:8]}...`")
        else:
            st.warning("No active session")

        # Quick stats in sidebar
        stats = get_collection_stats()
        if stats:
            st.divider()
            st.caption("Knowledge Base")
            for stat in stats:
                icons = {
                    "pdf_chunks":          "📄",
                    "excel_errors":        "📊",
                    "servicenow_tickets":  "🎫",
                }
                icon  = icons.get(stat["collection"], "📁")
                label = stat["collection"].replace("_", " ").title()
                st.caption(f"{icon} {label}: **{stat['count']}**")

        return page


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    initialize_session()
    page = render_sidebar()

    if page == "💬 Chat":
        render_chat_page()
    elif page == "📄 Upload Documents":
        render_upload_page()
    elif page == "📊 Stats":
        render_stats_page()


if __name__ == "__main__":
    main()