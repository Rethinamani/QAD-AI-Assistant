# streamlit_app.py

import streamlit as st
import requests

# ── Configuration ──────────────────────────────────────────────────────────────
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Infor Support Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── ChatGPT-style styling ──────────────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu, header, footer {visibility: hidden;}

    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                     Helvetica, Arial, sans-serif;
    }

    /* Center the conversation column, ChatGPT width */
    .block-container {
        max-width: 46rem;
        padding-top: 2.5rem;
        padding-bottom: 6rem;
    }

    /* Chat bubbles → flat full-width rows like ChatGPT */
    [data-testid="stChatMessage"] {
        background: transparent;
        padding: 0.35rem 0;
    }

    /* Sidebar: quiet grey, list-like */
    section[data-testid="stSidebar"] {
        background-color: #f7f7f8;
        border-right: 1px solid #e5e5e5;
    }
    section[data-testid="stSidebar"] .stButton > button {
        text-align: left;
        justify-content: flex-start;
        font-weight: 400;
        border: none;
        padding: 0.4rem 0.6rem;
        border-radius: 8px;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #ececed;
    }

    /* "New chat" button — outlined, full width */
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]
        > div:first-child .stButton > button {
        border: 1px solid #d0d0d0;
        font-weight: 500;
    }

    .assistant-meta {
        font-size: 11px;
        color: #8e8ea0;
        margin: 2px 0 0 2px;
    }
    .conf-high   { color: #2e7d32; }
    .conf-medium { color: #ef6c00; }
    .conf-low    { color: #c62828; }

    .empty-state {
        text-align: center;
        color: #8e8ea0;
        padding: 18vh 1rem 0;
        font-size: 15px;
        line-height: 1.7;
    }
    .empty-state .robot-logo {
        font-size: 56px;
        line-height: 1;
        display: block;
        margin: 0 auto 0.6rem;
    }
    .empty-state .app-name {
        display: block;
        font-size: 22px;
        font-weight: 700;
        color: #2d2d2d;
        margin-bottom: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)


# ── API helpers ────────────────────────────────────────────────────────────────
def api_list_conversations() -> list:
    try:
        r = requests.get(f"{API_BASE_URL}/conversations", timeout=5)
        if r.status_code == 200:
            return r.json()["conversations"]
    except Exception:
        pass
    return []


def api_load_conversation(conversation_id: str) -> dict | None:
    try:
        r = requests.get(
            f"{API_BASE_URL}/conversations/{conversation_id}", timeout=5
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def api_delete_conversation(conversation_id: str) -> None:
    try:
        requests.delete(
            f"{API_BASE_URL}/conversations/{conversation_id}", timeout=5
        )
    except Exception:
        pass


def api_rename_conversation(conversation_id: str, title: str) -> None:
    try:
        requests.patch(
            f"{API_BASE_URL}/conversations/{conversation_id}",
            json={"title": title},
            timeout=5,
        )
    except Exception:
        pass


def api_send_message(query: str, conversation_id: str | None) -> dict:
    try:
        r = requests.post(
            f"{API_BASE_URL}/chat",
            json={"query": query, "session_id": conversation_id},
            timeout=180,
        )
        if r.status_code == 200:
            return r.json()
        return {
            "answer": f"API error: {r.status_code}",
            "confidence": 0.0, "confidence_band": "low",
            "sources": [], "images": [], "should_escalate": False,
            "status": "error",
        }
    except requests.exceptions.Timeout:
        return {
            "answer": "Request timed out — the model is taking too long. Please try again.",
            "confidence": 0.0, "confidence_band": "low",
            "sources": [], "images": [], "should_escalate": False,
            "status": "error",
        }
    except Exception as e:
        return {
            "answer": f"Connection error: {e}",
            "confidence": 0.0, "confidence_band": "low",
            "sources": [], "images": [], "should_escalate": False,
            "status": "error",
        }


# ── Session state ──────────────────────────────────────────────────────────────
def init_state():
    st.session_state.setdefault("conversation_id", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending_escalation", False)
    st.session_state.setdefault("conversations", api_list_conversations())


def start_new_chat():
    st.session_state.conversation_id = None
    st.session_state.messages = []
    st.session_state.pending_escalation = False


def open_conversation(conversation_id: str):
    convo = api_load_conversation(conversation_id)
    if not convo:
        return
    st.session_state.conversation_id = conversation_id
    st.session_state.messages = convo["messages"]
    last = convo["messages"][-1] if convo["messages"] else None
    st.session_state.pending_escalation = bool(
        last and last["role"] == "assistant" and last.get("status") == "escalate"
    )


# ── Sidebar: conversation list ─────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        if st.button("➕  New chat", use_container_width=True, key="new_chat"):
            start_new_chat()
            st.rerun()

        st.caption("Chats")

        for convo in st.session_state.conversations:
            cid = convo["conversation_id"]
            is_active = cid == st.session_state.conversation_id
            row, menu = st.columns([0.82, 0.18])

            with row:
                if st.button(
                    convo["title"] or "New chat",
                    key=f"open_{cid}",
                    use_container_width=True,
                    type="primary" if is_active else "tertiary",
                ):
                    open_conversation(cid)
                    st.rerun()

            with menu:
                with st.popover("⋯", use_container_width=True):
                    new_title = st.text_input(
                        "Rename", value=convo["title"], key=f"rename_{cid}"
                    )
                    if st.button("Save", key=f"save_{cid}"):
                        api_rename_conversation(cid, new_title)
                        st.session_state.conversations = api_list_conversations()
                        st.rerun()
                    if st.button("🗑  Delete", key=f"del_{cid}"):
                        api_delete_conversation(cid)
                        if is_active:
                            start_new_chat()
                        st.session_state.conversations = api_list_conversations()
                        st.rerun()

        if not st.session_state.conversations:
            st.caption("_No conversations yet._")


# ── Main: chat transcript ──────────────────────────────────────────────────────
def render_message(msg: dict):
    role = msg["role"]
    with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🤖"):
        st.markdown(msg["content"])

        if role == "assistant":
            band = msg.get("confidence_band") or "low"
            pct = int((msg.get("confidence") or 0) * 100)
            if msg.get("status") not in ("escalate", "rejected", "error"):
                st.markdown(
                    f"<div class='assistant-meta conf-{band}'>"
                    f"Confidence {pct}% · {band.upper()}</div>",
                    unsafe_allow_html=True,
                )

            if msg.get("sources"):
                with st.expander(f"Sources ({len(msg['sources'])})"):
                    for s in msg["sources"]:
                        st.markdown(f"<div class='assistant-meta'>{s}</div>",
                                    unsafe_allow_html=True)

            if msg.get("images"):
                with st.expander(f"Figures ({len(msg['images'])})"):
                    for img in msg["images"]:
                        st.image(
                            f"{API_BASE_URL}{img['url']}",
                            caption=img.get("caption"),
                        )


def render_chat():
    if not st.session_state.messages:
        st.markdown(
            "<div class='empty-state'>"
            "<span class='robot-logo'>🤖</span>"
            "<span class='app-name'>Infor Support Assistant</span>"
            "Ask about an Infor error, a process, or a system issue.<br>"
            "Type <em>escalate</em> to raise a ServiceNow ticket."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        for msg in st.session_state.messages:
            render_message(msg)

    if st.session_state.pending_escalation:
        st.info("Reply **yes** to create a ServiceNow incident, or **no** to cancel.")

    prompt = st.chat_input("Message Infor Support Assistant…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    render_message({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking…"):
            result = api_send_message(prompt, st.session_state.conversation_id)

    st.session_state.conversation_id = result.get("session_id") \
        or st.session_state.conversation_id
    st.session_state.pending_escalation = result.get("should_escalate", False)
    st.session_state.messages.append({
        "role": "assistant",
        "content": result.get("answer", "No response received."),
        "confidence": result.get("confidence", 0.0),
        "confidence_band": result.get("confidence_band", "low"),
        "sources": result.get("sources", []),
        "images": result.get("images", []),
        "status": result.get("status", "unknown"),
    })
    st.session_state.conversations = api_list_conversations()
    st.rerun()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    init_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
