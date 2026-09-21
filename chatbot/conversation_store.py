# chatbot/conversation_store.py

"""
Server-side persistence for chat conversations.

ChatGPT-style history: every conversation is a row, every message is a row,
so the sidebar list and full transcripts survive API restarts and browser
refreshes. Single shared instance — no per-user scoping (a user_id column
can be added later without touching existing rows).
"""

import json
import uuid
import sqlite3
import logging
from datetime import datetime

from config import CONVERSATION_DB_PATH

logger = logging.getLogger(__name__)

DEFAULT_TITLE = "New chat"
_TITLE_MAX_CHARS = 60


def _now() -> str:
    return datetime.utcnow().isoformat()


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(CONVERSATION_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize_conversation_store() -> None:
    """Create the conversation tables if they don't exist."""
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                title           TEXT NOT NULL DEFAULT 'New chat',
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL
                    REFERENCES conversations(conversation_id) ON DELETE CASCADE,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                confidence      REAL,
                confidence_band TEXT,
                sources         TEXT,
                images          TEXT,
                status          TEXT,
                created_at      TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conversation
            ON messages(conversation_id, id)
        """)
        conn.commit()
    logger.info("Conversation store initialized.")


# ── Conversations ──────────────────────────────────────────────────────────────
def create_conversation(conversation_id: str | None = None) -> str:
    """Create a new conversation and return its id."""
    conversation_id = conversation_id or str(uuid.uuid4())
    ts = _now()
    with _get_connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO conversations "
            "(conversation_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conversation_id, DEFAULT_TITLE, ts, ts),
        )
        conn.commit()
    return conversation_id


def ensure_conversation(conversation_id: str) -> None:
    """Make sure a conversation row exists for this id (idempotent)."""
    create_conversation(conversation_id)


def list_conversations() -> list[dict]:
    """All conversations, most recently updated first, with message counts."""
    with _get_connection() as conn:
        rows = conn.execute("""
            SELECT c.conversation_id,
                   c.title,
                   c.created_at,
                   c.updated_at,
                   COUNT(m.id) AS message_count
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.conversation_id
            GROUP BY c.conversation_id
            ORDER BY c.updated_at DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_conversation(conversation_id: str) -> dict | None:
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
    return dict(row) if row else None


def rename_conversation(conversation_id: str, title: str) -> None:
    title = (title or "").strip()[:_TITLE_MAX_CHARS] or DEFAULT_TITLE
    with _get_connection() as conn:
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? "
            "WHERE conversation_id = ?",
            (title, _now(), conversation_id),
        )
        conn.commit()


def delete_conversation(conversation_id: str) -> None:
    with _get_connection() as conn:
        conn.execute(
            "DELETE FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        )
        conn.commit()


# ── Messages ───────────────────────────────────────────────────────────────────
def append_message(
    conversation_id: str,
    role: str,
    content: str,
    confidence: float | None = None,
    confidence_band: str | None = None,
    sources: list | None = None,
    images: list | None = None,
    status: str | None = None,
) -> None:
    """Append one message and bump the conversation's updated_at."""
    ensure_conversation(conversation_id)
    ts = _now()
    with _get_connection() as conn:
        conn.execute("""
            INSERT INTO messages
                (conversation_id, role, content, confidence, confidence_band,
                 sources, images, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            conversation_id,
            role,
            content,
            confidence,
            confidence_band,
            json.dumps(sources or []),
            json.dumps(images or []),
            status,
            ts,
        ))
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
            (ts, conversation_id),
        )
        conn.commit()


def get_messages(conversation_id: str) -> list[dict]:
    """Full transcript for a conversation, oldest first."""
    with _get_connection() as conn:
        rows = conn.execute("""
            SELECT role, content, confidence, confidence_band,
                   sources, images, status, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
        """, (conversation_id,)).fetchall()

    out = []
    for r in rows:
        d = dict(r)
        d["sources"] = json.loads(d.get("sources") or "[]")
        d["images"] = json.loads(d.get("images") or "[]")
        out.append(d)
    return out


def message_count(conversation_id: str) -> int:
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
    return row["n"] if row else 0


def recent_turns(conversation_id: str, limit: int) -> list[dict]:
    """
    Last `limit` messages as [{"role", "content"}], oldest first.
    Used to rehydrate the chain's in-memory window after a restart.
    """
    with _get_connection() as conn:
        rows = conn.execute("""
            SELECT role, content FROM messages
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (conversation_id, limit)).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


# ── Titles ─────────────────────────────────────────────────────────────────────
def auto_title_from(text: str) -> str:
    """Cheap title from the first user message — first words, capped."""
    text = " ".join((text or "").split())
    if not text:
        return DEFAULT_TITLE
    if len(text) <= _TITLE_MAX_CHARS:
        return text
    return text[:_TITLE_MAX_CHARS].rsplit(" ", 1)[0] + "…"
