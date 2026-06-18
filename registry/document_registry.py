# registry/document_registry.py
import sqlite3
import uuid
from datetime import datetime
from config import REGISTRY_DB_PATH


def _get_connection():
    conn = sqlite3.connect(REGISTRY_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_registry():
    """Create the registry table if it doesn't exist."""
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id        TEXT PRIMARY KEY,
                filename      TEXT NOT NULL,
                source_type   TEXT NOT NULL,
                version       INTEGER DEFAULT 1,
                status        TEXT DEFAULT 'processing',
                chunk_count   INTEGER DEFAULT 0,
                ingested_at   TEXT NOT NULL,
                notes         TEXT
            )
        """)
        conn.commit()
    print("✅ Document registry initialized.")


def register_document(filename: str, source_type: str, notes: str = "") -> str:
    """
    Register a new document. If filename already exists, creates a new version.
    Returns the new doc_id.
    """
    doc_id = str(uuid.uuid4())
    ingested_at = datetime.utcnow().isoformat()

    existing = get_by_filename(filename)
    version = (existing["version"] + 1) if existing else 1

    with _get_connection() as conn:
        conn.execute("""
            INSERT INTO documents
                (doc_id, filename, source_type, version, status, chunk_count, ingested_at, notes)
            VALUES
                (?, ?, ?, ?, 'processing', 0, ?, ?)
        """, (doc_id, filename, source_type, version, ingested_at, notes))
        conn.commit()

    print(f"📝 Registered: {filename} | version={version} | doc_id={doc_id}")
    return doc_id


def update_status(doc_id: str, status: str, chunk_count: int = None):
    """Update the processing status and optionally chunk count."""
    if chunk_count is not None:
        with _get_connection() as conn:
            conn.execute("""
                UPDATE documents SET status=?, chunk_count=? WHERE doc_id=?
            """, (status, chunk_count, doc_id))
            conn.commit()
    else:
        with _get_connection() as conn:
            conn.execute("""
                UPDATE documents SET status=? WHERE doc_id=?
            """, (status, doc_id))
            conn.commit()


def get_by_filename(filename: str) -> dict | None:
    """Get the latest version of a document by filename."""
    with _get_connection() as conn:
        row = conn.execute("""
            SELECT * FROM documents
            WHERE filename=?
            ORDER BY version DESC
            LIMIT 1
        """, (filename,)).fetchone()
    return dict(row) if row else None


def get_by_doc_id(doc_id: str) -> dict | None:
    """Get a document record by its doc_id."""
    with _get_connection() as conn:
        row = conn.execute("""
            SELECT * FROM documents WHERE doc_id=?
        """, (doc_id,)).fetchone()
    return dict(row) if row else None


def list_all(source_type: str = None) -> list[dict]:
    """List all documents, optionally filtered by source_type."""
    with _get_connection() as conn:
        if source_type:
            rows = conn.execute("""
                SELECT * FROM documents
                WHERE source_type=?
                ORDER BY ingested_at DESC
            """, (source_type,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM documents
                ORDER BY ingested_at DESC
            """).fetchall()
    return [dict(r) for r in rows]


def mark_failed(doc_id: str, reason: str):
    """Mark a document as failed with a reason."""
    with _get_connection() as conn:
        conn.execute("""
            UPDATE documents SET status='failed', notes=? WHERE doc_id=?
        """, (reason, doc_id))
        conn.commit()
    print(f"❌ Marked failed: doc_id={doc_id} | reason={reason}")