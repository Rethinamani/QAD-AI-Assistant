# delete_from_registry.py
import sqlite3
from config import REGISTRY_DB_PATH

def list_documents():
    with sqlite3.connect(REGISTRY_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT doc_id, filename, source_type, version, status, chunk_count, ingested_at FROM documents ORDER BY ingested_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]

def delete_by_doc_id(doc_id: str):
    with sqlite3.connect(REGISTRY_DB_PATH) as conn:
        conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        conn.commit()
    print(f"✅ Deleted doc_id: {doc_id}")

def delete_by_filename(filename: str):
    with sqlite3.connect(REGISTRY_DB_PATH) as conn:
        conn.execute("DELETE FROM documents WHERE filename = ?", (filename,))
        conn.commit()
    print(f"✅ Deleted all records for filename: {filename}")

def delete_all():
    with sqlite3.connect(REGISTRY_DB_PATH) as conn:
        conn.execute("DELETE FROM documents")
        conn.commit()
    print("✅ All registry records deleted.")

# ── Run ────────────────────────────────────────────────────────────────────────
print("\n── Current Registry ──\n")
docs = list_documents()

if not docs:
    print("Registry is empty.")
else:
    for doc in docs:
        print(f"  doc_id   : {doc['doc_id']}")
        print(f"  filename : {doc['filename']}")
        print(f"  type     : {doc['source_type']}")
        print(f"  version  : {doc['version']}")
        print(f"  status   : {doc['status']}")
        print(f"  chunks   : {doc['chunk_count']}")
        print(f"  ingested : {doc['ingested_at']}")
        print()

    print("Options:")
    print("  1. Delete by doc_id")
    print("  2. Delete by filename")
    print("  3. Delete ALL records")
    print("  4. Exit")

    choice = input("\nEnter choice (1/2/3/4): ").strip()

    if choice == "1":
        doc_id = input("Enter doc_id: ").strip()
        delete_by_doc_id(doc_id)
    elif choice == "2":
        filename = input("Enter filename: ").strip()
        delete_by_filename(filename)
    elif choice == "3":
        confirm = input("Are you sure? This deletes ALL records (yes/no): ").strip().lower()
        if confirm == "yes":
            delete_all()
        else:
            print("Cancelled.")
    elif choice == "4":
        print("Exited.")
    else:
        print("Invalid choice.")