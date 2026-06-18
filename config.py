# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────
PROJECT_ROOT        = os.getenv("PROJECT_ROOT", ".")
DATA_DIR            = os.getenv("DATA_DIR", "./data")
UPLOADS_DIR         = os.getenv("UPLOADS_DIR", "./data/uploads")
PROCESSED_DIR       = os.getenv("PROCESSED_DIR", "./data/processed")
NIGHTLY_DROP_DIR    = os.getenv("NIGHTLY_DROP_DIR", "./data/nightly_drop")
FIGURES_DIR         = os.getenv("FIGURES_DIR", "./data/figures")
LOGS_DIR            = os.getenv("LOGS_DIR", "./logs")

# ── ChromaDB ───────────────────────────────────────────────
CHROMA_PERSIST_DIR      = os.getenv("CHROMA_PERSIST_DIR", "./vectorstore/chroma_db")
CHROMA_COLLECTION_PDF   = os.getenv("CHROMA_COLLECTION_PDF", "pdf_chunks")
CHROMA_COLLECTION_EXCEL = os.getenv("CHROMA_COLLECTION_EXCEL", "excel_errors")
CHROMA_COLLECTION_SN    = os.getenv("CHROMA_COLLECTION_SERVICENOW", "servicenow_tickets")

# ── Ollama ─────────────────────────────────────────────────
OLLAMA_BASE_URL     = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL    = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")
OLLAMA_EMBED_MODEL  = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# ── Retrieval ──────────────────────────────────────────────
TOP_K_DENSE         = int(os.getenv("TOP_K_DENSE", 20))
TOP_K_BM25          = int(os.getenv("TOP_K_BM25", 20))
TOP_K_AFTER_FUSION  = int(os.getenv("TOP_K_AFTER_FUSION", 10))
TOP_K_FINAL         = int(os.getenv("TOP_K_FINAL", 5))
CONFIDENCE_HIGH     = float(os.getenv("CONFIDENCE_HIGH", 0.75))
CONFIDENCE_LOW      = float(os.getenv("CONFIDENCE_LOW", 0.40))

# ── Memory ─────────────────────────────────────────────────
MEMORY_WINDOW_SIZE  = int(os.getenv("MEMORY_WINDOW_SIZE", 5))

# ── Guardrails ─────────────────────────────────────────────
MAX_QUERY_LENGTH        = int(os.getenv("MAX_QUERY_LENGTH", 500))
MAX_RESPONSE_TOKENS     = int(os.getenv("MAX_RESPONSE_TOKENS", 400))

# ── Security ───────────────────────────────────────────────
SENSITIVITY_HIGH_KEYWORDS = [
    kw.strip()
    for kw in os.getenv(
        "SENSITIVITY_HIGH_KEYWORDS",
        "salary,payroll,confidential,executive,password,credential"
    ).split(",")
]

# ── Registry ───────────────────────────────────────────────
REGISTRY_DB_PATH = os.getenv("REGISTRY_DB_PATH", "./registry/document_registry.db")