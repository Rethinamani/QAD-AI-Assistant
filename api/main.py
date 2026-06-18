# api/main.py

import logging
import threading
import shutil
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import UPLOADS_DIR
from chatbot.chain import chat, create_session, get_session_history
from chatbot.servicenow_handler import handle_escalation
from registry.document_registry import initialize_registry, list_all
from retrieval.bm25_index import build_all_indexes
from vectorstore.chroma_store import chroma_store
from ingestion.file_watcher import start_file_watcher
from config import (
    CHROMA_COLLECTION_PDF,
    CHROMA_COLLECTION_EXCEL,
    CHROMA_COLLECTION_SN,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Startup / Shutdown ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup tasks before serving requests."""
    logger.info("Starting QAD Support Assistant API...")

    initialize_registry()
    build_all_indexes(force_rebuild=False)

    watcher_thread = threading.Thread(
        target=start_file_watcher,
        daemon=True,
        name="file-watcher",
    )
    watcher_thread.start()
    logger.info("File watcher started in background.")
    logger.info("API ready.")
    yield
    logger.info("Shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="QAD Support Assistant",
    description="RAG-powered support chatbot for QAD ERP system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request size limit middleware ──────────────────────────────────────────────
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    """Reject requests larger than 100MB."""
    if request.method == "POST" and "upload" in str(request.url):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 100 * 1024 * 1024:
            return JSONResponse(
                status_code=413,
                content={"detail": "File too large. Maximum size is 100MB."},
            )
    return await call_next(request)


# ── Request / Response Models ──────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query:      str
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id:        str
    answer:            str
    confidence:        float
    confidence_band:   str
    sources:           list[str]
    should_escalate:   bool
    status:            str


class EscalationRequest(BaseModel):
    user_response:  str
    original_query: str
    session_id:     str


class EscalationResponse(BaseModel):
    action:      str
    message:     str
    incident_id: str | None = None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Check API and ChromaDB status."""
    try:
        stats = chroma_store.all_stats()
        return {
            "status":      "healthy",
            "collections": stats,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session")
def new_session():
    """Create a new conversation session."""
    session_id = create_session()
    return {"session_id": session_id}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint.
    Accepts a query and optional session_id.
    Returns answer with confidence and sources.
    """
    try:
        result = chat(
            query=request.query,
            session_id=request.session_id,
        )
        return ChatResponse(
            session_id=result["session_id"],
            answer=result["answer"],
            confidence=result["confidence"],
            confidence_band=result["confidence_band"],
            sources=result["sources"],
            should_escalate=result["should_escalate"],
            status=result["status"],
        )
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/escalate", response_model=EscalationResponse)
def escalate_endpoint(request: EscalationRequest):
    """
    Handle user's response to escalation prompt.
    Called after /chat returns should_escalate=True.
    """
    try:
        history = get_session_history(request.session_id)
        result  = handle_escalation(
            user_response=request.user_response,
            query=request.original_query,
            session_id=request.session_id,
            conversation_history=history,
        )
        return EscalationResponse(
            action=result["action"],
            message=result["message"],
            incident_id=(
                result["incident"]["incident_id"]
                if result.get("incident")
                else None
            ),
        )
    except Exception as e:
        logger.error(f"Escalation endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/upload-form", response_class=HTMLResponse)
def upload_form():
    """Simple HTML upload form for testing without Swagger."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>QAD Support — Upload Document</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 650px;
                margin: 50px auto;
                padding: 0 20px;
                background: #f5f5f5;
            }
            .card {
                background: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            h2 { color: #0066cc; }
            input[type=file] {
                display: block;
                margin: 15px 0;
                padding: 10px;
                border: 2px dashed #0066cc;
                border-radius: 4px;
                width: 100%;
                box-sizing: border-box;
            }
            button {
                padding: 12px 30px;
                background: #0066cc;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 15px;
                width: 100%;
            }
            button:hover { background: #0052a3; }
            #result {
                margin-top: 20px;
                padding: 15px;
                border-radius: 4px;
                display: none;
                white-space: pre-wrap;
                font-family: monospace;
                font-size: 13px;
            }
            .success { background: #e6ffe6; border: 1px solid #00cc00; }
            .error   { background: #ffe6e6; border: 1px solid #cc0000; }
            .duplicate { background: #fff3e6; border: 1px solid #ff9900; }
            .links { margin-top: 20px; }
            .links a {
                margin-right: 15px;
                color: #0066cc;
                text-decoration: none;
            }
            .links a:hover { text-decoration: underline; }
            .spinner { display: none; margin: 10px 0; color: #666; }
        </style>
    </head>
    <body>
        <div class="card">
            <h2>📄 QAD Support — Upload Document</h2>
            <p>Supported formats: <strong>PDF, XLSX, XLS</strong></p>
            <p style="color:#666; font-size:13px;">
                Duplicate files will be detected automatically.
                New files are ingested immediately after upload.
            </p>

            <form id="uploadForm">
                <input type="file" id="fileInput" name="file"
                       accept=".pdf,.xlsx,.xls" required>
                <div class="spinner" id="spinner">
                    ⏳ Ingesting document — this may take a few minutes
                    for large PDFs...
                </div>
                <button type="submit" id="submitBtn">
                    Upload and Ingest
                </button>
            </form>

            <div id="result"></div>

            <div class="links">
                <hr>
                <a href="/documents" target="_blank">📋 View Documents</a>
                <a href="/stats" target="_blank">📊 Stats</a>
                <a href="/health" target="_blank">❤️ Health</a>
                <a href="/docs" target="_blank">📖 API Docs</a>
            </div>
        </div>

        <script>
            document.getElementById('uploadForm').onsubmit = async function(e) {
                e.preventDefault();

                const fileInput = document.getElementById('fileInput');
                const resultDiv = document.getElementById('result');
                const spinner   = document.getElementById('spinner');
                const submitBtn = document.getElementById('submitBtn');

                if (!fileInput.files[0]) {
                    alert('Please select a file first.');
                    return;
                }

                // Show spinner
                spinner.style.display  = 'block';
                submitBtn.disabled     = true;
                submitBtn.textContent  = 'Uploading...';
                resultDiv.style.display = 'none';

                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body:   formData,
                    });

                    const data = await response.json();

                    resultDiv.style.display = 'block';
                    resultDiv.className     = '';

                    if (data.status === 'success') {
                        resultDiv.classList.add('success');
                        resultDiv.textContent = (
                            '✅ Success!\\n' +
                            'File: '        + data.filename    + '\\n' +
                            'Doc ID: '      + data.doc_id      + '\\n' +
                            'Chunks: '      + data.chunk_count + '\\n' +
                            'Status: Ready for querying.'
                        );
                    } else if (data.status === 'duplicate') {
                        resultDiv.classList.add('duplicate');
                        resultDiv.textContent = '⚠️ Duplicate\\n' + data.message;
                    } else {
                        resultDiv.classList.add('error');
                        resultDiv.textContent = '❌ Error\\n' + JSON.stringify(data, null, 2);
                    }

                } catch (err) {
                    resultDiv.style.display = 'block';
                    resultDiv.className     = 'error';
                    resultDiv.textContent   = '❌ Request failed: ' + err.message;
                } finally {
                    spinner.style.display = 'none';
                    submitBtn.disabled    = false;
                    submitBtn.textContent = 'Upload and Ingest';
                }
            };
        </script>
    </body>
    </html>
    """


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a new PDF or Excel document for ingestion.
    Checks for duplicates before saving.
    Triggers ingestion pipeline directly and returns status.
    """
    allowed_extensions = {".pdf", ".xlsx", ".xls"}
    _, ext = os.path.splitext(file.filename)

    if ext.lower() not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Allowed: {allowed_extensions}"
            ),
        )

    # ── Duplicate check ────────────────────────────────────────────────────────
    from registry.document_registry import get_by_filename
    existing = get_by_filename(file.filename)
    if existing and existing["status"] == "ready":
        return {
            "status":   "duplicate",
            "message":  (
                f"File '{file.filename}' has already been ingested "
                f"(version {existing['version']}, "
                f"{existing['chunk_count']} chunks, "
                f"ingested at {existing['ingested_at']}). "
                f"Upload a file with a different name to add new content."
            ),
            "doc_id":   existing["doc_id"],
            "filename": file.filename,
        }

    # ── Save file ──────────────────────────────────────────────────────────────
    save_path = os.path.join(UPLOADS_DIR, file.filename)
    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"File saved: {save_path}")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {e}"
        )

    # ── Trigger ingestion ──────────────────────────────────────────────────────
    try:
        from ingestion.pdf_ingester import ingest_pdf
        from ingestion.excel_ingester import ingest_excel
        from retrieval.bm25_index import build_bm25_index

        if ext.lower() == ".pdf":
            summary = ingest_pdf(save_path)
            build_bm25_index(CHROMA_COLLECTION_PDF, force_rebuild=True)
        else:
            summary = ingest_excel(save_path)
            build_bm25_index(CHROMA_COLLECTION_EXCEL, force_rebuild=True)

        return {
            "status":      "success",
            "message":     f"File '{file.filename}' ingested successfully.",
            "filename":    file.filename,
            "doc_id":      summary["doc_id"],
            "chunk_count": (
                summary.get("total_chunks")
                or summary.get("ingested")
            ),
        }

    except Exception as e:
        logger.error(f"Ingestion failed for {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"File saved but ingestion failed: {str(e)}",
        )


@app.get("/documents")
def list_documents(source_type: str = None):
    """List all ingested documents from the registry."""
    docs = list_all(source_type=source_type)
    return {"documents": docs, "count": len(docs)}


@app.get("/stats")
def collection_stats():
    """Return chunk counts for all ChromaDB collections."""
    return {"collections": chroma_store.all_stats()}