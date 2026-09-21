# api/main.py

import logging
import logging.handlers
import threading
import shutil
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import (
    UPLOADS_DIR, LOGS_DIR, FIGURES_DIR, CHROMA_COLLECTION_PDF, VIDEO_EXTENSIONS,
)
from chatbot.chain import chat, create_session, get_session_history
from chatbot.servicenow_handler import handle_escalation
from chatbot import conversation_store
from registry.document_registry import initialize_registry, list_all
from retrieval.bm25_index import build_all_indexes, build_bm25_index
from vectorstore.chroma_store import chroma_store
from ingestion.file_watcher import start_file_watcher

# ── Logging ────────────────────────────────────────────────────────────────────
# File handler always writes UTF-8 so emoji/unicode in messages (video titles,
# etc.) never crash logging regardless of the console's codepage.
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            os.path.join(LOGS_DIR, "api.log"),
            maxBytes=10_000_000,
            backupCount=5,
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)


# ── Startup / Shutdown ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup tasks before serving requests."""
    logger.info("Starting QAD Support Assistant API...")

    initialize_registry()
    conversation_store.initialize_conversation_store()
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

# Serve extracted figure images (data/figures/<doc_id>/fig_N.png) so the
# frontend can render them directly by URL.
os.makedirs(FIGURES_DIR, exist_ok=True)
app.mount("/figures", StaticFiles(directory=FIGURES_DIR), name="figures")


# ── Request size limit middleware ──────────────────────────────────────────────
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    """Reject oversized uploads (video files need more headroom than docs)."""
    max_mb = 500
    if request.method == "POST" and "upload" in str(request.url):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > max_mb * 1024 * 1024:
            return JSONResponse(
                status_code=413,
                content={"detail": f"File too large. Maximum size is {max_mb}MB."},
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
    images:            list[dict] = []
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


# ── Conversation history (ChatGPT-style sidebar) ───────────────────────────────
class RenameRequest(BaseModel):
    title: str


@app.get("/conversations")
def list_conversations():
    """All conversations, most recently updated first."""
    return {"conversations": conversation_store.list_conversations()}


@app.post("/conversations")
def create_conversation():
    """Create an empty conversation and return its id."""
    conversation_id = conversation_store.create_conversation()
    return {"conversation_id": conversation_id, "title": conversation_store.DEFAULT_TITLE}


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    """Full transcript for one conversation."""
    convo = conversation_store.get_conversation(conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {
        "conversation_id": conversation_id,
        "title":           convo["title"],
        "messages":        conversation_store.get_messages(conversation_id),
    }


@app.patch("/conversations/{conversation_id}")
def rename_conversation(conversation_id: str, request: RenameRequest):
    """Rename a conversation."""
    if not conversation_store.get_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    conversation_store.rename_conversation(conversation_id, request.title)
    return {"status": "ok"}


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    """Delete a conversation and all its messages."""
    conversation_store.delete_conversation(conversation_id)
    return {"status": "ok"}


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

        conversation_id = result["session_id"]

        # Persist the exchange so it shows up in the sidebar and survives
        # restarts. Title the conversation from its first user message.
        try:
            is_first = conversation_store.message_count(conversation_id) == 0
            conversation_store.append_message(
                conversation_id, "user", request.query,
            )
            conversation_store.append_message(
                conversation_id, "assistant", result["answer"],
                confidence=result["confidence"],
                confidence_band=result["confidence_band"],
                sources=result["sources"],
                images=result.get("images", []),
                status=result["status"],
            )
            if is_first:
                conversation_store.rename_conversation(
                    conversation_id,
                    conversation_store.auto_title_from(request.query),
                )
        except Exception as e:
            logger.warning(f"Could not persist conversation {conversation_id}: {e}")

        return ChatResponse(
            session_id=conversation_id,
            answer=result["answer"],
            confidence=result["confidence"],
            confidence_band=result["confidence_band"],
            sources=result["sources"],
            images=result.get("images", []),
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
            <p>Supported formats: <strong>PDF, XLSX, XLS, MP4/MOV/MKV, MP3/WAV</strong></p>
            <p style="color:#666; font-size:13px;">
                Duplicate files will be detected automatically.
                New files are ingested immediately after upload.
            </p>

            <form id="uploadForm">
                <input type="file" id="fileInput" name="file"
                       accept=".pdf,.xlsx,.xls,.mp4,.mov,.mkv,.avi,.webm,.m4v,.mp3,.wav,.m4a,.aac,.flac,.ogg" required>
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
    Upload a PDF or Excel document for ingestion.

    PDF   → pdf_chunks collection.
    Excel → collection chosen from the file name: 'incident' in the name →
            incident_tickets, 'defect'/'bug' → defect_records, otherwise
            the upload is rejected.
    Audio / video → transcribed locally, transcript → video_transcripts.
    """
    allowed_extensions = {".pdf", ".xlsx", ".xls"} | set(VIDEO_EXTENSIONS)
    _, ext = os.path.splitext(file.filename)
    ext = ext.lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(allowed_extensions)}",
        )

    # ── Duplicate check ────────────────────────────────────────────────────────
    from registry.document_registry import get_by_filename
    existing = get_by_filename(file.filename)
    if existing and existing["status"] == "ready":
        return {
            "status":   "duplicate",
            "message": (
                f"File '{file.filename}' has already been ingested "
                f"({existing['chunk_count']} chunks, {existing['ingested_at']}). "
                f"Upload it under a different name to re-ingest."
            ),
            "doc_id":   existing["doc_id"],
            "filename": file.filename,
        }

    # ── Excel routing check BEFORE saving ─────────────────────────────────────
    if ext in {".xlsx", ".xls"}:
        from ingestion.excel_ingester import resolve_excel_target
        try:
            resolve_excel_target(file.filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # ── Save file ──────────────────────────────────────────────────────────────
    save_path = os.path.join(UPLOADS_DIR, file.filename)
    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"File saved: {save_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # ── Trigger ingestion ─────────────────────────────────────────────────────
    try:
        if ext == ".pdf":
            from ingestion.pdf_ingester import ingest_pdf
            summary = ingest_pdf(save_path)
            build_bm25_index(CHROMA_COLLECTION_PDF, force_rebuild=True)
        elif ext in set(VIDEO_EXTENSIONS):
            from ingestion.video_ingester import ingest_video
            summary = ingest_video(save_path)
            build_bm25_index(summary["collection"], force_rebuild=True)
        else:
            from ingestion.excel_ingester import ingest_excel
            summary = ingest_excel(save_path)
            build_bm25_index(summary["collection"], force_rebuild=True)

        return {
            "status":      "success",
            "message":     f"File '{file.filename}' ingested successfully.",
            "filename":    file.filename,
            "doc_id":      summary["doc_id"],
            "chunk_count": summary.get("total_chunks") or summary.get("ingested"),
        }

    except Exception as e:
        logger.error(f"Ingestion failed for {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"File saved but ingestion failed: {e}",
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


