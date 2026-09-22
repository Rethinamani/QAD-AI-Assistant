# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────
PROJECT_ROOT        = os.getenv("PROJECT_ROOT", ".")
DATA_DIR            = os.getenv("DATA_DIR", "./data")
UPLOADS_DIR         = os.getenv("UPLOADS_DIR", "./data/uploads")
PROCESSED_DIR       = os.getenv("PROCESSED_DIR", "./data/processed")
FIGURES_DIR         = os.getenv("FIGURES_DIR", "./data/figures")
LOGS_DIR            = os.getenv("LOGS_DIR", "./logs")

# ── ChromaDB ───────────────────────────────────────────────
CHROMA_PERSIST_DIR         = os.getenv("CHROMA_PERSIST_DIR", "./vectorstore/chroma_db")
CHROMA_COLLECTION_PDF       = os.getenv("CHROMA_COLLECTION_PDF", "pdf_chunks")
# Two Excel collections, chosen by filename: a file whose name contains
# "incident" lands in the first, "defect"/"bug" in the second. Anything
# else is rejected at upload time.
CHROMA_COLLECTION_INCIDENTS = os.getenv("CHROMA_COLLECTION_INCIDENTS", "incident_tickets")
CHROMA_COLLECTION_DEFECTS   = os.getenv("CHROMA_COLLECTION_DEFECTS", "defect_records")
# Speech transcribed from uploaded audio/video files.
CHROMA_COLLECTION_VIDEO     = os.getenv("CHROMA_COLLECTION_VIDEO", "video_transcripts")

# All collections the retrieval layer searches / indexes.
ALL_COLLECTIONS = [
    CHROMA_COLLECTION_PDF,
    CHROMA_COLLECTION_INCIDENTS,
    CHROMA_COLLECTION_DEFECTS,
    CHROMA_COLLECTION_VIDEO,
]


# ── Ollama ─────────────────────────────────────────────────
OLLAMA_BASE_URL     = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL    = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")
OLLAMA_EMBED_MODEL  = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# ── LLM providers (chat completion only — embeddings stay on Ollama) ──
# Provider used when a request doesn't specify one (e.g. the UI dropdown
# hasn't loaded yet).
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "ollama")

# Groq (https://console.groq.com/keys) — fast hosted inference.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# OpenAI / ChatGPT.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ── Ingestion / embedding throughput ───────────────────────
# Chunks per Ollama embed request. Larger = fewer HTTP round trips, but
# nomic-embed-text crashes the Ollama runner above ~128 on CPU.
EMBED_BATCH_SIZE   = int(os.getenv("EMBED_BATCH_SIZE", 128))
# Embed requests kept in flight at once. Keep at 1 for a CPU-only Ollama —
# concurrent requests just contend for the same cores and measure slightly
# SLOWER. Raise to 3-4 only when Ollama runs on a GPU (and set the server's
# OLLAMA_NUM_PARALLEL to match).
EMBED_CONCURRENCY  = int(os.getenv("EMBED_CONCURRENCY", 1))

# ── Retrieval ──────────────────────────────────────────────
TOP_K_DENSE         = int(os.getenv("TOP_K_DENSE", 20))
TOP_K_BM25          = int(os.getenv("TOP_K_BM25", 20))
TOP_K_AFTER_FUSION  = int(os.getenv("TOP_K_AFTER_FUSION", 10))
TOP_K_FINAL         = int(os.getenv("TOP_K_FINAL", 5))
CONFIDENCE_HIGH     = float(os.getenv("CONFIDENCE_HIGH", 0.75))
CONFIDENCE_LOW      = float(os.getenv("CONFIDENCE_LOW", 0.40))

# ── Image OCR (figure text extraction) ─────────────────────
# Runs Tesseract over saved figure images at ingestion time and folds
# the recovered text into the figure chunk. Degrades gracefully — if the
# Tesseract binary or pytesseract is missing, figures are indexed without
# extracted text and ingestion continues.
OCR_ENABLED             = os.getenv("OCR_ENABLED", "true").lower() == "true"
# Absolute path to tesseract.exe. Leave blank to use the system PATH.
TESSERACT_CMD           = os.getenv("TESSERACT_CMD", "")
# Tesseract page segmentation config. psm 6 = assume a uniform block of text,
# a reasonable default for screenshots and captioned figures.
OCR_TESSERACT_CONFIG    = os.getenv("OCR_TESSERACT_CONFIG", "--psm 6")
# Extracted text shorter than this (after cleaning) is treated as noise
# and dropped.
OCR_MIN_TEXT_LENGTH     = int(os.getenv("OCR_MIN_TEXT_LENGTH", 15))
# Images whose smallest side is below this are upscaled 2x before OCR so
# glyphs have enough pixels for Tesseract.
OCR_UPSCALE_THRESHOLD_PX = int(os.getenv("OCR_UPSCALE_THRESHOLD_PX", 1000))

# ── Audio / video transcription ────────────────────────────
# Speech-to-text runs locally with faster-whisper (CTranslate2). It needs
# the ffmpeg binary on PATH to pull the audio track out of a video.
#   winget install Gyan.FFmpeg
# Video/audio files dropped into uploads/ are transcribed, then the
# transcript is chunked by time window and embedded like any other source.
VIDEO_EXTENSIONS = [
    e.strip().lower()
    for e in os.getenv(
        "VIDEO_EXTENSIONS",
        ".mp4,.mov,.mkv,.avi,.webm,.m4v,.mp3,.wav,.m4a,.aac,.flac,.ogg",
    ).split(",")
    if e.strip()
]
# faster-whisper model size. "medium" is the most reliable on poor audio;
# drop to "small" or "base" for ~2-4x faster transcription of clean audio.
WHISPER_MODEL         = os.getenv("WHISPER_MODEL", "medium")
WHISPER_DEVICE        = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE  = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
# Whisper segments are merged into chunks of roughly this size before
# embedding — long enough to carry context, short enough to rank precisely.
TRANSCRIPT_CHUNK_CHARS    = int(os.getenv("TRANSCRIPT_CHUNK_CHARS", 700))
TRANSCRIPT_CHUNK_MAX_SECS = int(os.getenv("TRANSCRIPT_CHUNK_MAX_SECS", 60))
# Where transcript sidecar files (.transcript.txt, .srt, .segments.json)
# are written after a successful transcription.
TRANSCRIPT_OUTPUT_DIR = os.getenv("TRANSCRIPT_OUTPUT_DIR", "./data/transcripts")

# ── Video screen-text extraction (visual track) ────────────
# Screen recordings carry information the narration never says out loud —
# menu paths, field labels, error dialogs, program numbers. When enabled,
# ingestion samples frames from an uploaded video (on scene change, plus a
# frame at least every VIDEO_FRAME_MAX_INTERVAL_SEC), OCRs each frame with
# the same Tesseract path used for PDF figures, and folds the recovered
# on-screen text into the transcript chunks by timestamp. Degrades
# gracefully: if ffmpeg or OCR is unavailable, the file is transcribed
# audio-only exactly as before.
VIDEO_FRAME_OCR_ENABLED      = os.getenv("VIDEO_FRAME_OCR_ENABLED", "true").lower() == "true"
# ffmpeg scene-change score (0-1) above which a frame is sampled. Lower =
# more frames = slower but finer-grained. 0.3 suits UI screen recordings.
VIDEO_SCENE_THRESHOLD        = float(os.getenv("VIDEO_SCENE_THRESHOLD", 0.3))
# Sample a frame at least this often even when nothing on screen changes.
VIDEO_FRAME_MAX_INTERVAL_SEC = int(os.getenv("VIDEO_FRAME_MAX_INTERVAL_SEC", 30))
# Hard cap on frames OCR'd per video — protects CPU on very long recordings.
VIDEO_FRAME_MAX_COUNT        = int(os.getenv("VIDEO_FRAME_MAX_COUNT", 400))
# Frames wider than this are downscaled before OCR (keeps Tesseract fast;
# UI text stays legible well below native 1080p/4K).
VIDEO_FRAME_SCALE_WIDTH      = int(os.getenv("VIDEO_FRAME_SCALE_WIDTH", 1920))
# Consecutive frames whose OCR text is at least this similar (0-1) are
# treated as the same screen and de-duplicated.
VIDEO_FRAME_DEDUP_RATIO      = float(os.getenv("VIDEO_FRAME_DEDUP_RATIO", 0.90))
# Tesseract config for full-screen frames. psm 11 ("sparse text") finds
# scattered UI labels a uniform-block mode (psm 6) misses.
VIDEO_OCR_TESSERACT_CONFIG   = os.getenv("VIDEO_OCR_TESSERACT_CONFIG", "--psm 11")
# How many representative frames to keep as displayable "figure" chunks
# (shown next to answers, like PDF figures). 0 disables figure chunks.
VIDEO_FIGURE_MAX_COUNT       = int(os.getenv("VIDEO_FIGURE_MAX_COUNT", 6))
# When a stretch of narration has no frame of its own (the presenter kept
# talking about a screen they navigated to earlier), carry the last-seen
# on-screen text forward onto that chunk so the words stay paired with the
# screen. Only carried when the previous frame is within CARRY_MAX_SEC.
VIDEO_SCREEN_CARRY_FORWARD   = os.getenv("VIDEO_SCREEN_CARRY_FORWARD", "true").lower() == "true"
VIDEO_SCREEN_CARRY_MAX_SEC   = int(os.getenv("VIDEO_SCREEN_CARRY_MAX_SEC", 120))
# Keep the raw extracted frames on disk after ingestion (debugging). When
# false, only the persisted figure frames survive.
VIDEO_KEEP_RAW_FRAMES        = os.getenv("VIDEO_KEEP_RAW_FRAMES", "false").lower() == "true"

# ── Memory ─────────────────────────────────────────────────
MEMORY_WINDOW_SIZE  = int(os.getenv("MEMORY_WINDOW_SIZE", 5))

# ── Guardrails ─────────────────────────────────────────────
MAX_QUERY_LENGTH        = int(os.getenv("MAX_QUERY_LENGTH", 500))
MAX_RESPONSE_TOKENS     = int(os.getenv("MAX_RESPONSE_TOKENS", 600))

# ── Security ───────────────────────────────────────────────
# PII entity types Presidio detects and redacts during ingestion.
# URL and LOCATION are excluded by default: URL redaction hurts retrieval
# (it nukes doc links) and LOCATION is a costly spaCy-NER label that rarely
# carries real PII in ERP documentation. Re-add via the env var if needed.
PII_ENTITIES = [
    e.strip()
    for e in os.getenv(
        "PII_ENTITIES",
        "PERSON,EMAIL_ADDRESS,PHONE_NUMBER,IP_ADDRESS,US_SSN,CREDIT_CARD,IBAN_CODE",
    ).split(",")
    if e.strip()
]
# spaCy model for Presidio NLP. Small model + a trimmed pipeline is ~2-4x
# faster than the large model with a negligible recall difference for the
# entity types above. Falls back to en_core_web_lg if this is not installed.
PII_SPACY_MODEL = os.getenv("PII_SPACY_MODEL", "en_core_web_sm")

SENSITIVITY_HIGH_KEYWORDS = [
    kw.strip()
    for kw in os.getenv(
        "SENSITIVITY_HIGH_KEYWORDS",
        "salary,payroll,confidential,executive,password,credential"
    ).split(",")
]

# ── Registry ───────────────────────────────────────────────
REGISTRY_DB_PATH = os.getenv("REGISTRY_DB_PATH", "./registry/document_registry.db")

# ── Conversation history ───────────────────────────────────
# Server-side store of chat conversations + messages so history survives
# API restarts and browser refreshes (ChatGPT-style sidebar).
CONVERSATION_DB_PATH = os.getenv(
    "CONVERSATION_DB_PATH", "./registry/conversations.db"
)

