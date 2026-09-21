# ingestion/video_ingester.py

"""
Audio / video transcript ingester.

Transcribes the media file (ingestion.transcriber), merges the Whisper
segments into time-windowed chunks, and embeds them into the
video_transcripts collection. One extra "transcript overview" chunk holds
the whole transcript so "what is the video about" style questions land.
"""

import os
import time
import logging

from ollama import Client

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    EMBED_BATCH_SIZE,
    CHROMA_COLLECTION_VIDEO,
    TRANSCRIPT_CHUNK_CHARS,
    TRANSCRIPT_CHUNK_MAX_SECS,
    TRANSCRIPT_OUTPUT_DIR,
    VIDEO_SCREEN_CARRY_FORWARD,
    VIDEO_SCREEN_CARRY_MAX_SEC,
)
from vectorstore.chroma_store import chroma_store
from registry.document_registry import (
    initialize_registry,
    register_document,
    update_status,
    mark_failed,
    get_by_filename,
)
from ingestion.transcriber import transcribe_media, format_timestamp
from ingestion.visual_extractor import extract_screen_text
from security.pii_scrubber import scrub_chunks_batch
from security.sensitivity_tagger import tag_sensitivity

logger = logging.getLogger(__name__)

ollama_client = Client(host=OLLAMA_BASE_URL)

# Overview chunk caps.
SUMMARY_MAX_CHARS = 1500


class DuplicateFileError(Exception):
    """Raised when a file with the same name was already ingested."""


# ── Chunking ──────────────────────────────────────────────────────────────────
def _window_segments(segments: list[dict]) -> list[dict]:
    """
    Merge consecutive Whisper segments into chunks that are at most
    TRANSCRIPT_CHUNK_CHARS long or TRANSCRIPT_CHUNK_MAX_SECS of speech.

    Returns [{start, end, text}, ...].
    """
    chunks: list[dict] = []
    buf: list[str] = []
    buf_start = None
    buf_end = None

    for seg in segments:
        if buf_start is None:
            buf_start = seg["start"]
        buf.append(seg["text"])
        buf_end = seg["end"]

        long_enough = len(" ".join(buf)) >= TRANSCRIPT_CHUNK_CHARS
        wide_enough = (buf_end - buf_start) >= TRANSCRIPT_CHUNK_MAX_SECS
        if long_enough or wide_enough:
            chunks.append({
                "start": buf_start,
                "end":   buf_end,
                "text":  " ".join(buf).strip(),
            })
            buf, buf_start, buf_end = [], None, None

    if buf:
        chunks.append({
            "start": buf_start,
            "end":   buf_end,
            "text":  " ".join(buf).strip(),
        })
    return chunks


def _segment_text(filename: str, start: float, end: float, body: str) -> str:
    return (
        f"[{filename} · {format_timestamp(start)}–{format_timestamp(end)}]\n"
        f"{body}"
    )


def _segment_metadata(
    filename: str, doc_id: str, index: int, start: float, end: float,
    chunk_type: str = "transcript_segment", has_screen_text: bool = False,
    screen_inherited: bool = False,
) -> dict:
    return {
        "source_type":   "video",
        "source_file":   filename,
        "doc_id":        doc_id,
        "chunk_type":    chunk_type,
        "segment_index": int(index),
        "start_sec":     round(float(start), 2),
        "end_sec":       round(float(end), 2),
        "start_time":    format_timestamp(start),
        "end_time":      format_timestamp(end),
        "has_screen_text":  bool(has_screen_text),
        "screen_inherited": bool(screen_inherited),
        # keep a 'page'-like field so shared logging / tagging code is happy
        "page":          int(index),
        "row_key":       format_timestamp(start),
    }


# ── Visual track (on-screen text) ─────────────────────────────────────────────
SCREEN_JOIN_MAX_CHARS = TRANSCRIPT_CHUNK_CHARS * 2
_WINDOW_SLACK_SECS = 2.0


def _join_screen_lines(screen_texts: list[str]) -> str:
    """Flatten OCR blocks to a deduped ' | '-joined single line."""
    seen: list[str] = []
    for block in screen_texts:
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            low = line.lower()
            if any(low == s.lower() or low in s.lower() for s in seen):
                continue
            seen.append(line)
    joined = " | ".join(seen)
    if len(joined) > SCREEN_JOIN_MAX_CHARS:
        joined = joined[:SCREEN_JOIN_MAX_CHARS].rsplit(" | ", 1)[0] + " | …"
    return joined


def _attach_screen_text(windows: list[dict], visuals: list[dict]) -> list[dict]:
    """
    Fold each OCR entry into the transcript window whose time span contains
    it. A window that gets no frame of its own inherits the last on-screen
    text from before it started (the presenter is still on that screen while
    they keep talking). Returns the entries that fell outside every window.
    """
    ordered = sorted(visuals, key=lambda v: v["start_sec"])
    for w in windows:
        w["screen"] = []
        w["screen_inherited"] = False

    orphans: list[dict] = []
    for v in ordered:
        home = next(
            (w for w in windows
             if w["start"] - _WINDOW_SLACK_SECS <= v["start_sec"]
             <= w["end"] + _WINDOW_SLACK_SECS),
            None,
        )
        if home is None:
            orphans.append(v)
        else:
            home["screen"].append(v["text"])

    if VIDEO_SCREEN_CARRY_FORWARD:
        for w in sorted(windows, key=lambda x: x["start"]):
            if w["screen"]:
                continue
            earlier = [v for v in ordered
                       if v["start_sec"] <= w["start"] + _WINDOW_SLACK_SECS]
            if earlier and (w["start"] - earlier[-1]["start_sec"]) <= VIDEO_SCREEN_CARRY_MAX_SEC:
                w["screen"] = [earlier[-1]["text"]]
                w["screen_inherited"] = True

    return orphans


def _window_visuals(orphans: list[dict]) -> list[dict]:
    """Group screen-only OCR entries (no narration) into time windows."""
    chunks: list[dict] = []
    buf: list[str] = []
    buf_start = buf_end = None
    for v in sorted(orphans, key=lambda x: x["start_sec"]):
        if buf_start is None:
            buf_start = v["start_sec"]
        elif (v["start_sec"] - buf_end > TRANSCRIPT_CHUNK_MAX_SECS
              or len(" ".join(buf)) >= TRANSCRIPT_CHUNK_CHARS):
            chunks.append({"start": buf_start, "end": buf_end,
                           "narration": "", "screen": buf})
            buf, buf_start = [], v["start_sec"]
        buf.append(v["text"])
        buf_end = v["start_sec"]
    if buf:
        chunks.append({"start": buf_start, "end": buf_end,
                       "narration": "", "screen": buf})
    return chunks


def _compose_body(narration: str, screen: list[str], inherited: bool = False) -> str:
    screen_line = _join_screen_lines(screen) if screen else ""
    label = "Still on screen" if inherited else "On screen"
    if narration and screen_line:
        return f"Narration: {narration}\n{label}: {screen_line}"
    if narration:
        return narration
    return f"{label}: {screen_line}"


def _figure_text(filename: str, start: float, ocr_text: str) -> str:
    return (
        f"[{filename} · screen at {format_timestamp(start)}]\n"
        f"Screenshot from the recording '{filename}' at {format_timestamp(start)}.\n"
        f"Text on screen: {ocr_text}"
    )


def _figure_metadata(
    filename: str, doc_id: str, index: int, start: float, image_path: str
) -> dict:
    meta = _segment_metadata(
        filename, doc_id, index, start, start,
        chunk_type="figure", has_screen_text=True,
    )
    meta["image_path"] = image_path
    return meta


def _summary_chunk(
    filename: str, doc_id: str, result: dict, n_segments: int
) -> tuple[str, dict]:
    dur = format_timestamp(result.get("duration_sec", 0.0))
    transcript = result["text"]
    if len(transcript) > SUMMARY_MAX_CHARS:
        transcript = transcript[:SUMMARY_MAX_CHARS].rsplit(" ", 1)[0] + " …"
    text = (
        f"[{filename} · transcript overview]\n"
        f"This is the spoken-word transcript of the recording '{filename}' "
        f"(duration {dur}, language {result.get('language', 'unknown')}, "
        f"{n_segments} spoken passages).\n"
        f"Full transcript: {transcript}"
    )
    meta = {
        "source_type":   "video",
        "source_file":   filename,
        "doc_id":        doc_id,
        "chunk_type":    "transcript_summary",
        "segment_index": 0,
        "start_sec":     0.0,
        "end_sec":       round(float(result.get("duration_sec", 0.0)), 2),
        "start_time":    format_timestamp(0),
        "end_time":      format_timestamp(result.get("duration_sec", 0.0)),
        "page":          0,
        "row_key":       "",
    }
    return text, meta


def _get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    return ollama_client.embed(model=OLLAMA_EMBED_MODEL, input=texts)["embeddings"]


def _write_screen_timeline(
    filename: str, result: dict, composed: list[dict], figures: list[dict]
) -> str | None:
    """
    Write a human-readable merged screen+audio timeline next to the transcript
    sidecars (data/transcripts/<base>/<base>.screen_timeline.md).

    Uses the raw pre-PII text — this is a local debugging aid, not stored or
    served. Best-effort: never raises, returns the path or None.
    """
    base = os.path.splitext(filename)[0]
    out_dir = os.path.join(TRANSCRIPT_OUTPUT_DIR, base)
    path = os.path.join(out_dir, f"{base}.screen_timeline.md")

    n_narr   = sum(1 for c in composed if c["narration"].strip())
    n_screen = sum(1 for c in composed if c["screen"])
    lines = [
        f"# Screen + audio timeline — {filename}",
        "",
        f"Duration {format_timestamp(result.get('duration_sec', 0.0))} · "
        f"language {result.get('language', 'unknown')} · "
        f"{n_narr} narration passage(s) · {n_screen} with on-screen text · "
        f"{len(figures)} screenshot(s)",
        "",
        "_Generated by ingestion/video_ingester.py. Raw OCR / transcript "
        "text (before PII scrub). Not stored in ChromaDB._",
        "",
        "---",
        "",
    ]

    for c in composed:
        span = f"{format_timestamp(c['start'])}–{format_timestamp(c['end'])}"
        tag = "  · _screen carried forward_" if c.get("inherited") else ""
        lines.append(f"## {span}{tag}")
        lines.append("")
        if c["narration"].strip():
            lines.append(f"**🔊 Narration:** {c['narration'].strip()}")
            lines.append("")
        if c["screen"]:
            lines.append(f"**🖥️ On screen:** {_join_screen_lines(c['screen'])}")
            lines.append("")

    if figures:
        lines += ["---", "", "## Screenshots", ""]
        for i, fig in enumerate(figures):
            preview = " ".join(fig["text"].split())[:140]
            lines.append(
                f"- **{format_timestamp(fig['start_sec'])}** — "
                f"`{os.path.basename(fig['image_path'])}` — {preview}"
            )

    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        logger.info(f"Screen timeline written to {path}")
        return path
    except Exception as e:
        logger.warning(f"Could not write screen timeline: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────
def ingest_video(file_path: str) -> dict:
    """
    Transcribe an audio/video file and ingest its transcript.

    Raises DuplicateFileError if a file with this name is already ingested.
    """
    initialize_registry()
    filename = os.path.basename(file_path)

    existing = get_by_filename(filename)
    if existing and existing["status"] == "ready":
        raise DuplicateFileError(
            f"'{filename}' was already ingested "
            f"({existing['chunk_count']} chunks, {existing['ingested_at']}). "
            f"Upload it under a different name to re-ingest."
        )

    logger.info(f"Video ingestion: {filename} → {CHROMA_COLLECTION_VIDEO}")
    doc_id = register_document(filename, source_type="video")

    try:
        result = transcribe_media(file_path)
        segments = result["segments"]
        if not segments or not result["text"].strip():
            raise ValueError(
                "Transcription produced no speech — the file may be silent "
                "or contain no intelligible audio."
            )

        windows = _window_segments(segments)

        # ── Visual track: OCR the on-screen text and align it to the audio ──
        screen   = extract_screen_text(file_path, doc_id)
        visuals  = screen["entries"]
        figures  = screen["figures"]

        orphans  = _attach_screen_text(windows, visuals)
        composed = [
            {"start": w["start"], "end": w["end"],
             "narration": w["text"], "screen": w.get("screen", []),
             "inherited": w.get("screen_inherited", False)}
            for w in windows
        ]
        composed += _window_visuals(orphans)
        composed.sort(key=lambda c: c["start"])

        if visuals or figures:
            _write_screen_timeline(filename, result, composed, figures)

        chunks: list[dict] = []
        n_screen_only = 0
        for i, c in enumerate(composed, start=1):
            has_narration = bool(c["narration"].strip())
            has_screen    = bool(c["screen"])
            inherited     = bool(c.get("inherited"))
            ctype = "transcript_segment" if has_narration else "screen_capture"
            if not has_narration:
                n_screen_only += 1
            chunks.append({
                "text": _segment_text(
                    filename, c["start"], c["end"],
                    _compose_body(c["narration"], c["screen"], inherited),
                ),
                "chunk_type": ctype,
                "page":       i,
                "_meta":      _segment_metadata(
                    filename, doc_id, i, c["start"], c["end"],
                    chunk_type=ctype, has_screen_text=has_screen,
                    screen_inherited=inherited,
                ),
            })

        next_index = len(composed) + 1
        for fig in figures:
            chunks.append({
                "text": _figure_text(filename, fig["start_sec"], fig["text"]),
                "chunk_type": "figure",
                "page":       next_index,
                "_meta":      _figure_metadata(
                    filename, doc_id, next_index,
                    fig["start_sec"], fig["image_path"],
                ),
            })
            next_index += 1

        s_text, s_meta = _summary_chunk(filename, doc_id, result, len(composed))
        chunks.append({
            "text": s_text, "chunk_type": "transcript_summary",
            "page": 0, "_meta": s_meta,
        })

        # PII scrub (mutates chunk["text"], adds pii_* keys).
        logger.info(f"Scrubbing PII for {len(chunks)} transcript chunks...")
        t0 = time.time()
        chunks = scrub_chunks_batch(chunks)
        logger.info(f"PII scrub done in {time.time() - t0:.1f}s.")

        documents, metadatas, ids = [], [], []
        blocked = 0
        for chunk in chunks:
            chunk = tag_sensitivity(chunk)
            if chunk["sensitivity_level"] == "high":
                blocked += 1
                continue
            meta = dict(chunk["_meta"])
            meta["sensitivity_level"] = chunk["sensitivity_level"]
            meta["pii_detected"]      = str(chunk.get("pii_detected", False))
            meta["pii_types_found"]   = chunk.get("pii_types_found", "")
            documents.append(chunk["text"])
            metadatas.append(meta)
            ids.append(f"{doc_id}_seg_{meta['segment_index']}")

        if not documents:
            raise ValueError("All transcript chunks were blocked by the sensitivity filter.")

        for b in range(0, len(documents), EMBED_BATCH_SIZE):
            sl = slice(b, b + EMBED_BATCH_SIZE)
            embeddings = _get_embeddings_batch(documents[sl])
            chroma_store.add_documents(
                collection_name=CHROMA_COLLECTION_VIDEO,
                documents=documents[sl],
                embeddings=embeddings,
                metadatas=metadatas[sl],
                ids=ids[sl],
            )

        update_status(doc_id, status="ready", chunk_count=len(documents))
        summary = {
            "doc_id":         doc_id,
            "filename":       filename,
            "collection":     CHROMA_COLLECTION_VIDEO,
            "record_type":    "video",
            "language":       result["language"],
            "duration_sec":   result["duration_sec"],
            "transcript_chunks":  len(composed) - n_screen_only,
            "screen_only_chunks": n_screen_only,
            "figure_chunks":      len(figures),
            "total_chunks":       len(documents),
            "blocked":            blocked,
            "status":             "ready",
        }
        logger.info(f"Video ingestion complete: {summary}")
        return summary

    except Exception as e:
        mark_failed(doc_id, reason=str(e))
        logger.error(f"Video ingestion failed for {filename}: {e}", exc_info=True)
        raise
