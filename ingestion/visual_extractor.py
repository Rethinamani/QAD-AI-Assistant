# ingestion/visual_extractor.py

"""
Screen-text extraction for uploaded videos (the "visual track").

A screen recording carries information the narrator never says out loud —
menu paths, field labels, error dialogs, program numbers. `transcriber.py`
throws the video stream away (`-vn`); this module recovers it.

Pipeline:
    1. ffmpeg samples frames on scene change, plus one at least every
       VIDEO_FRAME_MAX_INTERVAL_SEC seconds, and logs each frame's
       timestamp via the `showinfo` filter.
    2. Tesseract OCR (reused from ingestion.cleaner.image_ocr) reads the
       on-screen text off every frame. Every frame that carried real text
       is kept — the dense per-frame timeline lets the ingester pair
       narration with whatever was on screen at that moment, even during a
       long dwell on one screen.
    3. A handful of visually distinct, text-rich frames are persisted under
       FIGURES_DIR/<doc_id>/ so they can be shown next to answers.

Everything here is best-effort. If ffmpeg is missing, the file has no
video stream, or OCR is unavailable, `extract_screen_text` returns an
empty result and audio-only ingestion carries on unchanged.
"""

import os
import re
import shutil
import logging
import subprocess
from difflib import SequenceMatcher

from config import (
    FIGURES_DIR,
    TRANSCRIPT_OUTPUT_DIR,
    VIDEO_FRAME_OCR_ENABLED,
    VIDEO_SCENE_THRESHOLD,
    VIDEO_FRAME_MAX_INTERVAL_SEC,
    VIDEO_FRAME_MAX_COUNT,
    VIDEO_FRAME_SCALE_WIDTH,
    VIDEO_FRAME_DEDUP_RATIO,
    VIDEO_FIGURE_MAX_COUNT,
    VIDEO_KEEP_RAW_FRAMES,
    VIDEO_OCR_TESSERACT_CONFIG,
)
from ingestion.transcriber import _ffmpeg_bin, TranscriptionError
from ingestion.cleaner.image_ocr import extract_image_text

logger = logging.getLogger(__name__)

_PTS_RE = re.compile(r"pts_time:([0-9]+\.?[0-9]*)")
_WS_RE = re.compile(r"\s+")


_WORD_RE = re.compile(r"[A-Za-z]{3,}")


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", text or "").strip().lower()


def _denoise(text: str) -> str:
    """
    Drop OCR lines that are almost certainly toolbar/icon noise: a line is
    kept only if it contains a run of at least three letters (a real word)
    and is not mostly punctuation.
    """
    kept: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not _WORD_RE.search(line):
            continue
        letters = sum(c.isalnum() for c in line)
        if letters < len(line) * 0.5:
            continue
        kept.append(line)
    return "\n".join(kept)


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _extract_frames(media_path: str, frames_dir: str) -> list[tuple[str, float]]:
    """
    Run ffmpeg to sample frames on scene change (or every
    VIDEO_FRAME_MAX_INTERVAL_SEC seconds) into frames_dir.

    Returns [(frame_path, start_sec), ...] ordered by time. Empty list when
    the file has no usable video stream.
    """
    os.makedirs(frames_dir, exist_ok=True)

    select_expr = (
        f"eq(n\\,0)"
        f"+gt(scene\\,{VIDEO_SCENE_THRESHOLD})"
        f"+gte(t-prev_selected_t\\,{VIDEO_FRAME_MAX_INTERVAL_SEC})"
    )
    vf = (
        f"select='{select_expr}',showinfo,"
        f"scale='min({VIDEO_FRAME_SCALE_WIDTH}\\,iw)':-2"
    )
    pattern = os.path.join(frames_dir, "frame_%05d.jpg")
    cmd = [
        _ffmpeg_bin(), "-y",
        "-i", media_path,
        "-an",                       # no audio
        "-vf", vf,
        "-fps_mode", "vfr",          # emit only the frames `select` passed
        "-frames:v", str(VIDEO_FRAME_MAX_COUNT),
        "-q:v", "3",
        pattern,
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True)
    except Exception as e:
        logger.warning(f"Frame extraction failed to start: {e}")
        return []

    stderr = (proc.stderr or b"").decode("utf-8", "ignore")
    timestamps = [float(m) for m in _PTS_RE.findall(stderr)]

    frame_files = sorted(
        f for f in os.listdir(frames_dir)
        if f.startswith("frame_") and f.endswith(".jpg")
    )
    if not frame_files:
        # No video stream (audio-only upload) or ffmpeg produced nothing.
        if "Output file #0 does not contain any stream" not in stderr:
            logger.info("No frames sampled from video (audio-only or unreadable stream).")
        return []

    out: list[tuple[str, float]] = []
    for i, name in enumerate(frame_files):
        ts = timestamps[i] if i < len(timestamps) else float(i * VIDEO_FRAME_MAX_INTERVAL_SEC)
        out.append((os.path.join(frames_dir, name), round(ts, 2)))
    return out


def _ocr_frames(frames: list[tuple[str, float]]) -> list[dict]:
    """OCR every sampled frame; keep the ones that carried real text."""
    entries: list[dict] = []
    for path, ts in frames:
        text = _denoise(extract_image_text(path, config=VIDEO_OCR_TESSERACT_CONFIG))
        if len(text) < 15:
            continue
        entries.append({"start_sec": ts, "text": text, "frame_path": path})
    return entries


def _persist_figures(entries: list[dict], doc_id: str) -> list[dict]:
    """
    Copy the most text-rich, visually distinct frames into
    FIGURES_DIR/<doc_id>/ and return them as figure descriptors
    [{start_sec, text, image_path}].
    """
    if VIDEO_FIGURE_MAX_COUNT <= 0 or not entries:
        return []

    # Greedily take the wordiest frames, skipping any that read almost the
    # same as one already chosen (so a static screen isn't kept six times).
    chosen: list[dict] = []
    for e in sorted(entries, key=lambda x: len(x["text"]), reverse=True):
        if len(chosen) >= VIDEO_FIGURE_MAX_COUNT:
            break
        if any(_similar(e["text"], c["text"]) >= VIDEO_FRAME_DEDUP_RATIO for c in chosen):
            continue
        chosen.append(e)
    chosen.sort(key=lambda e: e["start_sec"])

    out_dir = os.path.join(FIGURES_DIR, doc_id)
    os.makedirs(out_dir, exist_ok=True)

    figures: list[dict] = []
    for k, e in enumerate(chosen):
        dest_name = f"frame_{k}.jpg"
        try:
            shutil.copyfile(e["frame_path"], os.path.join(out_dir, dest_name))
        except Exception as exc:
            logger.warning(f"Could not persist figure frame {k}: {exc}")
            continue
        figures.append({
            "start_sec":  e["start_sec"],
            "text":       e["text"],
            "image_path": f"{doc_id}/{dest_name}",
        })
    return figures


def extract_screen_text(media_path: str, doc_id: str) -> dict:
    """
    Extract on-screen text from a video file.

    Returns:
        {
            "entries": [ {start_sec, text}, ... ],   # OCR'd screen text, time-ordered
            "figures": [ {start_sec, text, image_path}, ... ],
        }

    Always returns a dict; both lists are empty when extraction is disabled,
    ffmpeg / OCR is unavailable, or the file has no video stream.
    """
    empty = {"entries": [], "figures": []}
    if not VIDEO_FRAME_OCR_ENABLED:
        return empty

    try:
        _ffmpeg_bin()
    except TranscriptionError:
        logger.warning("Screen-text extraction skipped — ffmpeg not on PATH.")
        return empty

    base = os.path.splitext(os.path.basename(media_path))[0]
    frames_dir = os.path.join(TRANSCRIPT_OUTPUT_DIR, base, "frames")

    try:
        frames = _extract_frames(media_path, frames_dir)
        if not frames:
            return empty

        entries = _ocr_frames(frames)
        logger.info(
            f"Screen text: {len(frames)} frames sampled, "
            f"{len(entries)} carried usable on-screen text."
        )
        if not entries:
            return empty

        figures = _persist_figures(entries, doc_id)
        return {
            "entries": [{"start_sec": e["start_sec"], "text": e["text"]} for e in entries],
            "figures": figures,
        }
    except Exception as e:
        logger.warning(f"Screen-text extraction failed for {media_path}: {e}", exc_info=True)
        return empty
    finally:
        if not VIDEO_KEEP_RAW_FRAMES:
            shutil.rmtree(frames_dir, ignore_errors=True)
