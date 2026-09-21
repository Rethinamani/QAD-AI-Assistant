# ingestion/transcriber.py

"""
Local speech-to-text for audio / video files.

ffmpeg pulls a 16 kHz mono WAV out of the media file; faster-whisper
(CTranslate2 Whisper port) transcribes it. Everything runs on the CPU with
no network calls, matching the rest of the pipeline.

The "medium" model with VAD filtering is used by default because it is the
only configuration that stays coherent on quiet / noisy source audio; set
WHISPER_MODEL=small (or base) for faster runs on clean recordings.
"""

import os
import json
import shutil
import logging
import subprocess

from config import (
    WHISPER_MODEL,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    TRANSCRIPT_OUTPUT_DIR,
)

logger = logging.getLogger(__name__)

# faster-whisper model — heavy to load, kept as a process-wide singleton.
_model = None


class TranscriptionError(Exception):
    """Raised when audio extraction or transcription fails."""


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        logger.info(
            f"Loading Whisper model '{WHISPER_MODEL}' "
            f"({WHISPER_DEVICE}/{WHISPER_COMPUTE_TYPE}) — first time only..."
        )
        _model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        logger.info("Whisper model ready.")
    return _model


def _ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise TranscriptionError(
            "ffmpeg not found on PATH. Install it with "
            "'winget install Gyan.FFmpeg' and restart the API."
        )
    return exe


def _extract_audio(media_path: str, wav_path: str) -> None:
    """Decode the media file to 16 kHz mono PCM WAV (what Whisper expects)."""
    cmd = [
        _ffmpeg_bin(), "-y",
        "-i", media_path,
        "-vn",              # drop any video stream
        "-ac", "1",         # mono
        "-ar", "16000",     # 16 kHz
        wav_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or b"").decode("utf-8", "ignore")[-800:]
        raise TranscriptionError(f"ffmpeg failed to extract audio: {stderr}")
    if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
        raise TranscriptionError("ffmpeg produced no audio — is there a sound track?")


def format_timestamp(seconds: float) -> str:
    """Seconds → 'M:SS' or 'H:MM:SS'."""
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _write_sidecars(base_name: str, result: dict) -> str:
    """Write .transcript.txt / .srt / .segments.json next to each other."""
    out_dir = os.path.join(TRANSCRIPT_OUTPUT_DIR, base_name)
    os.makedirs(out_dir, exist_ok=True)

    txt_path = os.path.join(out_dir, f"{base_name}.transcript.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result["text"] + "\n")

    srt_path = os.path.join(out_dir, f"{base_name}.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(result["segments"], start=1):
            f.write(
                f"{i}\n"
                f"{_srt_time(seg['start'])} --> {_srt_time(seg['end'])}\n"
                f"{seg['text'].strip()}\n\n"
            )

    json_path = os.path.join(out_dir, f"{base_name}.segments.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"Transcript sidecars written to {out_dir}")
    return out_dir


def _srt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    ms = int(round((seconds - int(seconds)) * 1000))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def transcribe_media(media_path: str, write_sidecars: bool = True) -> dict:
    """
    Transcribe an audio or video file.

    Returns:
        {
            "source_file":  str,
            "language":     str,
            "duration_sec": float,
            "model":        str,
            "text":         str,              full transcript
            "segments":     [ {start, end, text}, ... ],
        }

    Raises TranscriptionError on ffmpeg / whisper failure.
    """
    if not os.path.exists(media_path):
        raise TranscriptionError(f"File not found: {media_path}")

    filename  = os.path.basename(media_path)
    base_name = os.path.splitext(filename)[0]
    work_dir  = os.path.join(TRANSCRIPT_OUTPUT_DIR, base_name)
    os.makedirs(work_dir, exist_ok=True)
    wav_path  = os.path.join(work_dir, f"{base_name}.wav")

    logger.info(f"Transcribing {filename} ...")
    _extract_audio(media_path, wav_path)

    model = _get_model()
    try:
        segments_iter, info = model.transcribe(
            wav_path,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=700),
        )
        segments = [
            {
                "start": round(float(s.start), 2),
                "end":   round(float(s.end), 2),
                "text":  s.text.strip(),
            }
            for s in segments_iter
            if s.text and s.text.strip()
        ]
    except Exception as e:
        raise TranscriptionError(f"Whisper transcription failed: {e}")

    full_text = " ".join(s["text"] for s in segments).strip()

    result = {
        "source_file":  filename,
        "language":     getattr(info, "language", "unknown"),
        "duration_sec": round(float(getattr(info, "duration", 0.0)), 2),
        "model":        WHISPER_MODEL,
        "text":         full_text,
        "segments":     segments,
    }

    logger.info(
        f"Transcribed {filename}: {len(segments)} segments, "
        f"{len(full_text)} chars, language={result['language']}."
    )

    if write_sidecars:
        _write_sidecars(base_name, result)

    return result
