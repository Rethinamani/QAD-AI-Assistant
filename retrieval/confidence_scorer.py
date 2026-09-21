# retrieval/confidence_scorer.py

from config import CONFIDENCE_HIGH, CONFIDENCE_LOW


def get_confidence_band(confidence: float) -> str:
    """
    Map a normalized confidence score to a band label.

    > 0.75  → high   (answer confidently)
    0.4-0.75 → medium (answer with caveat)
    < 0.4   → low    (route to ServiceNow)
    """
    if confidence >= CONFIDENCE_HIGH:
        return "high"
    elif confidence >= CONFIDENCE_LOW:
        return "medium"
    else:
        return "low"


def score_results(results: list[dict]) -> dict:
    """
    Given a list of reranked results, return the overall
    confidence assessment for the query.

    Uses the top result's confidence as the primary signal.

    Returns:
    {
        "top_confidence":  float,
        "confidence_band": str,
        "should_answer":   bool,
        "should_escalate": bool,
        "result_count":    int,
    }
    """
    if not results:
        return {
            "top_confidence":  0.0,
            "confidence_band": "low",
            "should_answer":   False,
            "should_escalate": True,
            "result_count":    0,
        }

    top_confidence = results[0].get("confidence", 0.0)
    band           = get_confidence_band(top_confidence)

    return {
        "top_confidence":  top_confidence,
        "confidence_band": band,
        "should_answer":   band in ("high", "medium"),
        "should_escalate": band == "low",
        "result_count":    len(results),
    }


def format_source_citation(result: dict) -> str:
    """
    Format a source citation string from a result's metadata.
    Used in the chatbot response alongside the answer.
    """
    meta        = result.get("metadata", {})
    source_type = meta.get("source_type", "unknown")
    source_file = meta.get("source_file", "unknown")
    confidence  = result.get("confidence", 0.0)

    if source_type == "pdf":
        page    = meta.get("page", "?")
        section = meta.get("section", "?")
        return (
            f"📄 Source: {source_file} | "
            f"Page {page} | "
            f"Section: {section} | "
            f"Confidence: {confidence:.0%}"
        )

    elif source_type == "video":
        chunk_type = meta.get("chunk_type")
        if chunk_type == "transcript_summary":
            return (
                f"🎬 Source: {source_file} (transcript overview) | "
                f"Confidence: {confidence:.0%}"
            )
        if chunk_type == "figure":
            return (
                f"🎬 Source: {source_file} (screenshot at {meta.get('start_time', '?')}) | "
                f"Confidence: {confidence:.0%}"
            )
        span = f"{meta.get('start_time', '?')}–{meta.get('end_time', '?')}"
        if chunk_type == "screen_capture":
            return (
                f"🎬 Source: {source_file} (on-screen text) | {span} | "
                f"Confidence: {confidence:.0%}"
            )
        return (
            f"🎬 Source: {source_file} | {span} | "
            f"Confidence: {confidence:.0%}"
        )

    elif source_type in ("incident", "defect"):
        if meta.get("chunk_type") == "sheet_summary":
            return (
                f"📊 Source: {source_file} (sheet overview) | "
                f"Confidence: {confidence:.0%}"
            )
        row_key = meta.get("row_key") or f"row {meta.get('row_number', '?')}"
        return (
            f"📊 Source: {source_file} | "
            f"{row_key} | "
            f"Confidence: {confidence:.0%}"
        )

    return f"📁 Source: {source_file} | Confidence: {confidence:.0%}"