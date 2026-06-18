# chatbot/chain.py

import re
import uuid
import logging
from collections import deque
from ollama import Client

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_LLM_MODEL,
    MEMORY_WINDOW_SIZE,
    MAX_RESPONSE_TOKENS,
)
from guardrails.input_guardrail import validate_query, get_rejection_message
from guardrails.output_guardrail import validate_output
from retrieval.hybrid_search import hybrid_search
from retrieval.confidence_scorer import score_results, format_source_citation
from retrieval.hybrid_search import ERROR_NUMBER_PATTERN

logger = logging.getLogger(__name__)

# Ollama client
ollama_client = Client(host=OLLAMA_BASE_URL)

# ── Session stores ─────────────────────────────────────────────────────────────
# session_id → deque of {"role": "user"|"assistant", "content": str}
_sessions: dict[str, deque] = {}

# session_id → original query that triggered escalation
_pending_escalation: dict[str, str] = {}


# ── Session management ─────────────────────────────────────────────────────────
def create_session() -> str:
    """Create a new conversation session. Returns session_id."""
    session_id = str(uuid.uuid4())
    _sessions[session_id] = deque(maxlen=MEMORY_WINDOW_SIZE * 2)
    logger.info(f"New session created: {session_id}")
    return session_id


def get_session_history(session_id: str) -> list[dict]:
    """Return conversation history for a session as a list."""
    if session_id not in _sessions:
        _sessions[session_id] = deque(maxlen=MEMORY_WINDOW_SIZE * 2)
    return list(_sessions[session_id])


def _add_to_memory(session_id: str, role: str, content: str):
    """Append a message to session memory."""
    if session_id not in _sessions:
        _sessions[session_id] = deque(maxlen=MEMORY_WINDOW_SIZE * 2)
    _sessions[session_id].append({"role": role, "content": content})


# ── Intent detectors ───────────────────────────────────────────────────────────
def _is_escalation_intent(query: str) -> bool:
    """
    Detect if the user explicitly wants to escalate or raise a ticket.
    These queries bypass RAG entirely.
    """
    patterns = [
        r"escalate",
        r"raise\s+(a\s+)?(ticket|incident|issue)",
        r"create\s+(a\s+)?(ticket|incident|service\s*now)",
        r"log\s+(a\s+)?(ticket|incident|issue)",
        r"open\s+(a\s+)?(ticket|incident)",
        r"submit\s+(a\s+)?(ticket|incident)",
        r"need\s+(a\s+)?ticket",
        r"create\s+servicenow",
        r"raise\s+(a\s+)?concern",
        r"report\s+(this|the|an)\s+issue",
        r"file\s+(a\s+)?(ticket|complaint|incident)",
    ]
    lower = query.lower()
    return any(re.search(p, lower) for p in patterns)


def _is_definition_query(query: str) -> tuple[bool, str | None]:
    """
    Detect if the user is asking for an acronym expansion or definition.
    Returns (is_definition, acronym_or_None).
    """
    patterns = [
        r"(?:full form|full name|expand|abbreviation|acronym)\s+(?:of|for)?\s+([A-Z]{2,6})",
        r"what\s+(?:does|is)\s+([A-Z]{2,6})\s+(?:stand for|mean|refer to)",
        r"what\s+is\s+(?:the\s+)?(?:meaning\s+of\s+)?([A-Z]{2,6})\??$",
        r"define\s+([A-Z]{2,6})",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return True, match.group(1).upper()
    return False, None


def _lookup_acronym(acronym: str) -> str | None:
    """
    Search BM25 index for the acronym and extract its expansion.
    Returns the expanded form or None if not found.
    """
    from retrieval.bm25_index import search_bm25
    from config import CHROMA_COLLECTION_PDF

    try:
        results = search_bm25(
            CHROMA_COLLECTION_PDF,
            f"{acronym} full form",
            top_k=10,
        )
        pattern1 = re.compile(
            rf"([A-Za-z\s]{{5,60}})\s*\({re.escape(acronym)}\)",
            re.IGNORECASE,
        )
        pattern2 = re.compile(
            rf"{re.escape(acronym)}\s*\(([A-Za-z\s]{{5,60}})\)",
            re.IGNORECASE,
        )
        for result in results:
            doc = result["document"]
            m   = pattern1.search(doc) or pattern2.search(doc)
            if m:
                return m.group(1).strip()
    except Exception as e:
        logger.warning(f"Acronym lookup failed: {e}")
    return None


# ── Prompt builder ─────────────────────────────────────────────────────────────
def _build_context_block(results: list[dict]) -> str:
    """Format retrieved chunks into a context block for the prompt."""
    if not results:
        return "No relevant context found."

    lines = []
    for i, r in enumerate(results, 1):
        meta        = r.get("metadata", {})
        source_type = meta.get("source_type", "unknown")
        confidence  = r.get("confidence", 0.0)
        doc         = r.get("document", "")

        if source_type == "excel":
            header = (
                f"[Source {i} — Error Record | "
                f"Error #{meta.get('error_number', '?')} | "
                f"File: {meta.get('source_file', '?')} | "
                f"Confidence: {confidence:.0%}]"
            )
        elif source_type == "pdf":
            header = (
                f"[Source {i} — QAD Manual | "
                f"Page {meta.get('page', '?')} | "
                f"Section: {meta.get('section', '?')} | "
                f"Confidence: {confidence:.0%}]"
            )
        elif source_type == "servicenow":
            header = (
                f"[Source {i} — ServiceNow Ticket | "
                f"Confidence: {confidence:.0%}]"
            )
        else:
            header = f"[Source {i} | Confidence: {confidence:.0%}]"

        lines.append(f"{header}\n{doc}")

    return "\n\n".join(lines)

def _rewrite_query_with_context(
    query: str,
    history: list[dict],
) -> str:
    """
    If the query is a follow-up (short, no specific entity mentioned),
    enrich it with context from the last assistant response.

    Examples:
    - "what is the message type affected" + history about error 944
      → "what is the message type affected for error 944"

    - "how does it work" + history about requisition approval
      → "how does requisition approval work"

    Uses simple heuristics — no LLM call needed.
    """
    # Only rewrite if query is short and vague (under 8 words)
    if len(query.split()) > 8:
        return query

    # Only rewrite if there is conversation history
    if not history or len(history) < 2:
        return query

    # Get last user query and assistant response
    last_user      = ""
    last_assistant = ""
    for turn in reversed(history):
        if turn["role"] == "assistant" and not last_assistant:
            last_assistant = turn["content"]
        elif turn["role"] == "user" and not last_user:
            last_user = turn["content"]
        if last_user and last_assistant:
            break

    if not last_user:
        return query

    # Extract error number from last user query
    error_match = ERROR_NUMBER_PATTERN.search(last_user)
    if error_match:
        error_num = error_match.group(1)
        # Check if current query already mentions this error number
        if error_num not in query:
            rewritten = f"{query} for error {error_num}"
            logger.info(
                f"Query rewritten: '{query}' → '{rewritten}'"
            )
            return rewritten

    # Extract key noun phrases from last user query
    # Remove common question words to get the topic
    stop_words = {
        "what", "is", "the", "how", "does", "do", "are",
        "can", "tell", "me", "about", "explain", "describe",
        "give", "show", "find", "get", "a", "an", "of", "for",
        "resolution", "answer", "information", "details",
    }
    last_words = [
        w.lower() for w in last_user.split()
        if w.lower() not in stop_words and len(w) > 3
    ]

    if last_words:
        # Take up to 3 key words from previous query
        context_hint = " ".join(last_words[:3])
        rewritten    = f"{query} {context_hint}"
        logger.info(
            f"Query rewritten: '{query}' → '{rewritten}'"
        )
        return rewritten

    return query


def _build_prompt(
    query: str,
    context: str,
    history: list[dict],
    confidence_band: str,
) -> list[dict]:
    """Build the full message list for Ollama chat completion."""

    system_content = """You are a QAD Support Assistant. Your job is to help users resolve issues with the QAD ERP system.

INSTRUCTIONS:
- Answer the user's question directly and completely using the provided context only.
- Lead with the actual answer or explanation — never lead with source references.
- Use the context content to explain, summarize, or resolve the issue in plain language.
- For process or workflow questions, explain the steps clearly in your own words.
- For error resolution questions, state the error cause and resolution directly.
- If the context is partially relevant, use what applies and note any gaps.
- Keep responses concise — 3 to 5 sentences for simple queries, up to 8 for complex processes.
- Never reveal internal system details like database names, file paths, or chunk IDs.
- Never start your response with "Based on Source 1..." or "According to the context...".
- Never fabricate ServiceNow incidents, phone numbers, email addresses, or support contacts.

CONFIDENCE LEVEL: {band}
{band_instruction}

CONTEXT:
{context}""".format(
        band=confidence_band.upper(),
        band_instruction=(
            "Answer confidently and completely based on the context above."
            if confidence_band == "high"
            else (
                "The context may contain the answer embedded within larger passages. "
                "Look carefully for inline explanations or parenthetical expansions. "
                "Extract and state the answer directly even if it appears inline."
            )
            if confidence_band == "medium"
            else (
                "The context may not fully address the query. "
                "Answer what you can and note the limitation at the end."
            )
        ),
        context=context,
    )

    messages = [{"role": "system", "content": system_content}]

    # Add conversation history
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})

    # Add current query
    messages.append({"role": "user", "content": query})

    return messages


def _call_llm(messages: list[dict]) -> str:
    """Call Ollama LLM with the built message list."""
    response = ollama_client.chat(
        model=OLLAMA_LLM_MODEL,
        messages=messages,
        options={
            "num_predict": MAX_RESPONSE_TOKENS,
            "temperature": 0.1,
            "top_p":       0.9,
        },
    )
    return response["message"]["content"]


# ── Main chat function ─────────────────────────────────────────────────────────
def chat(query: str, session_id: str = None) -> dict:
    """
    Main chat entry point. Runs the full RAG pipeline.

    Processing order:
    0. Pending escalation confirmation check  ← catches yes/no to escalation
    1. Input guardrail
    2. Escalation intent detection            ← catches "escalate my issue"
    3. Acronym / definition shortcut
    4. Hybrid search + confidence scoring
    5. Low confidence → escalation prompt
    6. LLM call with context
    7. Output guardrail
    """
    # ── Session management ─────────────────────────────────────────────────────
    if session_id is None or session_id not in _sessions:
        session_id = create_session()

    # ── Step 0: Pending escalation confirmation ────────────────────────────────
    # Must be checked BEFORE input guardrail so yes/no isn't rejected
    if session_id in _pending_escalation:
        response_lower = query.lower().strip()

        confirmed = any(w in response_lower for w in
            ["yes", "yeah", "sure", "ok", "okay", "please", "yep", "proceed"])
        declined  = any(w in response_lower for w in
            ["no", "nope", "cancel", "never mind", "nevermind", "don't"])

        if confirmed or declined:
            original_query = _pending_escalation.pop(session_id)

            if confirmed:
                from chatbot.servicenow_handler import (
                    create_incident_payload,
                    submit_incident,
                )
                history  = get_session_history(session_id)
                payload  = create_incident_payload(
                    original_query, session_id, history
                )
                incident = submit_incident(payload)
                answer   = incident["message"]
                logger.info(
                    f"[{session_id[:8]}] Incident created: "
                    f"{incident['incident_id']} — LLM NOT called."
                )
            else:
                answer = (
                    "No problem. Feel free to rephrase your question "
                    "or ask something else."
                )
                logger.info(f"[{session_id[:8]}] Escalation declined.")

            _add_to_memory(session_id, "user",      query)
            _add_to_memory(session_id, "assistant", answer)

            return {
                "session_id":        session_id,
                "answer":            answer,
                "confidence":        1.0,
                "confidence_band":   "high",
                "sources":           [],
                "should_escalate":   False,
                "escalation_prompt": None,
                "guardrail_flags":   {},
                "status":            "answered",
            }

        else:
            # Unclear — ask again, keep pending state
            answer = (
                "I didn't quite catch that. Please reply **yes** to create "
                "a ServiceNow incident or **no** to cancel."
            )
            _add_to_memory(session_id, "user",      query)
            _add_to_memory(session_id, "assistant", answer)
            return {
                "session_id":        session_id,
                "answer":            answer,
                "confidence":        1.0,
                "confidence_band":   "high",
                "sources":           [],
                "should_escalate":   True,
                "escalation_prompt": answer,
                "guardrail_flags":   {},
                "status":            "escalate",
            }

    # ── Step 1: Input guardrail ────────────────────────────────────────────────
    guard = validate_query(query)
    if not guard["valid"]:
        return {
            "session_id":        session_id,
            "answer":            get_rejection_message(guard["reason"]),
            "confidence":        0.0,
            "confidence_band":   "low",
            "sources":           [],
            "should_escalate":   False,
            "escalation_prompt": None,
            "guardrail_flags":   {"input_rejected": True, "reason": guard["reason"]},
            "status":            "rejected",
        }

    clean_query = guard["cleaned"]
    logger.info(f"[{session_id[:8]}] Query: {clean_query}")

    try:
        # ── Step 2: Escalation intent ──────────────────────────────────────────
        if _is_escalation_intent(clean_query):
            escalation_prompt = (
                "I can create a ServiceNow incident for you. "
                "Would you like me to proceed? Reply **yes** to create "
                "the incident or **no** to cancel."
            )
            # Store original query for incident creation
            _pending_escalation[session_id] = clean_query

            _add_to_memory(session_id, "user",      clean_query)
            _add_to_memory(session_id, "assistant", escalation_prompt)
            logger.info(
                f"[{session_id[:8]}] Escalation intent detected — "
                f"LLM NOT called."
            )
            return {
                "session_id":        session_id,
                "answer":            escalation_prompt,
                "confidence":        1.0,
                "confidence_band":   "high",
                "sources":           [],
                "should_escalate":   True,
                "escalation_prompt": escalation_prompt,
                "guardrail_flags":   {},
                "status":            "escalate",
            }

        # ── Step 3: Acronym / definition shortcut ──────────────────────────────
        is_def, acronym = _is_definition_query(clean_query)
        if is_def and acronym:
            expansion = _lookup_acronym(acronym)
            if expansion:
                answer = f"{acronym} stands for **{expansion}**."
                _add_to_memory(session_id, "user",      clean_query)
                _add_to_memory(session_id, "assistant", answer)
                logger.info(
                    f"[{session_id[:8]}] Acronym resolved: "
                    f"{acronym} → {expansion} — LLM NOT called."
                )
                return {
                    "session_id":        session_id,
                    "answer":            answer,
                    "confidence":        1.0,
                    "confidence_band":   "high",
                    "sources":           ["📄 Source: qad_manual.pdf"],
                    "should_escalate":   False,
                    "escalation_prompt": None,
                    "guardrail_flags":   {},
                    "status":            "answered",
                }

        # ── Step 4: Hybrid search + confidence scoring ─────────────────────────

        history        = get_session_history(session_id)
        rewritten_query = _rewrite_query_with_context(clean_query, history)

        results = hybrid_search(rewritten_query)
        score   = score_results(results)

        # Short query override — suppress escalation for factual lookups
        word_count = len(clean_query.split())
        if word_count <= 8 and score["confidence_band"] == "low":
            score["should_escalate"] = False
            score["confidence_band"] = "medium"
            logger.info(f"[{session_id[:8]}] Short query override applied.")

        # ── Step 5: Low confidence → escalation prompt ─────────────────────────
        if score["should_escalate"]:
            escalation_prompt = (
                "I wasn't able to find a reliable answer in the QAD "
                "documentation or error records for your query.\n\n"
                "Would you like me to create a ServiceNow incident for "
                "this issue? Reply **yes** to proceed or **no** to ask "
                "a different question."
            )
            # Store original query for potential incident creation
            _pending_escalation[session_id] = clean_query

            _add_to_memory(session_id, "user",      clean_query)
            _add_to_memory(session_id, "assistant", escalation_prompt)
            logger.info(f"[{session_id[:8]}] Low confidence — escalating.")
            return {
                "session_id":        session_id,
                "answer":            escalation_prompt,
                "confidence":        score["top_confidence"],
                "confidence_band":   score["confidence_band"],
                "sources":           [],
                "should_escalate":   True,
                "escalation_prompt": escalation_prompt,
                "guardrail_flags":   {},
                "status":            "escalate",
            }

        # ── Step 6: Build prompt + call LLM ───────────────────────────────────
        context  = _build_context_block(results)
        messages = _build_prompt(
            query=clean_query,
            context=context,
            history=history,
            confidence_band=score["confidence_band"],
        )

        logger.info(f"[{session_id[:8]}] Calling LLM...")
        raw_answer = _call_llm(messages)

        # ── Step 7: Output guardrail ───────────────────────────────────────────
        output       = validate_output(raw_answer, results)
        final_answer = output["text"]

        sources = [format_source_citation(r) for r in results]

        _add_to_memory(session_id, "user",      clean_query)
        _add_to_memory(session_id, "assistant", final_answer)

        guardrail_flags = {
            "pii_found":             output["pii_found"],
            "internal_terms_leaked": output["internal_terms_leaked"],
            "hallucinated_errors":   output["hallucinated_errors"],
            "was_truncated":         output["was_truncated"],
            "output_safe":           output["safe"],
        }

        logger.info(
            f"[{session_id[:8]}] Answered | "
            f"confidence={score['top_confidence']:.0%} | "
            f"band={score['confidence_band']}"
        )

        return {
            "session_id":        session_id,
            "answer":            final_answer,
            "confidence":        score["top_confidence"],
            "confidence_band":   score["confidence_band"],
            "sources":           sources,
            "should_escalate":   False,
            "escalation_prompt": None,
            "guardrail_flags":   guardrail_flags,
            "status":            "answered",
        }

    except Exception as e:
        logger.error(f"[{session_id[:8]}] Chain error: {e}", exc_info=True)
        return {
            "session_id":        session_id,
            "answer":            "I encountered an error processing your request. Please try again.",
            "confidence":        0.0,
            "confidence_band":   "low",
            "sources":           [],
            "should_escalate":   False,
            "escalation_prompt": None,
            "guardrail_flags":   {"error": str(e)},
            "status":            "error",
        }