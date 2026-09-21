# chatbot/servicenow_handler.py

"""
Escalation → ServiceNow incident.

This is DEMO plumbing: no real ServiceNow call is made. It builds the
payload that would be POSTed and returns a simulated incident id so the
"escalate / raise a ticket" conversation flow works end to end.
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def create_incident_payload(
    query: str,
    session_id: str,
    conversation_history: list[dict] | None = None,
) -> dict:
    """Build the incident payload that would be sent to ServiceNow."""
    history_text = ""
    if conversation_history:
        lines = []
        for turn in conversation_history[-6:]:  # last ~3 exchanges
            role    = "User" if turn.get("role") == "user" else "Assistant"
            content = str(turn.get("content", ""))[:300]
            lines.append(f"{role}: {content}")
        history_text = "\n".join(lines)

    description = (
        f"User Query: {query}\n\n"
        f"The Support Assistant was unable to find a resolution in the "
        f"knowledge base.\n\n"
        f"Conversation context:\n{history_text}"
        if history_text
        else (
            f"User Query: {query}\n\n"
            f"The Support Assistant was unable to find a resolution in the "
            f"knowledge base."
        )
    )

    payload = {
        "short_description": f"Support: {query[:100]}",
        "description":       description,
        "category":          "ERP Support",
        "priority":          "3 - Moderate",
        "created_at":        datetime.utcnow().isoformat(),
        "session_id":        session_id,
    }
    logger.info(f"Incident payload built for session {session_id[:8]}")
    return payload


def submit_incident(payload: dict) -> dict:
    """
    Simulate submitting a ServiceNow incident.

    Production would POST to {SERVICENOW_URL}/api/now/table/incident with
    credentials; here we just mint a deterministic-looking id.
    """
    simulated_id = f"INC{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    logger.info(f"[DEMO] Incident created: {simulated_id}")
    return {
        "success":     True,
        "incident_id": simulated_id,
        "message": (
            f"✅ ServiceNow incident **{simulated_id}** has been created. "
            f"The support team will contact you shortly."
        ),
    }


def handle_escalation(
    user_response: str,
    query: str,
    session_id: str,
    conversation_history: list[dict] | None = None,
) -> dict:
    """
    Handle the user's yes/no reply to an escalation prompt.

    Returns {"action": "created"|"declined"|"unclear", "message": str,
             "incident": dict | None}.
    """
    response_lower = user_response.strip().lower()

    if any(w in response_lower for w in
           ["yes", "yeah", "sure", "ok", "okay", "please", "yep", "proceed"]):
        payload  = create_incident_payload(query, session_id, conversation_history)
        incident = submit_incident(payload)
        return {"action": "created", "message": incident["message"], "incident": incident}

    if any(w in response_lower for w in
           ["no", "nope", "cancel", "never mind", "nevermind", "don't"]):
        return {
            "action":   "declined",
            "message":  "No problem. Feel free to rephrase your question or ask something else.",
            "incident": None,
        }

    return {
        "action":   "unclear",
        "message":  "I didn't quite catch that. Please reply **yes** to create a ServiceNow incident or **no** to cancel.",
        "incident": None,
    }
