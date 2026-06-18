# chatbot/servicenow_handler.py

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def create_incident_payload(
    query: str,
    session_id: str,
    conversation_history: list[dict] = None,
) -> dict:
    """
    Build a ServiceNow incident payload from the user's query.
    In production this would call the ServiceNow REST API.
    For demo, returns the payload that would be sent.

    Returns:
    {
        "short_description": str,
        "description":       str,
        "category":          str,
        "priority":          str,
        "created_at":        str,
        "session_id":        str,
    }
    """
    # Build description from conversation history if available
    history_text = ""
    if conversation_history:
        lines = []
        for turn in conversation_history[-6:]:  # last 3 exchanges
            role    = "User" if turn["role"] == "user" else "Assistant"
            content = turn["content"][:300]
            lines.append(f"{role}: {content}")
        history_text = "\n".join(lines)

    description = (
        f"User Query: {query}\n\n"
        f"The QAD Support Assistant was unable to find a resolution "
        f"in the knowledge base.\n\n"
        f"Conversation context:\n{history_text}"
        if history_text
        else f"User Query: {query}\n\nThe QAD Support Assistant was unable "
             f"to find a resolution in the knowledge base."
    )

    payload = {
        "short_description": f"QAD Support: {query[:100]}",
        "description":       description,
        "category":          "QAD ERP Support",
        "priority":          "3 - Moderate",
        "created_at":        datetime.utcnow().isoformat(),
        "session_id":        session_id,
    }

    logger.info(f"Incident payload built for session: {session_id[:8]}")
    return payload


def submit_incident(payload: dict) -> dict:
    """
    Submit a ServiceNow incident.

    In production: calls ServiceNow REST API with credentials.
    For demo: simulates the response.

    Returns:
    {
        "success":     bool,
        "incident_id": str,
        "message":     str,
    }
    """
    # ── Demo mode — simulate ServiceNow response ───────────────────────────────
    # In production replace this with:
    # response = requests.post(
    #     f"{SERVICENOW_URL}/api/now/table/incident",
    #     auth=(SERVICENOW_USER, SERVICENOW_PASSWORD),
    #     json=payload,
    #     headers={"Content-Type": "application/json"},
    # )

    simulated_id = f"INC{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    logger.info(f"[DEMO] Incident created: {simulated_id}")

    return {
        "success":     True,
        "incident_id": simulated_id,
        "message":     (
            f"✅ ServiceNow incident **{simulated_id}** has been created. "
            f"The support team will contact you shortly."
        ),
    }


def handle_escalation(
    user_response: str,
    query: str,
    session_id: str,
    conversation_history: list[dict] = None,
) -> dict:
    """
    Handle user's response to the escalation prompt.

    Args:
        user_response:         User's reply ("yes" / "no")
        query:                 Original query that triggered escalation
        session_id:            Current session ID
        conversation_history:  Full conversation history

    Returns:
    {
        "action":   "created" | "declined" | "unclear",
        "message":  str,
        "incident": dict | None,
    }
    """
    response_lower = user_response.strip().lower()

    # User confirmed escalation
    if any(word in response_lower for word in ["yes", "yeah", "sure", "ok", "okay", "please", "yep"]):
        payload  = create_incident_payload(query, session_id, conversation_history)
        incident = submit_incident(payload)

        return {
            "action":   "created",
            "message":  incident["message"],
            "incident": incident,
        }

    # User declined escalation
    elif any(word in response_lower for word in ["no", "nope", "cancel", "never mind", "nevermind"]):
        return {
            "action":   "declined",
            "message":  "No problem. Feel free to rephrase your question or ask something else.",
            "incident": None,
        }

    # Unclear response
    else:
        return {
            "action":   "unclear",
            "message":  "I didn't quite catch that. Please reply **yes** to create a ServiceNow incident or **no** to continue.",
            "incident": None,
        }