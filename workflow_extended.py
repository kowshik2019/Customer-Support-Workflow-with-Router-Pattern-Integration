"""
workflow.py
-----------
Customer Support Agent built with LangGraph + OpenAI.
Now uses the ROUTER PATTERN: after classification + urgency, a router
function dispatches to a category-specific handler node.

Architecture flow:
    User Complaint
        -> classify
        -> urgency
        -> router (conditional edge based on category)
              ├─ payment_handler
              ├─ login_handler
              ├─ refund_handler
              ├─ bug_handler
              └─ general_handler
        -> decide (escalation rules)
        -> finalize (RESOLVED / ESCALATED)

Why the Router Pattern?
    Each handler has its own domain knowledge, prompt, and resolution playbook.
    Adding/changing one category never affects the others.
"""

import os
import json
from typing import TypedDict, Literal, List, Optional, Tuple
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from openai import OpenAI

# ---------------------------------------------------------------------------
# 0. ENV + CLIENT
# ---------------------------------------------------------------------------
load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    print("[WARN] OPENAI_API_KEY not set. Set it in your .env file.")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ---------------------------------------------------------------------------
# 1. TYPES + STATE
# ---------------------------------------------------------------------------
CATEGORY = Literal[
    "Payment Issue",
    "Login Problem",
    "Refund Request",
    "Technical Bug",
    "General Inquiry",
]

URGENCY = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class TicketState(TypedDict, total=False):
    # input
    complaint: str
    customer_id: Optional[str]

    # produced by nodes
    category: CATEGORY
    category_confidence: float
    urgency: URGENCY
    urgency_reason: str
    response: str
    resolution_steps: List[str]
    handler_used: str            # NEW: which specialist handled it
    escalate: bool
    escalation_reason: str
    status: Literal["RESOLVED", "ESCALATED"]
    trace: List[str]


# ---------------------------------------------------------------------------
# 2. LLM HELPER
# ---------------------------------------------------------------------------
def _call_llm_json(system: str, user: str) -> dict:
    if client is None:
        raise RuntimeError("OPENAI_API_KEY missing. Add it to .env.")

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = resp.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}


def _append_trace(state: TicketState, msg: str) -> List[str]:
    trace = list(state.get("trace", []))
    trace.append(msg)
    return trace


def _parse_handler_output(result: dict, fallback_resp: str) -> Tuple[str, List[str]]:
    """Common parsing for every specialist handler's JSON output."""
    response_text = result.get("response", fallback_resp)
    steps = result.get("resolution_steps") or []
    if not isinstance(steps, list):
        steps = [str(steps)]
    steps = [str(s).strip() for s in steps if str(s).strip()]
    if not steps:
        steps = ["Acknowledge the issue", "Gather details", "Investigate", "Follow up with the customer"]
    return response_text, steps


# ---------------------------------------------------------------------------
# 3. NODE: CLASSIFICATION
# ---------------------------------------------------------------------------
def classify_ticket(state: TicketState) -> TicketState:
    system = (
        "You are a customer-support ticket classifier. "
        "Read the complaint and pick exactly ONE category from this list: "
        "['Payment Issue', 'Login Problem', 'Refund Request', 'Technical Bug', 'General Inquiry']. "
        "If nothing fits clearly, default to 'General Inquiry'. "
        "Return JSON: {\"category\": <one of the 5>, \"confidence\": <0.0-1.0>}."
    )
    user = f"Complaint:\n\"\"\"{state['complaint']}\"\"\""

    result = _call_llm_json(system, user)
    category = result.get("category", "General Inquiry")
    confidence = float(result.get("confidence", 0.5))

    valid = {
        "Payment Issue", "Login Problem", "Refund Request",
        "Technical Bug", "General Inquiry",
    }
    if category not in valid:
        category = "General Inquiry"
        confidence = 0.4

    return {
        **state,
        "category": category,
        "category_confidence": confidence,
        "trace": _append_trace(state, f"Classified as {category} (conf={confidence:.2f})"),
    }


# ---------------------------------------------------------------------------
# 4. NODE: URGENCY DETECTION
# ---------------------------------------------------------------------------
def detect_urgency(state: TicketState) -> TicketState:
    system = (
        "You assess support-ticket urgency. "
        "Return JSON: {\"urgency\": one of ['LOW','MEDIUM','HIGH','CRITICAL'], \"reason\": <short string>}.\n"
        "Rules:\n"
        "- CRITICAL: money lost / fraud / data breach / security risk.\n"
        "- HIGH: payment failed but money deducted, account locked, refund stuck > 7 days, blocker for paid user.\n"
        "- MEDIUM: feature broken with workaround, login intermittently failing, refund recently requested.\n"
        "- LOW: general question, how-to, feedback."
    )
    user = (
        f"Category: {state.get('category', 'General Inquiry')}\n"
        f"Complaint: \"\"\"{state['complaint']}\"\"\""
    )

    result = _call_llm_json(system, user)
    urgency = result.get("urgency", "LOW")
    reason = result.get("reason", "Default urgency assigned.")

    if urgency not in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
        urgency = "LOW"

    return {
        **state,
        "urgency": urgency,
        "urgency_reason": reason,
        "trace": _append_trace(state, f"Urgency = {urgency} ({reason})"),
    }


# ---------------------------------------------------------------------------
# 5. ROUTER FUNCTION  (the heart of the Router Pattern)
# ---------------------------------------------------------------------------
def route_to_handler(state: TicketState) -> str:
    """
    Read state['category'] and return the name of the handler node to dispatch to.
    LangGraph's add_conditional_edges uses this string to pick the next node.
    """
    category = state.get("category", "General Inquiry")
    routing_map = {
        "Payment Issue":   "payment_handler",
        "Login Problem":   "login_handler",
        "Refund Request":  "refund_handler",
        "Technical Bug":   "bug_handler",
        "General Inquiry": "general_handler",
    }
    return routing_map.get(category, "general_handler")


# ---------------------------------------------------------------------------
# 6. SPECIALIST HANDLER NODES
# ---------------------------------------------------------------------------
def payment_handler(state: TicketState) -> TicketState:
    """Specialist for Payment Issues — money deducted, double-charge, failed transactions."""
    system = (
        "You are the PAYMENTS specialist support agent. Domain context you MUST use:\n"
        "- Failed-payment debits are typically auto-reversed by the bank within 5-7 business days.\n"
        "- Always ask for: transaction reference / UTR / order ID, timestamp, last 4 digits of card or UPI ID.\n"
        "- If auto-reversal doesn't happen in 7 working days, offer to raise a formal chargeback dispute.\n"
        "- For duplicate charges, mention the duplicate-debit reversal SLA (3-5 business days).\n"
        "- For 'second time' or repeat incidents, acknowledge the pattern and escalate priority.\n"
        "Output JSON: {'response': 2-4 empathetic sentences, 'resolution_steps': 4-6 concrete steps}."
    )
    user = f"Urgency: {state.get('urgency')}\nComplaint: \"\"\"{state['complaint']}\"\"\""
    result = _call_llm_json(system, user)
    response_text, steps = _parse_handler_output(result, "We're investigating the payment issue right away.")

    return {
        **state,
        "response": response_text,
        "resolution_steps": steps,
        "handler_used": "payment_handler",
        "trace": _append_trace(state, "Routed to PAYMENT specialist handler"),
    }


def login_handler(state: TicketState) -> TicketState:
    """Specialist for Login / authentication / access issues."""
    system = (
        "You are the IDENTITY & ACCESS specialist support agent. Domain context you MUST use:\n"
        "- For password reset emails not arriving: check spam, verify the registered email, mention 5-10 minute delay.\n"
        "- For repeated lockouts: account may be auto-locked after 5 failed attempts; unlock window is 30 minutes.\n"
        "- Always offer: password reset link, magic-link login (if supported), 2FA bypass via verified email.\n"
        "- For suspected account takeover: ask user to immediately enable 2FA and review recent login activity.\n"
        "- Never ask for the user's password.\n"
        "Output JSON: {'response': 2-4 empathetic sentences, 'resolution_steps': 4-6 concrete steps}."
    )
    user = f"Urgency: {state.get('urgency')}\nComplaint: \"\"\"{state['complaint']}\"\"\""
    result = _call_llm_json(system, user)
    response_text, steps = _parse_handler_output(result, "Let's get you back into your account.")

    return {
        **state,
        "response": response_text,
        "resolution_steps": steps,
        "handler_used": "login_handler",
        "trace": _append_trace(state, "Routed to LOGIN specialist handler"),
    }


def refund_handler(state: TicketState) -> TicketState:
    """Specialist for refund requests, refund delays, and refund disputes."""
    system = (
        "You are the REFUNDS specialist support agent. Domain context you MUST use:\n"
        "- Standard refund SLA: 5-7 business days to original payment method.\n"
        "- For refunds beyond 7 days: apologize for delay, ask for order ID, commit to manual investigation within 24h.\n"
        "- For partial refunds: clarify what portion of the order is eligible.\n"
        "- Always set a clear expected timeline AND a follow-up channel (email update / ticket link).\n"
        "- Offer a goodwill gesture (store credit / coupon) when SLA has clearly been breached.\n"
        "Output JSON: {'response': 2-4 empathetic sentences, 'resolution_steps': 4-6 concrete steps}."
    )
    user = f"Urgency: {state.get('urgency')}\nComplaint: \"\"\"{state['complaint']}\"\"\""
    result = _call_llm_json(system, user)
    response_text, steps = _parse_handler_output(result, "Let me get your refund moving today.")

    return {
        **state,
        "response": response_text,
        "resolution_steps": steps,
        "handler_used": "refund_handler",
        "trace": _append_trace(state, "Routed to REFUND specialist handler"),
    }


def bug_handler(state: TicketState) -> TicketState:
    """Specialist for technical bugs, crashes, broken features."""
    system = (
        "You are the TECHNICAL SUPPORT specialist agent (Tier 1 engineering triage). Domain context:\n"
        "- Always gather: device + OS version, app version, exact steps to reproduce, screenshot/screen recording.\n"
        "- Standard quick fixes to suggest: clear cache, force-close & relaunch, update to latest app version, reinstall.\n"
        "- For crashes: ask if a crash report was offered (and ask the user to send it).\n"
        "- Set expectation: Engineering reviews logs within 1 business day; major bugs ship in next patch release.\n"
        "- Provide a workaround if any exists.\n"
        "Output JSON: {'response': 2-4 empathetic sentences, 'resolution_steps': 4-6 concrete steps}."
    )
    user = f"Urgency: {state.get('urgency')}\nComplaint: \"\"\"{state['complaint']}\"\"\""
    result = _call_llm_json(system, user)
    response_text, steps = _parse_handler_output(result, "Sorry about the bug — let's debug this together.")

    return {
        **state,
        "response": response_text,
        "resolution_steps": steps,
        "handler_used": "bug_handler",
        "trace": _append_trace(state, "Routed to BUG specialist handler"),
    }


def general_handler(state: TicketState) -> TicketState:
    """Catch-all specialist for general inquiries / FAQs / anything not fitting other buckets."""
    system = (
        "You are a GENERAL customer-support agent (catch-all). Domain context:\n"
        "- Treat the user warmly; answer the question directly if it's a simple how-to or factual query.\n"
        "- If you don't have the answer, never invent — say you'll connect them with the right team.\n"
        "- For shipping / pricing / hours / policy questions, give a brief answer + offer to share a link.\n"
        "Output JSON: {'response': 2-4 sentences, 'resolution_steps': 3-5 concrete next-actions}."
    )
    user = f"Urgency: {state.get('urgency')}\nComplaint: \"\"\"{state['complaint']}\"\"\""
    result = _call_llm_json(system, user)
    response_text, steps = _parse_handler_output(result, "Happy to help — let me look into that.")

    return {
        **state,
        "response": response_text,
        "resolution_steps": steps,
        "handler_used": "general_handler",
        "trace": _append_trace(state, "Routed to GENERAL specialist handler"),
    }


# ---------------------------------------------------------------------------
# 7. ESCALATION DECISION (pure-Python rule layer — unchanged)
# ---------------------------------------------------------------------------
def escalation_decision(state: TicketState) -> TicketState:
    urgency = state.get("urgency", "LOW")
    category = state.get("category", "General Inquiry")
    confidence = state.get("category_confidence", 1.0)

    escalate = False
    reason = "Auto-resolved by AI agent."

    if urgency == "CRITICAL":
        escalate = True
        reason = "Critical urgency — must reach a human agent immediately."
    elif urgency == "HIGH" and category in {"Payment Issue", "Refund Request"}:
        escalate = True
        reason = "High-urgency financial issue — routing to Payments team."
    elif urgency == "HIGH" and category == "Technical Bug":
        escalate = True
        reason = "High-urgency technical bug — routing to Engineering on-call."
    elif urgency == "HIGH" and category == "Login Problem":
        escalate = True
        reason = "High-urgency access issue — routing to Identity & Security team."
    elif confidence < 0.45:
        escalate = True
        reason = f"Low classifier confidence ({confidence:.2f}) — sending to human triage."

    return {
        **state,
        "escalate": escalate,
        "escalation_reason": reason,
        "trace": _append_trace(state, f"Escalate={escalate} :: {reason}"),
    }


# ---------------------------------------------------------------------------
# 8. CLOSE OR ESCALATE (terminal)
# ---------------------------------------------------------------------------
def close_or_escalate(state: TicketState) -> TicketState:
    status = "ESCALATED" if state.get("escalate") else "RESOLVED"
    return {
        **state,
        "status": status,
        "trace": _append_trace(state, f"Final status: {status}"),
    }


# ---------------------------------------------------------------------------
# 9. BUILD THE GRAPH (with the Router Pattern wired in)
# ---------------------------------------------------------------------------
def build_graph():
    g = StateGraph(TicketState)

    # core nodes
    g.add_node("classify", classify_ticket)
    g.add_node("urgency", detect_urgency)
    g.add_node("decide", escalation_decision)
    g.add_node("finalize", close_or_escalate)

    # specialist handler nodes (router targets)
    g.add_node("payment_handler", payment_handler)
    g.add_node("login_handler", login_handler)
    g.add_node("refund_handler", refund_handler)
    g.add_node("bug_handler", bug_handler)
    g.add_node("general_handler", general_handler)

    # linear edges before and after the branch
    g.set_entry_point("classify")
    g.add_edge("classify", "urgency")

    # === THE ROUTER ===
    # After urgency, conditionally branch to one of the 5 specialists.
    g.add_conditional_edges(
        "urgency",                    # source node
        route_to_handler,             # router function — returns a key
        {                             # mapping: key -> next node
            "payment_handler": "payment_handler",
            "login_handler":   "login_handler",
            "refund_handler":  "refund_handler",
            "bug_handler":     "bug_handler",
            "general_handler": "general_handler",
        },
    )

    # All specialists converge back to the escalation decision
    g.add_edge("payment_handler", "decide")
    g.add_edge("login_handler",   "decide")
    g.add_edge("refund_handler",  "decide")
    g.add_edge("bug_handler",     "decide")
    g.add_edge("general_handler", "decide")

    # decide -> finalize -> END
    g.add_edge("decide", "finalize")
    g.add_edge("finalize", END)

    return g.compile()


# Compiled graph singleton
SUPPORT_GRAPH = build_graph() if OPENAI_API_KEY else None


def run_ticket(complaint: str, customer_id: Optional[str] = None) -> TicketState:
    if SUPPORT_GRAPH is None:
        raise RuntimeError("Graph not initialized. Set OPENAI_API_KEY in .env and restart.")

    initial: TicketState = {
        "complaint": complaint,
        "customer_id": customer_id,
        "trace": [],
    }
    return SUPPORT_GRAPH.invoke(initial)


# ---------------------------------------------------------------------------
# 10. CLI sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = (
        "I tried to pay for my order today, the payment page said FAILED but "
        "Rs 2,499 was deducted from my bank account. This is the second time. "
        "Please refund immediately!"
    )
    out = run_ticket(sample, customer_id="CUS-123")
    print(json.dumps(out, indent=2))
