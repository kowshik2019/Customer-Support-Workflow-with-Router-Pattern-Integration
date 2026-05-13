"""
app.py — Streamlit UI for the Customer Support Agent.

Run with:
    streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

from workflow import run_ticket

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Customer Support Agent",
    page_icon="🛟",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🛟 Support Agent")
    st.markdown(
        "**Pipeline**\n\n"
        "1. Classify ticket\n"
        "2. Detect urgency\n"
        "3. **Router** → specialist handler\n"
        "4. Generate response + steps\n"
        "5. Escalation decision\n"
        "6. Close or escalate"
    )
    st.divider()
    st.caption("Categories handled:")
    st.markdown(
        "- 💳 Payment Issue\n"
        "- 🔐 Login Problem\n"
        "- ↩️ Refund Request\n"
        "- 🐞 Technical Bug\n"
        "- 💬 General Inquiry"
    )
    st.divider()
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY missing. Add it to your `.env`.")

    st.caption("Model: " + os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("AI Customer Support Agent")
st.caption("Submit a customer complaint. The LangGraph workflow will classify, prioritize, respond, and route it.")

# Example complaints
examples = {
    "💳 Payment failed but money deducted":
        "I tried to pay ₹2,499 for my order today. The website said 'Payment Failed' "
        "but the money has been deducted from my bank account. This is the second time "
        "this is happening. Please refund my money immediately!",
    "🔐 Can't log in":
        "I keep getting an 'invalid credentials' error even though I'm sure my password is right. "
        "I tried resetting it twice and the reset email never arrives.",
    "↩️ Refund stuck":
        "I requested a refund 10 days ago for order #88123 and I still haven't received it. "
        "Customer care said 5-7 days. Where is my money?",
    "🐞 App crashes":
        "Your Android app crashes every time I tap 'Checkout'. I've reinstalled twice. "
        "I'm on Android 14, app version 4.2.1.",
    "💬 General question":
        "Hi, do you ship internationally? I'm in Singapore and I'd like to place an order.",
}

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📝 Customer Complaint")
    example_pick = st.selectbox(
        "Load an example (optional):",
        options=["— none —"] + list(examples.keys()),
    )

    default_text = examples[example_pick] if example_pick != "— none —" else ""
    complaint = st.text_area(
        "Complaint text",
        value=default_text,
        height=220,
        placeholder="Paste or type the customer's complaint here...",
    )
    customer_id = st.text_input("Customer ID (optional)", value="CUS-1001")

    submit = st.button("🚀 Run support agent", type="primary", use_container_width=True)

with col_right:
    st.subheader("🎛️ Result")

    if submit:
        if not complaint.strip():
            st.warning("Please enter a complaint first.")
        elif not os.getenv("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY is missing. Set it in `.env` and restart the app.")
        else:
            with st.spinner("Routing through LangGraph workflow..."):
                try:
                    result = run_ticket(complaint, customer_id=customer_id or None)
                except Exception as e:
                    st.error(f"Workflow error: {e}")
                    st.stop()

            # ---- Top metrics ----
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Category", result.get("category", "—"))
            m2.metric("Urgency", result.get("urgency", "—"))
            handler = result.get("handler_used", "—").replace("_handler", "").title()
            m3.metric("Handler", handler)
            status = result.get("status", "—")
            m4.metric(
                "Status",
                status,
                delta="needs human" if status == "ESCALATED" else "auto-handled",
                delta_color=("inverse" if status == "ESCALATED" else "normal"),
            )

            # ---- Response card ----
            st.markdown("### ✉️ Suggested Response to Customer")
            st.info(result.get("response", "—"))

            # ---- Resolution steps ----
            st.markdown("### 🛠️ Suggested Resolution Steps")
            for i, step in enumerate(result.get("resolution_steps", []), 1):
                st.markdown(f"**{i}.** {step}")

            # ---- Escalation panel ----
            st.markdown("### 🚦 Routing Decision")
            if result.get("escalate"):
                st.error(f"Escalated → {result.get('escalation_reason', '')}")
            else:
                st.success(f"Auto-resolved → {result.get('escalation_reason', '')}")

            # ---- Diagnostics ----
            with st.expander("🔍 Workflow trace"):
                for line in result.get("trace", []):
                    st.write("•", line)

            with st.expander("🧾 Raw state (debug)"):
                st.json(result)
    else:
        st.info("Submit a complaint on the left to see the agent's output here.")
