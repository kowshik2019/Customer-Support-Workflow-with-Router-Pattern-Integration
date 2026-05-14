
🛟 AI Customer Support Agent — Router Pattern Integration to the Workflow

An intelligent, end-to-end customer-support automation agent built with LangGraph, OpenAI, and Streamlit. The agent ingests a free-text customer complaint, classifies it, scores its urgency, routes it to a category-specific specialist handler, drafts an empathetic response with a concrete resolution playbook, and decides whether to auto-resolve the ticket or escalate it to a human team.

This edition implements the Router Pattern — instead of a single generalist node handling every type of issue, the workflow branches after classification into one of five specialist handlers, each with its own domain knowledge and resolution logic.

---

📌 Project Description

Modern support inboxes drown in tickets that all look different — payment failures, login issues, refund delays, app crashes, general questions. Human agents spend most of their time reading, classifying, prioritizing, and writing similar replies before doing any actual problem-solving. This project automates that first 80% so humans only handle tickets that truly need their judgment.

What this agent does, for every incoming complaint:

1. Reads the complaint text and classifies it into one of five categories
2. Scores urgency on a four-level scale (LOW / MEDIUM / HIGH / CRITICAL)
3. Routes the ticket to a specialist handler that understands that category's domain
4. Generates a customer-facing response plus a 4–6 step resolution playbook
5. Applies rule-based escalation logic to either auto-resolve or hand off to a human team
6. Returns full state to the Streamlit UI for review

Key design principles

- Specialization over generalization — five focused handlers, each tuned with domain expertise (bank settlement windows for payments, lockout SLAs for logins, etc.)
- LLM-where-it-helps, code-where-it-matters — classification, urgency, and response generation use the LLM; routing and escalation decisions are pure Python rules so product/ops teams can change policy without prompt engineering
- Full observability — every node writes to a shared trace log, viewable in the UI
- Graceful degradation — every LLM output is validated; unknown values fall back to safe defaults
- Maintainability first — adding a new category means adding one handler function and one router entry; nothing else changes

---

🏗️ Architecture & Workflow

 Visual Workflow

```
                    ┌─→ payment_handler  ─┐
                    ├─→ login_handler    ─┤
classify → urgency ─┼─→ refund_handler   ─┼─→ decide → finalize → UI
                    ├─→ bug_handler      ─┤
                    └─→ general_handler  ─┘
                          ▲
                          │
                  Router function reads
                  state["category"] and
                  picks the next node
```

Step-by-Step Workflow

| Step | Node | What Happens | Powered By |
|------|------|--------------|------------|
| 1 | `classify` | Categorizes the complaint into one of 5 buckets; returns a confidence score | OpenAI (JSON mode) |
| 2 | `urgency` | Scores severity as LOW / MEDIUM / HIGH / CRITICAL with a reason | OpenAI (JSON mode) |
| 3 | `route_to_handler` | Pure-Python router function — reads category and returns the handler name | Python |
| 4 | One of: `payment_handler` / `login_handler` / `refund_handler` / `bug_handler` / `general_handler` | Specialist node generates response + resolution steps using domain-specific prompt | OpenAI (JSON mode) |
| 5 | `decide` | Applies escalation rules (urgency × category × confidence) | Python |
| 6 | `finalize` | Stamps terminal status: `RESOLVED` or `ESCALATED` | Python |
| 7 | UI render | Streamlit displays metrics, response, steps, routing, and trace | Streamlit |

Categories Handled

| Category | Specialist Handler | Domain Knowledge Embedded |
|---|---|---|
| 💳 Payment Issue | `payment_handler` | Bank settlement windows (5–7 days), UTR / transaction ID collection, chargeback escalation, duplicate-debit SLA |
| 🔐 Login Problem | `login_handler` | Lockout windows, password-reset SLA, 2FA recovery, account-takeover playbook (never asks for passwords) |
| ↩️ Refund Request | `refund_handler` | Standard 5–7 day refund SLA, breach escalation, partial refunds, goodwill gestures |
| 🐞 Technical Bug | `bug_handler` | Device/OS/app version collection, standard quick fixes, crash-report request, engineering review SLA |
| 💬 General Inquiry | `general_handler` | Warm catch-all; refuses to invent answers; safe fallback for anything that doesn't fit |

Escalation Rules

These live in `escalation_decision()` as plain Python — easy to tune without touching prompts:

```python
if urgency == "CRITICAL":                              # always escalate
    escalate = True
elif urgency == "HIGH" and category in financial:      # Payments team
    escalate = True
elif urgency == "HIGH" and category == "Technical Bug":# Engineering on-call
    escalate = True
elif urgency == "HIGH" and category == "Login Problem":# Identity & Security
    escalate = True
elif confidence < 0.45:                                # Human triage
    escalate = True
else:                                                  # Auto-resolve
    escalate = False
```

This is the maintainability seam — when your org adds a new team or changes routing policy, you change rules here without retraining anything.

---

📁 Project Structure

```
customer_support_agent/
├── workflow.py          # LangGraph state graph: nodes, router, edges
├── app.py               # Streamlit UI
├── requirements.txt     # Python dependencies
├── .env.example         # Template for environment variables
├── .env                 # YOUR keys (you create this; never commit it)
└── README.md            # This file
```

---

Prerequisites

Before you start, make sure you have:

1. Python 3.10 or newer installed
   - Check with: `python --version`
   - Download from https://www.python.org/downloads/ if needed
2. An OpenAI API key with available credits
   - Get one from https://platform.openai.com/api-keys
   - Verify billing is set up at https://platform.openai.com/settings/organization/billing
3. A code editor (Visual Studio Code recommended)
4. A terminal (PowerShell on Windows, Terminal on Mac/Linux, or VS Code's integrated terminal)

---

Step-by-Step Setup & Execution

Step 1 — Get the Project Files

Place all the files (`workflow.py`, `app.py`, `requirements.txt`, `.env.example`, `README.md`) into a folder named `customer_support_agent`.

Step 2 — Open a Terminal in the Project Folder

```bash
cd path/to/customer_support_agent
```

Confirm you're in the right place:

```bash
# Windows
dir

# Mac/Linux
ls
```

You should see `workflow.py`, `app.py`, etc.

Step 3 — (Recommended) Create a Virtual Environment

This keeps the project's dependencies isolated from your system Python.

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Windows (Command Prompt):
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

Mac/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

Your terminal prompt should now show `(.venv)` at the start.

Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs LangGraph, OpenAI SDK, Streamlit, python-dotenv, and their transitive dependencies. Takes 1–2 minutes the first time.

Step 5 — Configure Your OpenAI API Key

Copy the template:

Windows:
```cmd
copy .env.example .env
```

Mac/Linux:
```bash
cp .env.example .env
```

Open the new `.env` file in your editor and replace the placeholder:

```
OPENAI_API_KEY=sk-proj-your-real-key-here
OPENAI_MODEL=gpt-4o-mini
```

Save the file. Make sure:
- No quotes around the key
- No spaces around the `=` sign
- The whole key is on ONE line (no wraps)
- File is named exactly `.env` (not `.env.txt`)

> 💡 The project uses `load_dotenv(override=True)`, which means `.env` always wins over any leftover shell-level environment variables. Safe by default.

Step 6 — (Optional) Sanity-Check from CLI

Before launching the UI, you can verify the workflow runs end-to-end:

```bash
python workflow.py
```

You'll see the full JSON state for the built-in payment-deducted example printed to the console. If you see a JSON blob with `"status": "ESCALATED"` and `"handler_used": "payment_handler"`, everything works.

Step 7 — Launch the Streamlit UI

```bash
streamlit run app.py
```

Streamlit will print a URL like:

```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

Open that URL in your browser.

Step 8 — Test the Agent

In the UI:

1. Pick an example from the dropdown on the left (e.g., "💳 Payment failed but money deducted")
2. Click Run support agent
3. Watch the right-hand panel render:
   - Four top metrics: Category, Urgency, Handler, Status
   - A blue card with the suggested customer response
   - A numbered list of resolution steps
   - A red/green banner showing the routing decision
   - Expandable sections for the workflow trace and raw state

Try all five examples to see different handlers fire and different escalation paths.

Step 9 — Stop the App

In the terminal running Streamlit, press Ctrl+C.

To exit the virtual environment when you're done: /exit

Try These Test Scenarios

The Streamlit UI ships with example complaints, but feel free to type your own:

| Complaint | Expected Category | Expected Urgency | Expected Status |
|---|---|---|---|
| "Money got deducted but payment failed, second time today" | Payment Issue | HIGH | ESCALATED |
| "Password reset email never arrives, tried 3 times" | Login Problem | MEDIUM/HIGH | RESOLVED or ESCALATED |
| "Refund still not received after 10 days" | Refund Request | HIGH | ESCALATED |
| "App crashes on checkout on Android 14" | Technical Bug | HIGH | ESCALATED |
| "Do you ship internationally?" | General Inquiry | LOW | RESOLVED |
| "Help me understand my invoice" | General Inquiry | LOW | RESOLVED |

---

Configuration

All configuration lives in `.env`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key | Get your API key from open AI Keys Usage and navigate to API Key Create a new key and use it workflow in VS Code.
| `OPENAI_MODEL` | `gpt-4o-mini` | Any OpenAI chat-completions model. `gpt-4o-mini` is fast and cheap; `gpt-4o` is more capable for ambiguous tickets |

---

Extending the Project

The Router Pattern makes the project highly extensible. Common extensions:

Add a new category

1. Add it to the `CATEGORY` Literal in `workflow.py`
2. Update the classifier prompt's allowed list
3. Write a new specialist handler function (copy any existing one as a template)
4. Add an entry to `routing_map` inside `route_to_handler`
5. Register the new node and edge in `build_graph()`
6. (Optional) Add an escalation rule in `escalation_decision`

Add a clarifying-question loop

When the classifier's confidence is low, instead of immediately escalating, add a node that asks the customer for more details. This becomes a **second router** based on confidence.

Add real tools to handlers

- `payment_handler` could call a real Stripe API to look up the transaction
- `bug_handler` could create a Jira ticket
- `general_handler` could query a vector database of FAQ articles

Each handler is now an isolated unit, so adding tools to one doesn't touch the others.

Persist tickets

Wire `finalize` to also write the final state to a database (SQLite, Postgres) so you build an audit trail.

---

 Tech Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — state-graph framework that powers the workflow, including the conditional edges that implement the Router Pattern
- [OpenAI Python SDK](https://github.com/openai/openai-python) — calls `gpt-4o-mini` (or any chosen model) in strict JSON mode for reliable structured output
- [Streamlit](https://streamlit.io/) — instant Python-to-web UI; no HTML/CSS/JS required
- [python-dotenv](https://github.com/theskumar/python-dotenv) — loads secrets from `.env` so they stay out of code and out of git

