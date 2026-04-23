# AutoStream Agent

A complete LangGraph-powered conversational AI agent for the fictional SaaS company, AutoStream.

### How to Run
1. Clone the repo
2. run `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your Anthropic API key
4. run `python3 agent.py`

### Demo Video
[Watch the demo on Google Drive](https://drive.google.com/file/d/18MYCtIBgJns8zCccLlMLhcglj5voRvrO/view?usp=sharing)

### Architecture Explanation
This project utilizes **LangGraph** as a structured state machine, allowing explicit node transitions and maintaining a multi-turn memory without losing track of user context. LangGraph was chosen because the state machine approach ensures that conversational flows—like gathering lead details or routing to a general knowledge responder—are deliberate and verifiable, avoiding the "black box" nature of standard conversational chains.

The agent’s state is managed via an `AgentState` TypedDict, which is passed through every node in the graph, mutated, and returned. This state preserves the full conversation history (`messages`), the current `intent`, as well as incrementally gathered user details (`lead_name`, `lead_email`, `lead_platform`).

For **RAG**, the agent queries a local JSON `knowledge_base.json`. When a user asks a question, keyword retrieval maps their query against the chunked JSON data and strings together relevant text. This chunked information is directly injected as context into the LLM system prompt prior to inference, successfully grounding the responses without requiring a heavy vector database.

Finally, tool calling is strictly gated at the state machine level. The agent must fully populate the three specific state fields before traversing the conditional edge to `capture_lead`. Only then does the mock lead capture tool execute.

### WhatsApp Deployment via Webhooks
Deploying this conversational agent to WhatsApp is a straightforward process using Webhooks:
- Use Twilio's WhatsApp API or Meta's WhatsApp Business API directly.
- Set up a webhook endpoint (typically using frameworks like FastAPI or Flask in Python) that receives POST requests from WhatsApp.
- When an incoming message hits the webhook, trigger the LangGraph agent graph, maintaining the sender's phone number as the unique session ID.
- To persist multi-turn details across stateless webhooks, store the per-user `AgentState` in an in-memory dictionary or a persistent cache like Redis, keyed by the phone number.
- After processing through the LangGraph, send the agent's finalized assistant response back via the WhatsApp API to the user.
- The webhook server can be easily deployed on services like Railway, Render, or a standard VPS with HTTPS enabled.
