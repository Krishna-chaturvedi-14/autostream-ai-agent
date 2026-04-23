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

This project uses **LangGraph** as a structured state machine, enabling explicit node transitions and persistent multi-turn memory. LangGraph was chosen because the state machine approach makes conversational flows — like gathering lead details or routing to a knowledge responder — deliberate and verifiable, avoiding the black-box nature of standard LLM chains.

State is managed via an `AgentState` TypedDict passed through every node, mutated, and returned. It preserves the full conversation history (`messages`), the current `intent`, and incrementally gathered lead fields (`lead_name`, `lead_email`, `lead_platform`).

For **RAG**, the agent queries a local `knowledge_base.json` file. On each user query, keyword retrieval chunks the JSON data and injects the most relevant context directly into the LLM system prompt before inference — grounding responses without requiring a vector database.

The **LLM** used is Google Gemini 1.5 Flash, called via the `google-generativeai` SDK. Intent is classified by prompting the model to return one of three strict labels: `greeting`, `product_inquiry`, or `high_intent`.

Tool calling is strictly gated at the state machine level. The `capture_lead` node is only reachable via a conditional edge that checks all three fields (`lead_name`, `lead_email`, `lead_platform`) are populated. The mock tool never fires prematurely.

---

### WhatsApp Deployment via Webhooks

- Use Twilio's WhatsApp API or Meta's WhatsApp Business API.
- Set up a webhook endpoint using FastAPI or Flask that receives POST requests from WhatsApp.
- Each incoming message triggers the LangGraph agent, using the sender's phone number as the unique session ID.
- Persist per-user `AgentState` in an in-memory dictionary or Redis, keyed by phone number, to maintain multi-turn context across stateless HTTP requests.
- Send the agent's response back to the user via the WhatsApp API.
- Deploy the webhook server on Railway, Render, or any VPS with HTTPS enabled.
