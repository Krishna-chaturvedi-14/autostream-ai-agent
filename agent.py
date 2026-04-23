import os
from typing import TypedDict, List, Dict, Optional, Any
from dotenv import load_dotenv

from google import genai
from google.genai import types
from langgraph.graph import StateGraph, END

from rag import retrieve_context
from tools import mock_lead_capture

# Load environment variables
load_dotenv()

# Initialize Gemini client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL_NAME = "gemini-2.5-flash"

class AgentState(TypedDict):
    """State across the agent workflow."""
    messages: List[Dict[str, str]]
    intent: str
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_captured: bool

def classify_intent(state: AgentState) -> AgentState:
    """Classifies user intent into a predefined category."""
    # If the user is already in the lead gathering flow, keep that intent
    if state.get("intent") == "collecting_lead":
        return state
        
    messages = state.get("messages", [])
    if not messages:
        return state
        
    latest_msg = messages[-1]["content"]
    
    system_prompt = """You are an intent classifier. Classify the user's message into EXACTLY one of these three labels (greeting, product_inquiry, high_intent).

EXAMPLES:
User: "hello" -> Label: greeting
User: "what are your prices?" -> Label: product_inquiry
User: "sounds good. i want to try the pro plan for my youtube channel" -> Label: high_intent
User: "how do refunds work?" -> Label: product_inquiry
User: "cool what next" -> Label: high_intent
User: "let's sign up" -> Label: high_intent

Reply with ONLY the label. No explanation. No punctuation. Just one word."""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=latest_msg,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=10
        )
    )
    
    intent_text = response.text
    if intent_text is None:
        intent = "product_inquiry"
    else:
        intent = intent_text.strip().lower()
    
    valid_intents = ["greeting", "product_inquiry", "high_intent"]
    if intent not in valid_intents:
        # Default fallback
        intent = "product_inquiry"
        
    return {"intent": intent}

def handle_greeting(state: AgentState) -> AgentState:
    """Handles an initial greeting."""
    welcome_text = "Hello! Welcome to AutoStream, your automated video editing toolkit. How can I help you today?"
    messages = list(state.get("messages", []))
    messages.append({"role": "assistant", "content": welcome_text})
    
    return {"messages": messages}

def handle_rag(state: AgentState) -> AgentState:
    """Retrieves context and asks Claude to answer based on the knowledge base."""
    messages = list(state.get("messages", []))
    latest_msg = messages[-1]["content"]
    
    context = retrieve_context(latest_msg)
    
    system_prompt = f"""You are AutoStream's helpful sales assistant.
Answer only based on this knowledge:
{context}

If the knowledge doesn't contain the answer, say you don't know but offer to check with the team."""
    
    gemini_messages = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        gemini_messages.append({"role": role, "parts": [{"text": msg["content"]}]})
        
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=gemini_messages,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=300
        )
    )
    
    response_msg = response.text if response.text is not None else "I apologize, but I encountered an error processing your request. Could you rephrase?"
    messages.append({"role": "assistant", "content": response_msg})
    
    return {"messages": messages}

def handle_lead_collection(state: AgentState) -> AgentState:
    """Collects user details across multiple turns."""
    messages = list(state.get("messages", []))
    latest_msg = messages[-1]["content"]
    
    lead_name = state.get("lead_name")
    lead_email = state.get("lead_email")
    lead_platform = state.get("lead_platform")
    
    # If we are already mid-collection, treat the latest message as the value for the missing field
    if state.get("intent") == "collecting_lead":
        value = latest_msg.strip()
        if lead_name is None:
            lead_name = value
        elif lead_email is None:
            lead_email = value
        elif lead_platform is None:
            lead_platform = value
            
    # Now check what to ask next
    if lead_name is None:
        assistant_reply = "I'd love to help you get started! Let's get a few quick details. What is your name?"
    elif lead_email is None:
        assistant_reply = f"Thanks, {lead_name}! What's your best email address?"
    elif lead_platform is None:
        assistant_reply = "Got it. Finally, what platform do you primarily create content for (e.g., YouTube, TikTok)?"
    else:
        # All collected, routing handles the next step
        assistant_reply = "Perfect! Saving your details now..."
        
    messages.append({"role": "assistant", "content": assistant_reply})
    
    return {
        "intent": "collecting_lead",
        "lead_name": lead_name,
        "lead_email": lead_email,
        "lead_platform": lead_platform,
        "messages": messages
    }

def capture_lead(state: AgentState) -> AgentState:
    """Saves the lead info using our mock tool."""
    name = state.get("lead_name", "")
    email = state.get("lead_email", "")
    platform = state.get("lead_platform", "")
    
    # Tool call
    tool_msg = mock_lead_capture(name, email, platform)
    
    messages = list(state.get("messages", []))
    messages.append({"role": "assistant", "content": f"All set! {tool_msg}. We'll be in touch."})
    
    return {
        "lead_captured": True,
        "intent": "done",
        "messages": messages
    }

def router(state: AgentState) -> str:
    """Routes based on intent to the appropriate downstream node."""
    intent = state.get("intent")
    if intent == "greeting":
        return "handle_greeting"
    elif intent == "product_inquiry":
        return "handle_rag"
    elif intent == "high_intent":
        return "handle_lead_collection"
    elif intent == "collecting_lead":
        # Check if we should move to capture
        if state.get("lead_name") and state.get("lead_email") and state.get("lead_platform"):
            return "capture_lead"
        return "handle_lead_collection"
    
    # Default fallback
    return "handle_rag"

def route_after_collection(state: AgentState) -> str:
    """Check whether we finished collecting the lead to proceed to capture."""
    if state.get("lead_name") and state.get("lead_email") and state.get("lead_platform"):
        return "capture_lead"
    return END

def build_graph():
    """Compiles the LangGraph state machine."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("handle_greeting", handle_greeting)
    workflow.add_node("handle_rag", handle_rag)
    workflow.add_node("handle_lead_collection", handle_lead_collection)
    workflow.add_node("capture_lead", capture_lead)
    
    workflow.set_entry_point("classify_intent")
    
    # Branching from intent classification
    workflow.add_conditional_edges(
        "classify_intent",
        router,
        {
            "handle_greeting": "handle_greeting",
            "handle_rag": "handle_rag",
            "handle_lead_collection": "handle_lead_collection",
            "capture_lead": "capture_lead"
        }
    )
    
    # From lead collection, go to capture if done, otherwise wait for more input (END turn)
    workflow.add_conditional_edges(
        "handle_lead_collection",
        route_after_collection,
        {
            "capture_lead": "capture_lead",
            END: END
        }
    )
    
    workflow.add_edge("handle_greeting", END)
    workflow.add_edge("handle_rag", END)
    workflow.add_edge("capture_lead", END)
    
    return workflow.compile()

if __name__ == "__main__":
    app_graph = build_graph()
    
    # Initialize the persistence of the state object manually since we're using a CLI loop
    state = {
        "messages": [],
        "intent": "unknown",
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_captured": False
    }
    
    print("Welcome to the AutoStream AI Agent. (Type 'exit' to quit)\n")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.strip().lower() == "exit":
                print("Exiting...")
                break
                
            state["messages"].append({"role": "user", "content": user_input})
            
            # Invoke the graph with the current state payload
            final_state = app_graph.invoke(state)
            
            # Persist state variables by overwriting the initial object with the returned state 
            # (Note: In standard un-annotated TypedDicts for langgraph, keys returned by nodes overwrite previous state).
            # The invoke output contains all state keys that were returned from nodes.
            # Using update on final state merges any new keys, but wait, invoke() returns the complete state.
            state = final_state
                
            assistant_msg = state["messages"][-1]["content"]
            print(f"AutoStream Agent: {assistant_msg}\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
