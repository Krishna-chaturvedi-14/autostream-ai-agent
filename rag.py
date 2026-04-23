import json
import os

def retrieve_context(query: str) -> str:
    """
    RAG retrieval function.
    Reads knowledge_base.json and returns all plans and policies formatted as a comprehensive context string.
    """
    # Load knowledge_base.json
    kb_path = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
    with open(kb_path, "r", encoding="utf-8") as f:
        kb = json.load(f)
        
    context_lines = ["---", "PLANS:"]
    
    for plan in kb.get('plans', []):
        features = " | ".join(plan.get('features', []))
        context_lines.append(f"- {plan.get('name', 'Unknown')}: {plan.get('price', 'N/A')} | {features}")
        
    context_lines.extend(["", "POLICIES:"])
    
    for policy in kb.get('policies', []):
        context_lines.append(f"- {policy}")
        
    context_lines.append("---")
        
    return "\n".join(context_lines)
