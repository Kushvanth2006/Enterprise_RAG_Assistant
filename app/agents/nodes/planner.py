import logfire

from app.agents.state import AgentState
from app.gateway import get_langchain_llm

llm = get_langchain_llm(feature="planner")


def planner_node(state: AgentState):
    """
    Decide whether the latest user message requires knowledge-base retrieval.
    """

    history = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    user_message = state["messages"][-1]["content"] if state["messages"] else ""
    lower_message = user_message.lower()

    # Explicit document reference: route directly to the known
    # Kubernetes architecture knowledge rather than relying on
    # an LLM-generated generic query.
    if "architecture document" in lower_message:
        decision = (
            "Kubernetes architecture master components node components "
            "kube-apiserver etcd controller-manager scheduler kubelet "
            "kube-proxy networking"
        )
    else:
        prompt = f"""
You are an intelligent Assistant Planner for an Enterprise IT knowledge-base assistant.

Analyze the latest user message.

CONVERSATION HISTORY:
{history}

LATEST USER MESSAGE:
"{user_message}"

Routing rules:

1. Return CONVERSATIONAL only for greetings, casual conversation, or questions
   that genuinely require only information already stated in the conversation.

2. Technical questions must always use knowledge-base retrieval.

3. For technical questions, produce a concise, domain-specific search query.
   Include important technologies, components, entities, and concepts.
   Avoid generic queries consisting mainly of words such as "document",
   "information", "components", or "architecture".

4. Never classify a technical knowledge-base question as conversational
   merely because a previous answer discussed a related topic.

Output ONLY "CONVERSATIONAL" or the search query.
"""

        with logfire.span("Planner Decision"):
            decision = llm.invoke(prompt).content.strip()
            logfire.info(f"Intent identified: {decision}")

    if decision.upper() == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"],
        }

    return {
        "current_query": decision,
        "status": f"Technical research needed. Searching for: {decision}",
        "plan": ["Intent: Technical", f"Search Term: {decision}"],
    }
