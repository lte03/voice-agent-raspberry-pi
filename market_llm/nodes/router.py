"""
Market LLM - Router Node
Intent detection and routing
"""

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..state import GraphState
from ..prompts import ROUTER_PROMPT
from ..config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS


_router_llm = None


def get_router_llm() -> ChatOpenAI:
    """Router LLM instance'ını döndür (lazy initialization)"""
    global _router_llm
    if _router_llm is None:
        _router_llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )
    return _router_llm


def route_intent(state: GraphState) -> GraphState:
    """
    Kullanıcının niyetini tespit et.
    
    Args:
        state: Mevcut graph state
    
    Returns:
        GraphState: Intent eklenmiş state
    """
    user_input = state["user_input"]
    llm = get_router_llm()
    
    messages = [
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=user_input)
    ]
    
    response = llm.invoke(messages)
    
    try:
        content = response.content.strip()
        
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        result = json.loads(content)
        state["intent"] = result.get("category", "database_query")
        
    except (json.JSONDecodeError, Exception) as e:
        print(f"[Router] Parse error: {e}")
        state["intent"] = "database_query"
    
    return state


def route_by_intent(state: GraphState) -> str:
    """
    Intent'e göre sonraki node'u belirle.
    
    Args:
        state: Mevcut graph state
    
    Returns:
        str: Sonraki node adı
    """
    intent = state.get("intent", "database_query")
    
    routing_map = {
        "off_topic": "off_topic_node",
        "greeting": "greeting_node",
        "farewell": "farewell_node"
    }
    
    return routing_map.get(intent, "sql_generator_node")
