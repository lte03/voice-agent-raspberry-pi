"""
Market LLM - Graph State Definition
"""

from typing import TypedDict, List, Optional


class GraphState(TypedDict):
    """LangGraph state yapısı"""
    user_input: str
    intent: str
    sql_query: Optional[str]
    db_result: Optional[dict]
    response: str
    conversation_history: List[dict]
    error: Optional[str]
