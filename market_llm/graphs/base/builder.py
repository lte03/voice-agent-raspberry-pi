from langgraph.graph import StateGraph
from ...state import GraphState


def create_state_graph() -> StateGraph:
    """
    Yeni bir StateGraph oluştur.
    
    Returns:
        StateGraph: Boş state graph
    """
    return StateGraph(GraphState)
