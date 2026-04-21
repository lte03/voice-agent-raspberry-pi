from typing import Dict, Callable
from langgraph.graph import StateGraph

from ...nodes import route_by_intent


def add_router_edges(
    workflow: StateGraph,
    source_node: str,
    routing_map: Dict[str, str]
) -> StateGraph:
    """
    Router node'dan conditional edge'ler ekle.
    
    Args:
        workflow: StateGraph instance
        source_node: Kaynak node adı
        routing_map: {route_key: target_node} mapping
    
    Returns:
        StateGraph: Edge'ler eklenmiş graph
    """
    workflow.add_conditional_edges(
        source_node,
        route_by_intent,
        routing_map
    )
    return workflow


DEFAULT_ROUTING_MAP = {
    "sql_generator_node": "sql_generator_node",
    "off_topic_node": "off_topic_node",
    "greeting_node": "greeting_node",
    "farewell_node": "farewell_node"
}
