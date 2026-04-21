from langgraph.graph import END

from .base import create_state_graph, add_router_edges, DEFAULT_ROUTING_MAP, compile_graph
from ..nodes import (
    route_intent,
    generate_sql,
    execute_query,
    generate_response,
    handle_greeting,
    handle_farewell,
    handle_off_topic
)


def create_market_assistant_graph():
    """
    Market asistan graph'ını oluştur ve derle.
    
    Graph yapısı:
    
    [User Input]
         │
         ▼
    ┌─────────┐
    │ Router  │ ──► Intent Detection
    └────┬────┘
         │
         ├──► greeting_node ──► END
         │
         ├──► farewell_node ──► END
         │
         ├──► off_topic_node ──► END
         │
         └──► sql_generator_node
                    │
                    ▼
              sql_executor_node
                    │
                    ▼
            response_generator_node
                    │
                    ▼
                   END
    
    Returns:
        CompiledGraph: Çalıştırılabilir market assistant graph
    """
    
    workflow = create_state_graph()
    
    workflow.add_node("router", route_intent)
    workflow.add_node("sql_generator_node", generate_sql)
    workflow.add_node("sql_executor_node", execute_query)
    workflow.add_node("response_generator_node", generate_response)
    workflow.add_node("off_topic_node", handle_off_topic)
    workflow.add_node("greeting_node", handle_greeting)
    workflow.add_node("farewell_node", handle_farewell)
    
    workflow.set_entry_point("router")
    
    add_router_edges(workflow, "router", DEFAULT_ROUTING_MAP)
    
    workflow.add_edge("sql_generator_node", "sql_executor_node")
    workflow.add_edge("sql_executor_node", "response_generator_node")
    workflow.add_edge("response_generator_node", END)
    
    workflow.add_edge("off_topic_node", END)
    workflow.add_edge("greeting_node", END)
    workflow.add_edge("farewell_node", END)
    
    return compile_graph(workflow)
