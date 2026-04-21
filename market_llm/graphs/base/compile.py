from langgraph.graph import StateGraph

def compile_graph(workflow: StateGraph):
    return workflow.compile()
