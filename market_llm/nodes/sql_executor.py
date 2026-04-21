"""
Market LLM - SQL Executor Node
Execute SQL queries safely
"""

from ..state import GraphState
from ..tools import execute_sql


def execute_query(state: GraphState) -> GraphState:
    """
    SQL sorgusunu çalıştır.
    
    Args:
        state: Mevcut graph state (sql_query içermeli)
    
    Returns:
        GraphState: db_result eklenmiş state
    """
    sql = state.get("sql_query", "")
    
    if not sql:
        state["db_result"] = {
            "success": False,
            "error": "SQL sorgusu üretilemedi",
            "rows": []
        }
        return state
    
    result = execute_sql(sql)
    state["db_result"] = result
    
    print(f"[SQL Executor] Result: {result.get('row_count', 0)} rows")
    
    return state
