from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from ..state import GraphState
from ..prompts import TEXT_TO_SQL_PROMPT
from ..config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS


_sql_llm = None


def get_sql_llm() -> ChatOpenAI:
    """SQL LLM instance'ını döndür (lazy initialization)"""
    global _sql_llm
    if _sql_llm is None:
        _sql_llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0, 
            max_tokens=LLM_MAX_TOKENS
        )
    return _sql_llm


def generate_sql(state: GraphState) -> GraphState:
    """
    Kullanıcı sorusundan SQL sorgusu üret.
    
    Args:
        state: Mevcut graph state
    
    Returns:
        GraphState: SQL sorgusu eklenmiş state
    """
    user_input = state["user_input"]
    llm = get_sql_llm()
    
    prompt = TEXT_TO_SQL_PROMPT.format(question=user_input)
    
    messages = [
        SystemMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    sql = response.content.strip()
    
    if sql.startswith("```"):
        parts = sql.split("```")
        if len(parts) > 1:
            sql = parts[1]
            if sql.startswith("sql"):
                sql = sql[3:]
    
    sql = sql.strip()
    
    state["sql_query"] = sql
    
    print(f"[SQL Generator] Generated: {sql}")
    
    return state
