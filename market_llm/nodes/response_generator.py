import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from ..state import GraphState
from ..prompts import RESPONSE_PROMPT
from ..config import LLM_MODEL, RESPONSE_LLM_TEMPERATURE, RESPONSE_LLM_MAX_TOKENS


# LLM instance for response generation
_response_llm = None


def get_response_llm() -> ChatOpenAI:
    """Response LLM instance'ını döndür (lazy initialization)"""
    global _response_llm
    if _response_llm is None:
        _response_llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=RESPONSE_LLM_TEMPERATURE,
            max_tokens=RESPONSE_LLM_MAX_TOKENS
        )
    return _response_llm


def generate_response(state: GraphState) -> GraphState:
    """
    SQL sonucuna göre doğal dil yanıtı oluştur.
    
    Args:
        state: Mevcut graph state (db_result içermeli)
    
    Returns:
        GraphState: response eklenmiş state
    """
    user_input = state["user_input"]
    db_result = state.get("db_result", {})
    
    # Hata kontrolü
    if not db_result.get("success", False):
        error_msg = db_result.get("error", "Bilinmeyen hata")
        state["response"] = f"Bir sorun oluştu: {error_msg}"
        state["error"] = error_msg
        return state
    
    llm = get_response_llm()
    
    # Prompt'u formatla
    prompt = RESPONSE_PROMPT.format(
        question=user_input,
        result=json.dumps(db_result, ensure_ascii=False, indent=2)
    )
    
    messages = [
        SystemMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    state["response"] = response.content
    
    return state
