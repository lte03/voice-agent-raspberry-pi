import random
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from ..state import GraphState
from ..prompts import OFF_TOPIC_PROMPT, GREETINGS, FAREWELLS
from ..tools import get_random_products
from ..config import LLM_MODEL, RESPONSE_LLM_TEMPERATURE, RESPONSE_LLM_MAX_TOKENS


# LLM instance for off-topic handling
_handler_llm = None


def get_handler_llm() -> ChatOpenAI:
    """Handler LLM instance'ını döndür (lazy initialization)"""
    global _handler_llm
    if _handler_llm is None:
        _handler_llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=RESPONSE_LLM_TEMPERATURE,
            max_tokens=RESPONSE_LLM_MAX_TOKENS
        )
    return _handler_llm


def handle_greeting(state: GraphState) -> GraphState:
    """
    Selamlama yanıtı oluştur.
    
    Args:
        state: Mevcut graph state
    
    Returns:
        GraphState: response eklenmiş state
    """
    state["response"] = random.choice(GREETINGS)
    return state


def handle_farewell(state: GraphState) -> GraphState:
    """
    Vedalaşma yanıtı oluştur.
    
    Args:
        state: Mevcut graph state
    
    Returns:
        GraphState: response eklenmiş state
    """
    state["response"] = random.choice(FAREWELLS)
    return state


def handle_off_topic(state: GraphState) -> GraphState:
    """
    Konu dışı soruları işle - kibarca reddet ve ürüne yönlendir.
    
    Args:
        state: Mevcut graph state
    
    Returns:
        GraphState: response eklenmiş state
    """
    user_input = state["user_input"]
    
    # Rastgele ürünler al
    products = get_random_products(limit=3)
    products_text = ", ".join([
        f"{p['name']} ({p['price']} TL)" 
        for p in products
    ])
    
    llm = get_handler_llm()
    
    prompt = OFF_TOPIC_PROMPT.format(
        question=user_input,
        products=products_text
    )
    
    messages = [
        SystemMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    state["response"] = response.content
    
    return state
