from typing import List, Dict

try:
    from .graphs import create_market_assistant_graph
    from .state import GraphState
except ImportError:
    # Allow running as a script: python market_llm/main.py
    import os
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    from market_llm.graphs import create_market_assistant_graph
    from market_llm.state import GraphState


class MarketAssistant:
    """
    Market Asistan - Text-to-SQL powered
    
    Kullanım:
        assistant = MarketAssistant()
        response = assistant.chat("Yoğurt var mı?")
        print(response)
    """
    
    def __init__(self):
        """MarketAssistant'ı başlat."""
        self.graph = create_market_assistant_graph()
        self.conversation_history: List[Dict[str, str]] = []
    
    def chat(self, user_input: str) -> str:
        """
        Kullanıcı mesajını işle ve yanıt döndür.
        
        Args:
            user_input: Kullanıcının mesajı
        
        Returns:
            str: Asistanın yanıtı
        """
        
        initial_state: GraphState = {
            "user_input": user_input,
            "intent": "",
            "sql_query": None,
            "db_result": None,
            "response": "",
            "conversation_history": self.conversation_history,
            "error": None
        }
        
        result = self.graph.invoke(initial_state)
        
        self.conversation_history.append({
            "role": "user", 
            "content": user_input
        })
        self.conversation_history.append({
            "role": "assistant", 
            "content": result["response"]
        })
        
        # Son 20 mesajı tut
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        return result["response"]
    
    def debug_sql(self, user_input: str) -> dict:
        """
        Debug için SQL ve sonucu göster.
        
        Args:
            user_input: Kullanıcının mesajı
        
        Returns:
            dict: {intent, sql, db_result, response}
        """
        initial_state: GraphState = {
            "user_input": user_input,
            "intent": "",
            "sql_query": None,
            "db_result": None,
            "response": "",
            "conversation_history": [],
            "error": None
        }
        
        result = self.graph.invoke(initial_state)
        
        return {
            "intent": result.get("intent"),
            "sql": result.get("sql_query"),
            "db_result": result.get("db_result"),
            "response": result.get("response")
        }
    
    def clear_history(self):
        """Konuşma geçmişini temizle."""
        self.conversation_history = []

def run_cli():
    import sys
    sys.path.insert(0, '..')
    
    from product_db import ProductDB
    
    print("=" * 60)
    print("🛒 Market Asistan - Text-to-SQL")
    print("=" * 60)
    print("Çıkmak için 'q' veya 'quit' yazın.\n")
    
    # Veritabanını hazırla
    db = ProductDB("product_catalog.db")
    db.seed_example_data()
    
    assistant = MarketAssistant()
    
    while True:
        try:
            user_input = input("👤 Sen: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['q', 'quit', 'exit', 'çık']:
                print("Görüşmek üzere! 👋")
                break
            
            response = assistant.chat(user_input)
            print(f"🤖 Asistan: {response}\n")
            
        except KeyboardInterrupt:
            print("\nGörüşmek üzere! 👋")
            break
        except Exception as e:
            print(f"Hata: {e}\n")


if __name__ == "__main__":
    run_cli()
