import sqlite3
from datetime import datetime
import time

class ConversationDB:
    def __init__(self, db_path="conversation_history.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name TEXT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def add_message(self, user_name, session_id, role, content):
        """Mesaj ekle - user_name ile ilişkilendir"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO conversations (user_name, session_id, role, content, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_name, session_id, role, content, datetime.now().isoformat()))
        conn.commit()
        conn.close()
    
    def get_conversation_history(self, session_id, limit=20):
        """Session ID'ye göre konuşma geçmişini getir"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT role, content FROM conversations 
            WHERE session_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (session_id, limit))
        results = cursor.fetchall()
        conn.close()
        return list(reversed(results))
    
    def get_all_history_for_user(self, user_name, limit=50):
        """Kullanıcının TÜM konuşma geçmişini getir (tüm session'lardan)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT role, content FROM conversations 
            WHERE user_name = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (user_name, limit))
        results = cursor.fetchall()
        conn.close()
        return list(reversed(results))
    
    def get_latest_session_for_user(self, user_name):
        """Kullanıcı ismine göre en son session_id'yi getir"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT session_id FROM conversations 
            WHERE user_name = ? 
            ORDER BY timestamp DESC 
            LIMIT 1
        ''', (user_name,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    
    def create_new_session(self, user_name):
        """Kullanıcı için yeni session oluştur"""
        session_id = f"{user_name}_{int(time.time())}"
        return session_id
    
    def get_all_sessions_for_user(self, user_name):
        """Kullanıcının tüm session'larını getir"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT DISTINCT session_id FROM conversations 
            WHERE user_name = ? 
            ORDER BY timestamp DESC
        ''', (user_name,))
        results = cursor.fetchall()
        conn.close()
        return [r[0] for r in results]
    