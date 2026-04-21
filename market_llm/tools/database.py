"""
Market LLM - Database Tools
"""

import sqlite3
from ..config import DB_PATH


def execute_sql(query: str, db_path: str = None) -> dict:
    """
    SQL sorgusunu güvenli şekilde çalıştır.
    
    Args:
        query: Çalıştırılacak SQL sorgusu
        db_path: Veritabanı yolu (varsayılan: config'den alınır)
    
    Returns:
        dict: {success, row_count, rows, error}
    """
    if db_path is None:
        db_path = DB_PATH
        
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Sadece SELECT sorgularına izin ver (güvenlik)
        query_upper = query.strip().upper()
        if not query_upper.startswith("SELECT"):
            return {
                "success": False,
                "error": "Sadece SELECT sorguları destekleniyor",
                "rows": []
            }
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # Row objelerini dict'e çevir
        result_rows = [dict(row) for row in rows]
        
        conn.close()
        
        return {
            "success": True,
            "row_count": len(result_rows),
            "rows": result_rows[:20]  # Max 20 sonuç
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "rows": []
        }


def get_random_products(limit: int = 3, db_path: str = None) -> list:
    """
    Rastgele ürünler getir (off-topic yanıtlar için).
    
    Args:
        limit: Maksimum ürün sayısı
        db_path: Veritabanı yolu
    
    Returns:
        list: Ürün listesi
    """
    result = execute_sql(
        f"""
        SELECT p.name, p.price FROM products p 
        WHERE p.stock_qty > 0 
        ORDER BY RANDOM() 
        LIMIT {limit}
        """,
        db_path
    )
    return result.get("rows", [])
