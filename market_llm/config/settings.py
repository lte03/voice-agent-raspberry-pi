import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# DATABASE SETTINGS
# ============================================================================

DB_PATH = os.getenv("DB_PATH", "product_catalog.db")

# ============================================================================
# LLM SETTINGS
# ============================================================================

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1000"))

RESPONSE_LLM_TEMPERATURE = float(os.getenv("RESPONSE_LLM_TEMPERATURE", "0.7"))
RESPONSE_LLM_MAX_TOKENS = int(os.getenv("RESPONSE_LLM_MAX_TOKENS", "500"))

# ============================================================================
# DATABASE SCHEMA (for Text-to-SQL)
# ============================================================================

DB_SCHEMA = """
VERITABANI ŞEMASI (SQLite):

TABLE categories (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,        -- Kategori adı (örn: "Gida", "Icecek", "Temizlik")
    description TEXT                   -- Kategori açıklaması
)

TABLE brands (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE          -- Marka adı (örn: "Ulker", "Pinar", "Sutas")
)

TABLE suppliers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,         -- Tedarikçi adı
    phone TEXT,
    email TEXT
)

TABLE locations (
    id INTEGER PRIMARY KEY,
    aisle_code TEXT NOT NULL,          -- Koridor kodu (örn: "A1", "B2")
    shelf_code TEXT NOT NULL,          -- Raf kodu (örn: "R1", "R3")
    description TEXT                   -- Konum açıklaması
)

TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,                -- Ürün adı (örn: "Sut 1L", "Yogurt 1kg", "Ekmek")
    description TEXT,                  -- Ürün açıklaması
    category_id INTEGER NOT NULL,      -- Kategori (FOREIGN KEY -> categories.id)
    supplier_id INTEGER,               -- Tedarikçi (FOREIGN KEY -> suppliers.id)
    location_id INTEGER,               -- Konum (FOREIGN KEY -> locations.id)
    brand_id INTEGER,                  -- Marka (FOREIGN KEY -> brands.id)
    price REAL NOT NULL DEFAULT 0,     -- Fiyat (TL)
    stock_qty INTEGER NOT NULL DEFAULT 0,  -- Stok miktarı
    is_active INTEGER NOT NULL DEFAULT 1   -- Aktif mi? (1=evet, 0=hayır)
)

ÖNEMLİ NOTLAR:
- Türkçe karakterler normalize edilmiş olabilir (ö->o, ü->u, ş->s, ı->i, ğ->g, ç->c)
- Ürün ararken LIKE kullan ve büyük/küçük harf duyarsız ara
- Örnek: "yoğurt" aramak için: WHERE LOWER(name) LIKE '%yogurt%'
- Örnek: "süt" aramak için: WHERE LOWER(name) LIKE '%sut%'
"""
