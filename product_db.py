import sqlite3
import re
from datetime import datetime


class ProductDB:
    def __init__(self, db_path="product_catalog.db"):
        self.db_path = db_path
        self.init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    @staticmethod
    def _normalize_text(value):
        if value is None:
            return ""
        value = value.casefold()
        value = re.sub(r"[^0-9a-zA-Z ]+", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        return value

    def _find_category_candidates(self, name):
        target = self._normalize_text(name)
        if not target:
            return []
        rows = self.fetch_all_categories()
        exact = [row for row in rows if self._normalize_text(row["name"]) == target]
        if exact:
            return exact
        return [row for row in rows if target in self._normalize_text(row["name"])]

    def _find_product_candidates(self, name):
        target = self._normalize_text(name)
        if not target:
            return []
        rows = self.fetch_all_products()
        exact = [row for row in rows if self._normalize_text(row["name"]) == target]
        if exact:
            return exact
        return [row for row in rows if target in self._normalize_text(row["name"])]

    def fetch_all_categories(self):
        conn = self._get_connection()
        try:
            return conn.execute("SELECT id, name FROM categories ORDER BY name").fetchall()
        finally:
            conn.close()

    def fetch_all_brands(self):
        conn = self._get_connection()
        try:
            return conn.execute("SELECT id, name FROM brands ORDER BY name").fetchall()
        finally:
            conn.close()

    def fetch_all_products(self):
        conn = self._get_connection()
        try:
            return conn.execute(
                """
                SELECT p.id, p.name, p.price, p.stock_qty, b.name AS brand_name
                FROM products AS p
                LEFT JOIN brands AS b ON b.id = p.brand_id
                ORDER BY p.name
                """
            ).fetchall()
        finally:
            conn.close()

    def delete_category_by_name(self, category_name):
        candidates = self._find_category_candidates(category_name)
        if not candidates:
            return {"status": "not_found", "category_name": category_name}
        if len(candidates) > 1:
            return {
                "status": "multiple",
                "category_name": category_name,
                "candidates": [row["name"] for row in candidates],
            }
        category = candidates[0]
        conn = self._get_connection()
        try:
            product_ids = [
                row["id"]
                for row in conn.execute(
                    "SELECT id FROM products WHERE category_id = ?",
                    (category["id"],),
                ).fetchall()
            ]
            product_count = len(product_ids)
            if product_ids:
                conn.execute(
                    "DELETE FROM products WHERE category_id = ?",
                    (category["id"],),
                )
            conn.execute("DELETE FROM categories WHERE id = ?", (category["id"],))
            conn.commit()
        finally:
            conn.close()
        return {
            "status": "deleted",
            "category_name": category["name"],
            "deleted_product_count": product_count,
        }

    def delete_product_by_name(self, product_name):
        candidates = self._find_product_candidates(product_name)
        if not candidates:
            return {"status": "not_found", "product_name": product_name}
        if len(candidates) > 1:
            return {
                "status": "multiple",
                "product_name": product_name,
                "candidates": [row["name"] for row in candidates],
            }
        product = candidates[0]
        conn = self._get_connection()
        try:
            conn.execute("DELETE FROM products WHERE id = ?", (product["id"],))
            conn.commit()
        finally:
            conn.close()
        return {
            "status": "deleted",
            "product_name": product["name"],
        }

    def list_categories(self, limit=12):
        rows = self.fetch_all_categories()
        return {
            "status": "ok",
            "count": len(rows),
            "items": [row["name"] for row in rows[:limit]],
        }

    def list_products(self, limit=12):
        rows = self.fetch_all_products()
        seen = set()
        items = []
        for row in rows:
            key = self._normalize_text(row["name"])
            if not key or key in seen:
                continue
            seen.add(key)
            items.append(
                {
                    "name": row["name"],
                    "brand": row["brand_name"],
                    "price": row["price"],
                    "stock_qty": row["stock_qty"],
                }
            )
        return {
            "status": "ok",
            "count": len(items),
            "items": items[:limit],
        }

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS suppliers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                phone TEXT,
                email TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS locations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                aisle_code TEXT NOT NULL,
                shelf_code TEXT NOT NULL,
                description TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS brands (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                category_id INTEGER NOT NULL,
                supplier_id INTEGER,
                location_id INTEGER,
                brand_id INTEGER,
                price REAL NOT NULL DEFAULT 0,
                stock_qty INTEGER NOT NULL DEFAULT 0,
                is_active INTEGER NOT NULL DEFAULT 1,
                FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE CASCADE,
                FOREIGN KEY (supplier_id) REFERENCES suppliers(id) ON DELETE SET NULL,
                FOREIGN KEY (location_id) REFERENCES locations(id) ON DELETE SET NULL,
                FOREIGN KEY (brand_id) REFERENCES brands(id) ON DELETE SET NULL
            )
            """
        )

        cursor.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_categories_name_not_empty_insert
            BEFORE INSERT ON categories
            FOR EACH ROW
            WHEN TRIM(NEW.name) = ''
            BEGIN
                SELECT RAISE(ABORT, 'category name is required');
            END;
            """
        )
        cursor.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_categories_name_not_empty_update
            BEFORE UPDATE ON categories
            FOR EACH ROW
            WHEN TRIM(NEW.name) = ''
            BEGIN
                SELECT RAISE(ABORT, 'category name is required');
            END;
            """
        )
        cursor.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_brands_name_not_empty_insert
            BEFORE INSERT ON brands
            FOR EACH ROW
            WHEN TRIM(NEW.name) = ''
            BEGIN
                SELECT RAISE(ABORT, 'brand name is required');
            END;
            """
        )
        cursor.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_brands_name_not_empty_update
            BEFORE UPDATE ON brands
            FOR EACH ROW
            WHEN TRIM(NEW.name) = ''
            BEGIN
                SELECT RAISE(ABORT, 'brand name is required');
            END;
            """
        )
        cursor.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_suppliers_name_not_empty_insert
            BEFORE INSERT ON suppliers
            FOR EACH ROW
            WHEN TRIM(NEW.name) = ''
            BEGIN
                SELECT RAISE(ABORT, 'supplier name is required');
            END;
            """
        )
        cursor.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_suppliers_name_not_empty_update
            BEFORE UPDATE ON suppliers
            FOR EACH ROW
            WHEN TRIM(NEW.name) = ''
            BEGIN
                SELECT RAISE(ABORT, 'supplier name is required');
            END;
            """
        )

        cursor.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_products_name_not_empty_insert
            BEFORE INSERT ON products
            FOR EACH ROW
            WHEN TRIM(NEW.name) = ''
            BEGIN
                SELECT RAISE(ABORT, 'product name is required');
            END;
            """
        )
        cursor.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_products_name_not_empty_update
            BEFORE UPDATE ON products
            FOR EACH ROW
            WHEN TRIM(NEW.name) = ''
            BEGIN
                SELECT RAISE(ABORT, 'product name is required');
            END;
            """
        )
        cursor.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_products_price_nonnegative_insert
            BEFORE INSERT ON products
            FOR EACH ROW
            WHEN NEW.price < 0
            BEGIN
                SELECT RAISE(ABORT, 'price must be non-negative');
            END;
            """
        )
        cursor.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_products_price_nonnegative_update
            BEFORE UPDATE ON products
            FOR EACH ROW
            WHEN NEW.price < 0
            BEGIN
                SELECT RAISE(ABORT, 'price must be non-negative');
            END;
            """
        )
        cursor.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_products_stock_nonnegative_insert
            BEFORE INSERT ON products
            FOR EACH ROW
            WHEN NEW.stock_qty < 0
            BEGIN
                SELECT RAISE(ABORT, 'stock must be non-negative');
            END;
            """
        )
        cursor.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_products_stock_nonnegative_update
            BEFORE UPDATE ON products
            FOR EACH ROW
            WHEN NEW.stock_qty < 0
            BEGIN
                SELECT RAISE(ABORT, 'stock must be non-negative');
            END;
            """
        )
        cursor.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_products_is_active_valid_insert
            BEFORE INSERT ON products
            FOR EACH ROW
            WHEN NEW.is_active NOT IN (0, 1)
            BEGIN
                SELECT RAISE(ABORT, 'is_active must be 0 or 1');
            END;
            """
        )
        cursor.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_products_is_active_valid_update
            BEFORE UPDATE ON products
            FOR EACH ROW
            WHEN NEW.is_active NOT IN (0, 1)
            BEGIN
                SELECT RAISE(ABORT, 'is_active must be 0 or 1');
            END;
            """
        )
        cursor.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_products_deactivate_on_zero_stock_insert
            AFTER INSERT ON products
            FOR EACH ROW
            WHEN NEW.stock_qty = 0 AND NEW.is_active <> 0
            BEGIN
                UPDATE products SET is_active = 0 WHERE id = NEW.id;
            END;
            """
        )
        cursor.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_products_deactivate_on_zero_stock_update
            AFTER UPDATE OF stock_qty ON products
            FOR EACH ROW
            WHEN NEW.stock_qty = 0 AND NEW.is_active <> 0
            BEGIN
                UPDATE products SET is_active = 0 WHERE id = NEW.id;
            END;
            """
        )
        cursor.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_products_activate_on_positive_stock_update
            AFTER UPDATE OF stock_qty ON products
            FOR EACH ROW
            WHEN NEW.stock_qty > 0 AND NEW.is_active = 0
            BEGIN
                UPDATE products SET is_active = 1 WHERE id = NEW.id;
            END;
            """
        )

        cursor.execute("PRAGMA table_info(products)")
        product_columns = {row[1] for row in cursor.fetchall()}
        if "description" not in product_columns:
            cursor.execute("ALTER TABLE products ADD COLUMN description TEXT")
        if "stock_qty" not in product_columns:
            cursor.execute(
                "ALTER TABLE products ADD COLUMN stock_qty INTEGER NOT NULL DEFAULT 0"
            )
        if "location_id" not in product_columns:
            cursor.execute("ALTER TABLE products ADD COLUMN location_id INTEGER")
        if "brand_id" not in product_columns:
            cursor.execute("ALTER TABLE products ADD COLUMN brand_id INTEGER")
        if "is_active" not in product_columns:
            cursor.execute(
                "ALTER TABLE products ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1"
            )

        cursor.execute("SELECT id FROM locations ORDER BY id LIMIT 1")
        location_row = cursor.fetchone()
        if location_row:
            default_location_id = location_row[0]
        else:
            cursor.execute(
                """
                INSERT INTO locations (aisle_code, shelf_code, description)
                VALUES (?, ?, ?)
                """,
                ("A1", "R1", "Default location"),
            )
            default_location_id = cursor.lastrowid

        cursor.execute("SELECT id FROM brands ORDER BY id LIMIT 1")
        brand_row = cursor.fetchone()
        if brand_row:
            default_brand_id = brand_row[0]
        else:
            cursor.execute(
                "INSERT INTO brands (name) VALUES (?)",
                ("DailyBrand",),
            )
            default_brand_id = cursor.lastrowid

        cursor.execute("UPDATE products SET stock_qty = 0 WHERE stock_qty IS NULL")
        cursor.execute("UPDATE products SET description = '' WHERE description IS NULL")
        cursor.execute("UPDATE products SET is_active = 1 WHERE is_active IS NULL")
        cursor.execute(
            "UPDATE products SET location_id = ? WHERE location_id IS NULL",
            (default_location_id,),
        )
        cursor.execute(
            "UPDATE products SET brand_id = ? WHERE brand_id IS NULL",
            (default_brand_id,),
        )

        conn.commit()
        conn.close()

    def seed_example_data(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")
        cursor.execute("SELECT COUNT(*) FROM products")
        if cursor.fetchone()[0] > 0:
            conn.close()
            return

        categories = [
            ("Gida", "Gunluk gida urunleri"),
            ("Icecek", "Gunluk icecekler"),
            ("Temizlik", "Ev temizlik urunleri"),
            ("Kisisel Bakim", "Gunluk kisisel bakim"),
            ("Ev Gerecleri", "Evde kullanilan temel gerecler"),
        ]

        suppliers = [
            ("MarketHub", "0212-111-0001", "siparis@markethub.com"),
            ("AnadoluDagitim", "0212-222-0002", "info@anadoludagitim.com"),
            ("FreshLine", "0212-333-0003", "destek@freshline.com"),
        ]

        locations = [
            ("A1", "R1", "Giris sag raf"),
            ("A2", "R3", "Orta koridor"),
            ("B1", "R2", "Arka bolum"),
        ]

        brands = [
            ("Ulker",),
            ("Eti",),
            ("Pinar",),
            ("Sutas",),
            ("Torku",),
            ("Caykur",),
            ("Uludag",),
            ("Peros",),
            ("Evyap",),
            ("Selpak",),
            ("Solo",),
        ]

        cursor.executemany(
            "INSERT OR IGNORE INTO categories (name, description) VALUES (?, ?)",
            categories,
        )
        cursor.executemany(
            "INSERT OR IGNORE INTO suppliers (name, phone, email) VALUES (?, ?, ?)",
            suppliers,
        )
        cursor.executemany(
            "INSERT OR IGNORE INTO locations (aisle_code, shelf_code, description) VALUES (?, ?, ?)",
            locations,
        )
        cursor.executemany(
            "INSERT OR IGNORE INTO brands (name) VALUES (?)",
            brands,
        )

        cursor.execute("SELECT id, name FROM categories")
        category_map = {name: _id for _id, name in cursor.fetchall()}
        cursor.execute("SELECT id, name FROM suppliers")
        supplier_map = {name: _id for _id, name in cursor.fetchall()}
        cursor.execute("SELECT id FROM locations ORDER BY id")
        location_ids = [row[0] for row in cursor.fetchall()]
        cursor.execute("SELECT id, name FROM brands")
        brand_map = {name: _id for _id, name in cursor.fetchall()}

        if not location_ids:
            cursor.execute(
                """
                INSERT INTO locations (aisle_code, shelf_code, description)
                VALUES (?, ?, ?)
                """,
                ("A1", "R1", "Default location"),
            )
            location_ids = [cursor.lastrowid]

        now = datetime.now().isoformat()
        base_products = [
            ("Ekmek", "Gunluk tuketim ekmegi", "Gida", "MarketHub", "Ulker", 12.5, 120),
            ("Sut 1L", "Gunluk inek sutu", "Gida", "FreshLine", "Sutas", 24.0, 0),
            ("Yumurta 10lu", "Gunluk tuketim yumurta", "Gida", "FreshLine", "Pinar", 52.0, 60),
            ("Peynir 500g", "Beyaz peynir", "Gida", "MarketHub", "Pinar", 89.0, 40),
            ("Yogurt 1kg", "Dogal yogurt", "Gida", "FreshLine", "Sutas", 64.0, 55),
            ("Pirinc 1kg", "Uzun taneli pirinc", "Gida", "AnadoluDagitim", "Torku", 48.0, 70),
            ("Makarna 500g", "Fiyonk makarna", "Gida", "AnadoluDagitim", "Eti", 22.0, 90),
            ("Sivi Yag 1L", "Ay cicek yagi", "Gida", "AnadoluDagitim", "Torku", 79.0, 45),
            ("Seker 1kg", "Toz seker", "Gida", "MarketHub", "Torku", 44.0, 50),
            ("Tuz 750g", "Iyotlu tuz", "Gida", "MarketHub", "Torku", 12.0, 110),
            ("Cay 500g", "Siyah cay", "Icecek", "MarketHub", "Caykur", 68.0, 65),
            ("Kahve 200g", "Filtre kahve", "Icecek", "MarketHub", "Ulker", 92.0, 30),
            ("Maden Suyu 6li", "Sade maden suyu", "Icecek", "FreshLine", "Uludag", 36.0, 75),
            ("Bulasik Deterjani", "Limon kokulu", "Temizlik", "AnadoluDagitim", "Peros", 55.0, 40),
            ("Camasir Deterjani", "Renkli", "Temizlik", "AnadoluDagitim", "Peros", 129.0, 25),
            ("Yuzey Temizleyici", "Genel amacli", "Temizlik", "MarketHub", "Peros", 42.0, 50),
            ("Sampuan 500ml", "Gunluk kullanim", "Kisisel Bakim", "MarketHub", "Evyap", 59.0, 35),
            ("Dis Macunu", "Florur 75ml", "Kisisel Bakim", "MarketHub", "Evyap", 38.0, 70),
            ("Sabun 4lu", "El sabunu", "Kisisel Bakim", "FreshLine", "Evyap", 28.0, 60),
            ("Kagit Havlu", "2 katli", "Ev Gerecleri", "AnadoluDagitim", "Selpak", 49.0, 55),
            ("Tuvalet Kagidi", "8li paket", "Ev Gerecleri", "AnadoluDagitim", "Solo", 84.0, 45),
            ("Cop Torbasi", "Buzdolabi boy", "Ev Gerecleri", "MarketHub", "Solo", 26.0, 90),
        ]

        products = []
        for index, item in enumerate(base_products):
            name, description, category_name, supplier_name, brand_name, price, stock_qty = item
            location_id = location_ids[index % len(location_ids)]
            products.append(
                (
                    name,
                    description,
                    category_map[category_name],
                    supplier_map[supplier_name],
                    location_id,
                    brand_map[brand_name],
                    price,
                    stock_qty,
                    1,
                )
            )

        cursor.execute("PRAGMA table_info(products)")
        product_columns = {row[1] for row in cursor.fetchall()}
        if "created_at" in product_columns:
            products = [row + (now,) for row in products]
            insert_sql = """
                INSERT OR IGNORE INTO products
                    (name, description, category_id, supplier_id, location_id, brand_id, price, stock_qty, is_active, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
        else:
            insert_sql = """
                INSERT OR IGNORE INTO products
                    (name, description, category_id, supplier_id, location_id, brand_id, price, stock_qty, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

        cursor.executemany(insert_sql, products)

        conn.commit()
        conn.close()


if __name__ == "__main__":
    prod = ProductDB()
    prod.seed_example_data()
