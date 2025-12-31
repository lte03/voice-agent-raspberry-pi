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
        value = re.sub(r"[^0-9a-zA-ZçğıöşüÇĞİÖŞÜ ]+", " ", value)
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

    def fetch_all_products(self):
        conn = self._get_connection()
        try:
            return conn.execute(
                "SELECT id, name, price, stock_qty FROM products ORDER BY name"
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
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category_id INTEGER NOT NULL,
                supplier_id INTEGER,
                price REAL NOT NULL DEFAULT 0,
                stock_qty INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE CASCADE,
                FOREIGN KEY (supplier_id) REFERENCES suppliers(id) ON DELETE SET NULL
            )
            """
        )

        cursor.execute("DROP TABLE IF EXISTS inventory_movements")

        cursor.execute("PRAGMA table_info(products)")
        product_columns = {row[1] for row in cursor.fetchall()}
        if "stock_qty" not in product_columns:
            cursor.execute(
                "ALTER TABLE products ADD COLUMN stock_qty INTEGER NOT NULL DEFAULT 0"
            )
        cursor.execute(
            "UPDATE products SET stock_qty = 0 WHERE stock_qty IS NULL"
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
            ("Elektronik", "Gundelik elektronik urunler"),
            ("FPGA", "FPGA gelistirme kartlari ve aksesuarlar"),
            ("Mikrodenetleyici", "MCU kartlari ve moduller"),
            ("Sensor", "Endustriyel ve hobi sensorleri"),
        ]

        suppliers = [
            ("SiliconHub", "0212-000-0001", "info@siliconhub.com"),
            ("EmbeddedWorks", "0212-000-0002", "hello@embeddedworks.com"),
            ("SignalCore", "0212-000-0003", "contact@signalcore.com"),
        ]

        cursor.executemany(
            "INSERT OR IGNORE INTO categories (name, description) VALUES (?, ?)",
            categories,
        )
        cursor.executemany(
            "INSERT OR IGNORE INTO suppliers (name, phone, email) VALUES (?, ?, ?)",
            suppliers,
        )

        cursor.execute("SELECT id, name FROM categories")
        category_map = {name: _id for _id, name in cursor.fetchall()}
        cursor.execute("SELECT id, name FROM suppliers")
        supplier_map = {name: _id for _id, name in cursor.fetchall()}

        now = datetime.now().isoformat()
        products = [
            ("Akilli Telefon", category_map["Elektronik"], supplier_map["SiliconHub"], 21999.0, 18, now),
            ("Laptop", category_map["Elektronik"], supplier_map["SiliconHub"], 32999.0, 12, now),
            ("Kablosuz Kulaklik", category_map["Elektronik"], supplier_map["SiliconHub"], 1799.0, 45, now),
            ("USB-C Hub", category_map["Elektronik"], supplier_map["SiliconHub"], 499.0, 60, now),
            ("4K Monitor", category_map["Elektronik"], supplier_map["SiliconHub"], 9999.0, 9, now),
            ("Mekanik Klavye", category_map["Elektronik"], supplier_map["SiliconHub"], 2499.0, 22, now),
            ("Gaming Mouse", category_map["Elektronik"], supplier_map["SiliconHub"], 1199.0, 30, now),
            ("5V 3A Guc Adaptoru", category_map["Elektronik"], supplier_map["SignalCore"], 179.0, 55, now),
            ("Lityum Pil 18650", category_map["Elektronik"], supplier_map["SignalCore"], 129.0, 120, now),
            ("Breadboard 830 Nokta", category_map["Elektronik"], supplier_map["SignalCore"], 89.0, 70, now),
            ("Jumper Kablo Seti", category_map["Elektronik"], supplier_map["SignalCore"], 59.0, 90, now),
            ("Xilinx Artix-7 Dev Kit", category_map["FPGA"], supplier_map["EmbeddedWorks"], 14999.0, 6, now),
            ("Intel Cyclone V Board", category_map["FPGA"], supplier_map["EmbeddedWorks"], 17999.0, 4, now),
            ("FPGA PMOD Sensor Pack", category_map["FPGA"], supplier_map["EmbeddedWorks"], 1199.0, 25, now),
            ("Zynq-7000 SoC Board", category_map["FPGA"], supplier_map["EmbeddedWorks"], 23999.0, 3, now),
            ("FPGA Logic Analyzer PMOD", category_map["FPGA"], supplier_map["EmbeddedWorks"], 1599.0, 14, now),
            ("STM32 Nucleo-64", category_map["Mikrodenetleyici"], supplier_map["EmbeddedWorks"], 649.0, 40, now),
            ("STM32 Discovery Kit", category_map["Mikrodenetleyici"], supplier_map["EmbeddedWorks"], 1299.0, 16, now),
            ("ESP32 DevKit", category_map["Mikrodenetleyici"], supplier_map["SignalCore"], 449.0, 80, now),
            ("ESP8266 NodeMCU", category_map["Mikrodenetleyici"], supplier_map["SignalCore"], 329.0, 65, now),
            ("Raspberry Pi Pico", category_map["Mikrodenetleyici"], supplier_map["SignalCore"], 279.0, 75, now),
            ("Arduino Uno R3", category_map["Mikrodenetleyici"], supplier_map["SignalCore"], 499.0, 35, now),
            ("I2C Sicaklik Sensoru", category_map["Sensor"], supplier_map["SignalCore"], 99.0, 120, now),
            ("IMU 6-Eksen Sensor", category_map["Sensor"], supplier_map["SignalCore"], 349.0, 28, now),
            ("Ultrasonik Mesafe Sensoru", category_map["Sensor"], supplier_map["SignalCore"], 129.0, 65, now),
            ("Gaz Sensoru (MQ-2)", category_map["Sensor"], supplier_map["SignalCore"], 149.0, 42, now),
            ("Basinc Sensoru (BMP280)", category_map["Sensor"], supplier_map["SignalCore"], 249.0, 33, now),
            ("Isik Sensoru (LDR Modul)", category_map["Sensor"], supplier_map["SignalCore"], 49.0, 110, now),
        ]

        cursor.executemany(
            """
            INSERT OR IGNORE INTO products
                (name, category_id, supplier_id, price, stock_qty, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            products,
        )

        conn.commit()
        conn.close()


if __name__ == "__main__":
    prod = ProductDB()
    prod.seed_example_data()
