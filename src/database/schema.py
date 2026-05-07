CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS companies (
    company_id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_name TEXT NOT NULL UNIQUE,
    industry TEXT DEFAULT 'General',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS upload_batches (
    batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER NOT NULL,
    source_filename TEXT,
    row_count INTEGER NOT NULL,
    data_quality_score INTEGER NOT NULL,
    uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (company_id) REFERENCES companies(company_id)
);

CREATE TABLE IF NOT EXISTS stores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER NOT NULL,
    store_id TEXT NOT NULL,
    store_location TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company_id, store_id),
    FOREIGN KEY (company_id) REFERENCES companies(company_id)
);

CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER NOT NULL,
    product_id TEXT NOT NULL,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    unit_price REAL NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company_id, product_id),
    FOREIGN KEY (company_id) REFERENCES companies(company_id)
);

CREATE TABLE IF NOT EXISTS inventory_records (
    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER NOT NULL,
    company_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    store_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    opening_stock REAL NOT NULL,
    purchased_quantity REAL NOT NULL,
    sold_quantity REAL NOT NULL,
    wasted_quantity REAL NOT NULL,
    closing_stock REAL NOT NULL,
    unit_price REAL NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (batch_id) REFERENCES upload_batches(batch_id),
    FOREIGN KEY (company_id) REFERENCES companies(company_id)
);
"""

CREATE_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_inventory_records_batch_id
ON inventory_records(batch_id);

CREATE INDEX IF NOT EXISTS idx_inventory_records_company_id
ON inventory_records(company_id);

CREATE INDEX IF NOT EXISTS idx_inventory_records_product_id
ON inventory_records(product_id);

CREATE INDEX IF NOT EXISTS idx_inventory_records_store_id
ON inventory_records(store_id);

CREATE INDEX IF NOT EXISTS idx_inventory_records_date
ON inventory_records(date);
"""