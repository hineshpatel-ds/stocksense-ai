from __future__ import annotations

from typing import Optional

import pandas as pd

from src.data.validation import REQUIRED_COLUMNS, standardize_column_names
from src.database.connection import get_connection, initialize_database


class InventoryRepository:
    """
    Repository for saving and loading inventory data.

    The repository pattern keeps database logic separate from the API,
    dashboard, analytics, forecasting, and recommendation modules.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path

    def setup(self) -> None:
        """
        Initialize database tables.
        """

        initialize_database(self.db_path)

    def get_or_create_company(
        self,
        company_name: str = "Demo Company",
        industry: str = "General",
    ) -> int:
        """
        Create company if it does not exist and return company_id.
        """

        self.setup()

        with get_connection(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO companies (company_name, industry)
                VALUES (?, ?)
                ON CONFLICT(company_name)
                DO UPDATE SET industry = excluded.industry
                """,
                (company_name, industry),
            )

            company_row = connection.execute(
                """
                SELECT company_id
                FROM companies
                WHERE company_name = ?
                """,
                (company_name,),
            ).fetchone()

            connection.commit()

        return int(company_row["company_id"])

    def save_inventory_dataframe(
        self,
        df: pd.DataFrame,
        company_name: str = "Demo Company",
        industry: str = "General",
        source_filename: str = "uploaded_inventory.csv",
        data_quality_score: int = 100,
    ) -> int:
        """
        Save validated inventory data into the database.

        Returns:
            batch_id for the uploaded file.
        """

        self.setup()

        prepared_df = self._prepare_dataframe_for_storage(df)
        company_id = self.get_or_create_company(company_name, industry)

        with get_connection(self.db_path) as connection:
            batch_cursor = connection.execute(
                """
                INSERT INTO upload_batches (
                    company_id,
                    source_filename,
                    row_count,
                    data_quality_score
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    company_id,
                    source_filename,
                    len(prepared_df),
                    int(data_quality_score),
                ),
            )

            batch_id = int(batch_cursor.lastrowid)

            self._upsert_stores(connection, prepared_df, company_id)
            self._upsert_products(connection, prepared_df, company_id)
            self._insert_inventory_records(
                connection=connection,
                df=prepared_df,
                batch_id=batch_id,
                company_id=company_id,
            )

            connection.commit()

        return batch_id

    def load_inventory_records(self, batch_id: Optional[int] = None) -> pd.DataFrame:
        """
        Load inventory records from the database.

        If batch_id is not provided, the latest upload batch is loaded.
        """

        self.setup()

        with get_connection(self.db_path) as connection:
            if batch_id is None:
                batch_row = connection.execute(
                    """
                    SELECT batch_id
                    FROM upload_batches
                    ORDER BY batch_id DESC
                    LIMIT 1
                    """
                ).fetchone()

                if batch_row is None:
                    return pd.DataFrame(columns=REQUIRED_COLUMNS)

                batch_id = int(batch_row["batch_id"])

            query = """
                SELECT
                    r.date,
                    r.store_id,
                    r.product_id,
                    p.product_name,
                    p.category,
                    r.opening_stock,
                    r.purchased_quantity,
                    r.sold_quantity,
                    r.wasted_quantity,
                    r.closing_stock,
                    r.unit_price
                FROM inventory_records r
                LEFT JOIN products p
                    ON p.company_id = r.company_id
                    AND p.product_id = r.product_id
                WHERE r.batch_id = ?
                ORDER BY r.date, r.store_id, r.product_id
            """

            records_df = pd.read_sql_query(query, connection, params=(batch_id,))

        return records_df

    def get_upload_batches(self, limit: int = 10) -> pd.DataFrame:
        """
        Return recent upload batches.
        """

        self.setup()

        with get_connection(self.db_path) as connection:
            query = """
                SELECT
                    b.batch_id,
                    c.company_name,
                    c.industry,
                    b.source_filename,
                    b.row_count,
                    b.data_quality_score,
                    b.uploaded_at
                FROM upload_batches b
                INNER JOIN companies c
                    ON c.company_id = b.company_id
                ORDER BY b.batch_id DESC
                LIMIT ?
            """

            batches_df = pd.read_sql_query(query, connection, params=(limit,))

        return batches_df

    def _prepare_dataframe_for_storage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize and clean DataFrame before saving to SQLite.
        """

        prepared_df = standardize_column_names(df)

        missing_columns = [
            column for column in REQUIRED_COLUMNS if column not in prepared_df.columns
        ]

        if missing_columns:
            raise ValueError(f"Missing required columns for database save: {missing_columns}")

        prepared_df = prepared_df.copy()

        prepared_df["date"] = pd.to_datetime(
            prepared_df["date"],
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")

        numeric_columns = [
            "opening_stock",
            "purchased_quantity",
            "sold_quantity",
            "wasted_quantity",
            "closing_stock",
            "unit_price",
        ]

        for column in numeric_columns:
            prepared_df[column] = pd.to_numeric(
                prepared_df[column],
                errors="coerce",
            ).fillna(0)

        if "store_location" not in prepared_df.columns:
            prepared_df["store_location"] = None

        return prepared_df

    def _upsert_stores(
        self,
        connection,
        df: pd.DataFrame,
        company_id: int,
    ) -> None:
        """
        Insert or update store records.
        """

        store_rows = (
            df[["store_id", "store_location"]]
            .drop_duplicates(subset=["store_id"])
            .values
            .tolist()
        )

        connection.executemany(
            """
            INSERT INTO stores (company_id, store_id, store_location)
            VALUES (?, ?, ?)
            ON CONFLICT(company_id, store_id)
            DO UPDATE SET store_location = excluded.store_location
            """,
            [
                (
                    company_id,
                    str(store_id),
                    None if pd.isna(store_location) else str(store_location),
                )
                for store_id, store_location in store_rows
            ],
        )

    def _upsert_products(
        self,
        connection,
        df: pd.DataFrame,
        company_id: int,
    ) -> None:
        """
        Insert or update product records.
        """

        product_rows = (
            df[["product_id", "product_name", "category", "unit_price"]]
            .drop_duplicates(subset=["product_id"], keep="last")
            .values
            .tolist()
        )

        connection.executemany(
            """
            INSERT INTO products (
                company_id,
                product_id,
                product_name,
                category,
                unit_price
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(company_id, product_id)
            DO UPDATE SET
                product_name = excluded.product_name,
                category = excluded.category,
                unit_price = excluded.unit_price,
                updated_at = CURRENT_TIMESTAMP
            """,
            [
                (
                    company_id,
                    str(product_id),
                    str(product_name),
                    str(category),
                    float(unit_price),
                )
                for product_id, product_name, category, unit_price in product_rows
            ],
        )

    def _insert_inventory_records(
        self,
        connection,
        df: pd.DataFrame,
        batch_id: int,
        company_id: int,
    ) -> None:
        """
        Insert inventory movement records.
        """

        record_rows = []

        for _, row in df.iterrows():
            record_rows.append(
                (
                    batch_id,
                    company_id,
                    str(row["date"]),
                    str(row["store_id"]),
                    str(row["product_id"]),
                    float(row["opening_stock"]),
                    float(row["purchased_quantity"]),
                    float(row["sold_quantity"]),
                    float(row["wasted_quantity"]),
                    float(row["closing_stock"]),
                    float(row["unit_price"]),
                )
            )

        connection.executemany(
            """
            INSERT INTO inventory_records (
                batch_id,
                company_id,
                date,
                store_id,
                product_id,
                opening_stock,
                purchased_quantity,
                sold_quantity,
                wasted_quantity,
                closing_stock,
                unit_price
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            record_rows,
        )