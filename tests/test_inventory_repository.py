import pandas as pd

from src.database.inventory_repository import InventoryRepository


def create_repository_test_df():
    return pd.DataFrame(
        {
            "date": ["2025-10-01", "2025-10-02"],
            "store_id": ["S001", "S001"],
            "product_id": ["P001", "P001"],
            "product_name": ["Veggie Burger", "Veggie Burger"],
            "category": ["Food", "Food"],
            "opening_stock": [100, 85],
            "purchased_quantity": [20, 0],
            "sold_quantity": [30, 20],
            "wasted_quantity": [5, 0],
            "closing_stock": [85, 65],
            "unit_price": [5.99, 5.99],
        }
    )


def test_repository_can_save_and_load_inventory_records(tmp_path):
    db_path = tmp_path / "test_stocksense.db"
    repository = InventoryRepository(db_path=str(db_path))
    repository.setup()

    df = create_repository_test_df()

    batch_id = repository.save_inventory_dataframe(
        df=df,
        company_name="Test Company",
        industry="Food",
        source_filename="test.csv",
        data_quality_score=100,
    )

    loaded_df = repository.load_inventory_records(batch_id=batch_id)

    assert len(loaded_df) == 2
    assert loaded_df["product_name"].iloc[0] == "Veggie Burger"
    assert loaded_df["store_id"].iloc[0] == "S001"


def test_repository_creates_upload_batch(tmp_path):
    db_path = tmp_path / "test_stocksense.db"
    repository = InventoryRepository(db_path=str(db_path))
    repository.setup()

    df = create_repository_test_df()

    batch_id = repository.save_inventory_dataframe(
        df=df,
        company_name="Test Company",
        industry="Food",
        source_filename="test.csv",
        data_quality_score=95,
    )

    batches_df = repository.get_upload_batches()

    assert batch_id in batches_df["batch_id"].values
    assert batches_df["company_name"].iloc[0] == "Test Company"
    assert batches_df["data_quality_score"].iloc[0] == 95


def test_repository_loads_latest_batch_when_no_batch_id_is_given(tmp_path):
    db_path = tmp_path / "test_stocksense.db"
    repository = InventoryRepository(db_path=str(db_path))
    repository.setup()

    df = create_repository_test_df()

    repository.save_inventory_dataframe(
        df=df,
        company_name="Test Company",
        industry="Food",
        source_filename="first.csv",
        data_quality_score=100,
    )

    latest_batch_id = repository.save_inventory_dataframe(
        df=df,
        company_name="Test Company",
        industry="Food",
        source_filename="second.csv",
        data_quality_score=100,
    )

    latest_df = repository.load_inventory_records()

    batches_df = repository.get_upload_batches()
    assert batches_df["batch_id"].iloc[0] == latest_batch_id
    assert len(latest_df) == 2


def test_repository_returns_empty_dataframe_when_no_data_exists(tmp_path):
    db_path = tmp_path / "test_stocksense.db"
    repository = InventoryRepository(db_path=str(db_path))
    repository.setup()

    loaded_df = repository.load_inventory_records()

    assert loaded_df.empty