from src.analytics.kpi_engine import calculate_inventory_kpis
from src.data.validation import validate_inventory_file
from src.database.inventory_repository import InventoryRepository


def main() -> None:
    """
    Demonstrate saving and loading inventory data from SQLite.
    """

    file_path = "data/sample/sample_inventory.csv"

    validation_result = validate_inventory_file(file_path)

    if not validation_result.is_valid:
        print("Data validation failed.")
        for error in validation_result.errors:
            print(f"- {error}")
        return

    repository = InventoryRepository()
    repository.setup()

    batch_id = repository.save_inventory_dataframe(
        df=validation_result.cleaned_data,
        company_name="Demo Company",
        industry="General Inventory",
        source_filename=file_path,
        data_quality_score=validation_result.data_quality_score,
    )

    print(f"Saved inventory upload to database.")
    print(f"Batch ID: {batch_id}")

    loaded_df = repository.load_inventory_records(batch_id=batch_id)

    print(f"Loaded {len(loaded_df)} records from database.")

    kpi_result = calculate_inventory_kpis(loaded_df)

    print("\n===== KPI Summary from Database Records =====")
    for key, value in kpi_result["summary_metrics"].items():
        print(f"{key}: {value}")

    print("\n===== Recent Upload Batches =====")
    print(repository.get_upload_batches().to_string(index=False))


if __name__ == "__main__":
    main()