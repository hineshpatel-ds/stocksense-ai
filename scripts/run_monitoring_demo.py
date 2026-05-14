from src.analytics.kpi_engine import calculate_inventory_kpis
from src.data.validation import validate_inventory_file
from src.monitoring.drift_monitor import monitor_inventory_drift


def main() -> None:
    """
    Run inventory drift monitoring demo.
    """

    file_path = "data/sample/sample_inventory.csv"

    validation_result = validate_inventory_file(file_path)

    if not validation_result.is_valid:
        print("Data validation failed.")
        for error in validation_result.errors:
            print(f"- {error}")
        return

    kpi_result = calculate_inventory_kpis(validation_result.cleaned_data)

    monitoring_result = monitor_inventory_drift(
        enriched_data=kpi_result["enriched_data"],
        current_window_days=30,
    )

    print("===== Monitoring Summary =====")
    for key, value in monitoring_result.summary.items():
        print(f"{key}: {value}")

    print("\n===== Numeric Drift =====")
    print(monitoring_result.numeric_drift.to_string(index=False))

    print("\n===== Categorical Drift =====")
    print(monitoring_result.categorical_drift.to_string(index=False))

    print("\n===== Product Demand Shift =====")
    print(monitoring_result.demand_shift.head(10).to_string(index=False))


if __name__ == "__main__":
    main()