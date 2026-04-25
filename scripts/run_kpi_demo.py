from src.analytics.kpi_engine import calculate_inventory_kpis
from src.data.validation import validate_inventory_file


def main() -> None:
    """
    Run KPI analytics on the sample inventory dataset.
    """

    file_path = "data/sample/sample_inventory.csv"

    validation_result = validate_inventory_file(file_path)

    if not validation_result.is_valid:
        print("Data validation failed.")
        print("Errors:")
        for error in validation_result.errors:
            print(f"- {error}")
        return

    print("Data validation passed.")
    print(f"Data quality score: {validation_result.data_quality_score}/100")

    kpi_result = calculate_inventory_kpis(validation_result.cleaned_data)

    print("\n===== Summary Metrics =====")
    for key, value in kpi_result["summary_metrics"].items():
        print(f"{key}: {value}")

    print("\n===== Risk Summary =====")
    for key, value in kpi_result["risk_summary"].items():
        print(f"{key}: {value}")

    print("\n===== Top 5 Products by Revenue =====")
    top_products = kpi_result["product_performance"][
        [
            "product_name",
            "category",
            "total_revenue",
            "total_units_sold",
            "current_stock",
            "stockout_risk",
            "overstock_risk",
            "product_health_score",
        ]
    ].head(5)

    print(top_products.to_string(index=False))


if __name__ == "__main__":
    main()