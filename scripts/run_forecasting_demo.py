from src.analytics.kpi_engine import calculate_inventory_kpis
from src.data.validation import validate_inventory_file
from src.models.forecasting_engine import (
    forecasts_to_dataframe,
    generate_product_forecasts,
    summarize_forecasts,
)


def main() -> None:
    file_path = "data/sample/sample_inventory.csv"

    validation_result = validate_inventory_file(file_path)

    if not validation_result.is_valid:
        print("Data validation failed.")
        for error in validation_result.errors:
            print(f"- {error}")
        return

    kpi_result = calculate_inventory_kpis(validation_result.cleaned_data)

    forecasts = generate_product_forecasts(
        enriched_data=kpi_result["enriched_data"],
        product_performance=kpi_result["product_performance"],
        forecast_horizon_days=30,
    )

    forecast_df = forecasts_to_dataframe(forecasts)
    forecast_summary = summarize_forecasts(forecast_df)

    print("===== Forecast Summary =====")
    for key, value in forecast_summary.items():
        print(f"{key}: {value}")

    print("\n===== Product Forecasts =====")

    display_columns = [
        "product_name",
        "category",
        "current_stock",
        "predicted_demand",
        "safety_stock",
        "recommended_reorder_quantity",
        "trend_direction",
        "forecast_stockout_risk",
        "confidence",
    ]

    print(forecast_df[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()