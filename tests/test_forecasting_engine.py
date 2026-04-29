import pandas as pd

from src.analytics.kpi_engine import calculate_inventory_kpis
from src.models.forecasting_engine import (
    forecasts_to_dataframe,
    generate_product_forecasts,
    summarize_forecasts,
)


def create_forecasting_test_df():
    rows = []

    dates = pd.date_range(start="2025-10-01", periods=40, freq="D")

    for index, date in enumerate(dates):
        sold_quantity = 10 + (index // 10)

        rows.append(
            {
                "date": date,
                "store_id": "S001",
                "product_id": "P001",
                "product_name": "Veggie Burger",
                "category": "Food",
                "opening_stock": 500 - index * 5,
                "purchased_quantity": 0,
                "sold_quantity": sold_quantity,
                "wasted_quantity": 1,
                "closing_stock": 500 - index * 5 - sold_quantity - 1,
                "unit_price": 5.0,
            }
        )

    return pd.DataFrame(rows)


def test_generate_product_forecasts_returns_forecasts():
    df = create_forecasting_test_df()
    kpi_result = calculate_inventory_kpis(df)

    forecasts = generate_product_forecasts(
        enriched_data=kpi_result["enriched_data"],
        product_performance=kpi_result["product_performance"],
        forecast_horizon_days=30,
    )

    assert len(forecasts) == 1
    assert forecasts[0].product_name == "Veggie Burger"


def test_forecast_contains_predicted_demand_and_reorder_quantity():
    df = create_forecasting_test_df()
    kpi_result = calculate_inventory_kpis(df)

    forecasts = generate_product_forecasts(
        enriched_data=kpi_result["enriched_data"],
        product_performance=kpi_result["product_performance"],
        forecast_horizon_days=30,
    )

    forecast = forecasts[0]

    assert forecast.predicted_demand >= 0
    assert forecast.safety_stock >= 0
    assert forecast.recommended_reorder_quantity >= 0


def test_forecasts_to_dataframe_has_expected_columns():
    df = create_forecasting_test_df()
    kpi_result = calculate_inventory_kpis(df)

    forecasts = generate_product_forecasts(
        enriched_data=kpi_result["enriched_data"],
        product_performance=kpi_result["product_performance"],
        forecast_horizon_days=30,
    )

    forecast_df = forecasts_to_dataframe(forecasts)

    expected_columns = [
        "product_name",
        "predicted_demand",
        "safety_stock",
        "recommended_reorder_quantity",
        "trend_direction",
        "forecast_stockout_risk",
        "confidence",
    ]

    for column in expected_columns:
        assert column in forecast_df.columns


def test_summarize_forecasts_returns_expected_keys():
    df = create_forecasting_test_df()
    kpi_result = calculate_inventory_kpis(df)

    forecasts = generate_product_forecasts(
        enriched_data=kpi_result["enriched_data"],
        product_performance=kpi_result["product_performance"],
        forecast_horizon_days=30,
    )

    forecast_df = forecasts_to_dataframe(forecasts)
    summary = summarize_forecasts(forecast_df)

    assert "total_products_forecasted" in summary
    assert "total_predicted_demand" in summary
    assert "total_recommended_reorder" in summary


def test_forecast_horizon_changes_prediction():
    df = create_forecasting_test_df()
    kpi_result = calculate_inventory_kpis(df)

    forecast_7 = generate_product_forecasts(
        enriched_data=kpi_result["enriched_data"],
        product_performance=kpi_result["product_performance"],
        forecast_horizon_days=7,
    )[0]

    forecast_30 = generate_product_forecasts(
        enriched_data=kpi_result["enriched_data"],
        product_performance=kpi_result["product_performance"],
        forecast_horizon_days=30,
    )[0]

    assert forecast_30.predicted_demand > forecast_7.predicted_demand