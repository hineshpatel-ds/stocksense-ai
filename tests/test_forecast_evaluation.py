import pandas as pd
import pytest

from src.analytics.kpi_engine import calculate_inventory_kpis
from src.models.forecast_evaluation import (
    evaluate_moving_average_forecast,
    get_evaluation_metrics_dict,
    prepare_daily_demand,
)


def create_evaluation_test_df():
    rows = []
    dates = pd.date_range(start="2025-10-01", periods=40, freq="D")

    for index, date in enumerate(dates):
        sold_quantity = 10 + (index % 5)

        rows.append(
            {
                "date": date,
                "store_id": "S001",
                "product_id": "P001",
                "product_name": "Veggie Burger",
                "category": "Food",
                "opening_stock": 1000 - index * 10,
                "purchased_quantity": 0,
                "sold_quantity": sold_quantity,
                "wasted_quantity": 1,
                "closing_stock": 1000 - index * 10 - sold_quantity - 1,
                "unit_price": 5.0,
            }
        )

    return pd.DataFrame(rows)


def test_prepare_daily_demand_returns_expected_columns():
    df = create_evaluation_test_df()
    kpi_result = calculate_inventory_kpis(df)

    daily_demand = prepare_daily_demand(kpi_result["enriched_data"])

    expected_columns = [
        "date",
        "product_id",
        "product_name",
        "category",
        "actual_demand",
    ]

    for column in expected_columns:
        assert column in daily_demand.columns


def test_evaluate_moving_average_forecast_returns_metrics():
    df = create_evaluation_test_df()
    kpi_result = calculate_inventory_kpis(df)

    result = evaluate_moving_average_forecast(
        enriched_data=kpi_result["enriched_data"],
        recent_window_days=7,
        test_window_days=7,
    )

    assert result.evaluated_products == 1
    assert result.mae >= 0
    assert result.rmse >= 0
    assert result.mape >= 0
    assert not result.evaluation_data.empty


def test_get_evaluation_metrics_dict_returns_expected_keys():
    df = create_evaluation_test_df()
    kpi_result = calculate_inventory_kpis(df)

    result = evaluate_moving_average_forecast(
        enriched_data=kpi_result["enriched_data"],
        recent_window_days=7,
        test_window_days=7,
    )

    metrics = get_evaluation_metrics_dict(result)

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "mape" in metrics
    assert "evaluated_products" in metrics


def test_invalid_recent_window_raises_error():
    df = create_evaluation_test_df()
    kpi_result = calculate_inventory_kpis(df)

    with pytest.raises(ValueError):
        evaluate_moving_average_forecast(
            enriched_data=kpi_result["enriched_data"],
            recent_window_days=0,
            test_window_days=7,
        )


def test_invalid_test_window_raises_error():
    df = create_evaluation_test_df()
    kpi_result = calculate_inventory_kpis(df)

    with pytest.raises(ValueError):
        evaluate_moving_average_forecast(
            enriched_data=kpi_result["enriched_data"],
            recent_window_days=7,
            test_window_days=0,
        )