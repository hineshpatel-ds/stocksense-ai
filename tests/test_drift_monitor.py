import pandas as pd

from src.analytics.kpi_engine import calculate_inventory_kpis
from src.monitoring.drift_monitor import (
    calculate_population_stability_index,
    classify_demand_shift,
    classify_drift_score,
    monitor_inventory_drift,
    split_reference_current_periods,
)


def create_monitoring_test_df():
    rows = []
    dates = pd.date_range(start="2025-10-01", periods=70, freq="D")

    for index, date in enumerate(dates):
        if index < 40:
            sold_quantity = 10
        else:
            sold_quantity = 25

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


def test_classify_drift_score():
    assert classify_drift_score(0.05) == "Low"
    assert classify_drift_score(0.15) == "Medium"
    assert classify_drift_score(0.25) == "High"


def test_classify_demand_shift():
    assert classify_demand_shift(0.30) == "Strong Increase"
    assert classify_demand_shift(0.15) == "Moderate Increase"
    assert classify_demand_shift(0.03) == "Stable"
    assert classify_demand_shift(-0.15) == "Moderate Decrease"
    assert classify_demand_shift(-0.30) == "Strong Decrease"


def test_calculate_population_stability_index_returns_number():
    reference = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
    current = pd.Series([5, 6, 7, 8, 9, 10, 11, 12])

    score = calculate_population_stability_index(reference, current)

    assert isinstance(score, float)
    assert score >= 0


def test_split_reference_current_periods():
    df = create_monitoring_test_df()
    kpi_result = calculate_inventory_kpis(df)

    reference_df, current_df = split_reference_current_periods(
        kpi_result["enriched_data"],
        current_window_days=30,
    )

    assert not reference_df.empty
    assert not current_df.empty
    assert current_df["date"].min() > reference_df["date"].max()


def test_monitor_inventory_drift_returns_summary():
    df = create_monitoring_test_df()
    kpi_result = calculate_inventory_kpis(df)

    result = monitor_inventory_drift(
        enriched_data=kpi_result["enriched_data"],
        current_window_days=30,
    )

    assert "monitoring_status" in result.summary
    assert "high_drift_features" in result.summary
    assert not result.numeric_drift.empty
    assert not result.demand_shift.empty


def test_monitor_inventory_drift_handles_insufficient_data():
    df = create_monitoring_test_df().head(10)
    kpi_result = calculate_inventory_kpis(df)

    result = monitor_inventory_drift(
        enriched_data=kpi_result["enriched_data"],
        current_window_days=30,
    )

    assert result.summary["monitoring_status"] == "Insufficient Data"