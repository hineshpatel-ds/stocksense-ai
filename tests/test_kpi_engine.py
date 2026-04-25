import pandas as pd

from src.analytics.kpi_engine import calculate_inventory_kpis, prepare_kpi_data


def create_test_inventory_df():
    return pd.DataFrame(
        {
            "date": ["2025-10-01", "2025-10-02", "2025-10-01"],
            "store_id": ["S001", "S001", "S001"],
            "product_id": ["P001", "P001", "P002"],
            "product_name": ["Veggie Burger", "Veggie Burger", "Cola 500ml"],
            "category": ["Food", "Food", "Beverage"],
            "opening_stock": [100, 85, 50],
            "purchased_quantity": [20, 0, 10],
            "sold_quantity": [30, 20, 5],
            "wasted_quantity": [5, 0, 0],
            "closing_stock": [85, 65, 55],
            "unit_price": [10.0, 10.0, 5.0],
        }
    )


def test_prepare_kpi_data_adds_expected_columns():
    df = create_test_inventory_df()

    prepared_df = prepare_kpi_data(df)

    expected_columns = [
        "available_stock",
        "revenue",
        "waste_value",
        "closing_inventory_value",
        "average_inventory",
        "sell_through_rate",
        "waste_rate",
        "stock_turnover",
    ]

    for column in expected_columns:
        assert column in prepared_df.columns


def test_summary_metrics_are_calculated_correctly():
    df = create_test_inventory_df()

    result = calculate_inventory_kpis(df)

    summary = result["summary_metrics"]

    assert summary["total_revenue"] == 525.0
    assert summary["total_units_sold"] == 55
    assert summary["total_units_wasted"] == 5
    assert summary["total_waste_value"] == 50.0


def test_latest_inventory_value_uses_latest_snapshot_not_all_rows():
    df = create_test_inventory_df()

    result = calculate_inventory_kpis(df)

    summary = result["summary_metrics"]

    # Latest inventory:
    # Veggie Burger latest closing stock = 65 × 10 = 650
    # Cola latest closing stock = 55 × 5 = 275
    # Total = 925
    assert summary["latest_inventory_value"] == 925.0


def test_product_performance_contains_risk_columns():
    df = create_test_inventory_df()

    result = calculate_inventory_kpis(df)

    product_performance = result["product_performance"]

    assert "stockout_risk" in product_performance.columns
    assert "overstock_risk" in product_performance.columns
    assert "product_health_score" in product_performance.columns


def test_category_performance_is_generated():
    df = create_test_inventory_df()

    result = calculate_inventory_kpis(df)

    category_performance = result["category_performance"]

    assert len(category_performance) == 2
    assert "Food" in category_performance["category"].values
    assert "Beverage" in category_performance["category"].values


def test_inventory_health_score_is_between_zero_and_hundred():
    df = create_test_inventory_df()

    result = calculate_inventory_kpis(df)

    score = result["summary_metrics"]["inventory_health_score"]

    assert 0 <= score <= 100