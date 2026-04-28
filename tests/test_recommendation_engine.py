import pandas as pd

from src.recommendations.recommendation_engine import (
    generate_recommendations,
    recommendations_to_dataframe,
    summarize_recommendations,
)


def create_product_performance_df():
    return pd.DataFrame(
        {
            "product_id": ["P001", "P002", "P003"],
            "product_name": ["Veggie Burger", "Cola 500ml", "French Fries"],
            "category": ["Food", "Beverage", "Food"],
            "total_revenue": [1000.0, 500.0, 700.0],
            "total_units_sold": [200, 50, 100],
            "total_units_wasted": [5, 20, 2],
            "total_waste_value": [25.0, 40.0, 6.0],
            "total_available_stock": [300, 500, 150],
            "average_inventory": [100.0, 300.0, 70.0],
            "avg_unit_price": [5.0, 2.0, 3.0],
            "active_days": [30, 30, 30],
            "current_stock": [20, 500, 80],
            "sell_through_rate": [0.66, 0.10, 0.67],
            "waste_rate": [0.02, 0.29, 0.02],
            "stock_turnover": [2.0, 0.16, 1.4],
            "avg_daily_demand": [6.67, 1.67, 3.33],
            "days_of_stock_left": [3.0, 299.4, 24.0],
            "stockout_risk": ["High", "Low", "Low"],
            "overstock_risk": ["Low", "High", "Low"],
            "product_health_score": [70, 40, 90],
        }
    )


def test_generate_recommendations_returns_list():
    df = create_product_performance_df()

    recommendations = generate_recommendations(df)

    assert isinstance(recommendations, list)
    assert len(recommendations) > 0


def test_high_stockout_generates_stockout_recommendation():
    df = create_product_performance_df()

    recommendations = generate_recommendations(df)

    assert any(
        rec.product_name == "Veggie Burger"
        and rec.recommendation_type == "Stockout Prevention"
        for rec in recommendations
    )


def test_high_overstock_generates_overstock_recommendation():
    df = create_product_performance_df()

    recommendations = generate_recommendations(df)

    assert any(
        rec.product_name == "Cola 500ml"
        and rec.recommendation_type == "Overstock Reduction"
        for rec in recommendations
    )


def test_high_waste_generates_waste_recommendation():
    df = create_product_performance_df()

    recommendations = generate_recommendations(df)

    assert any(
        rec.product_name == "Cola 500ml"
        and rec.recommendation_type == "Waste Reduction"
        for rec in recommendations
    )


def test_recommendations_to_dataframe_has_expected_columns():
    df = create_product_performance_df()

    recommendations = generate_recommendations(df)
    recommendation_df = recommendations_to_dataframe(recommendations)

    expected_columns = [
        "priority",
        "product_name",
        "category",
        "recommendation_type",
        "action",
        "reason",
        "confidence",
    ]

    for column in expected_columns:
        assert column in recommendation_df.columns


def test_summarize_recommendations_counts_priorities():
    df = create_product_performance_df()

    recommendations = generate_recommendations(df)
    summary = summarize_recommendations(recommendations)

    assert "total_recommendations" in summary
    assert "high_priority" in summary
    assert summary["total_recommendations"] == len(recommendations)