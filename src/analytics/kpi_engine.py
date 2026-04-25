from typing import Any, Dict

import numpy as np
import pandas as pd

from src.data.validation import REQUIRED_COLUMNS, standardize_column_names


def _ensure_required_columns(df: pd.DataFrame) -> None:
    """
    Ensure the required inventory columns exist before KPI calculation.
    """

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns for KPI calculation: {missing_columns}")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Safely divide two pandas Series.

    If denominator is zero, return 0 instead of infinity or error.
    """

    denominator = denominator.replace(0, np.nan)
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan).fillna(0)


def prepare_kpi_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare clean inventory data for KPI calculation.

    This function adds row-level business metrics such as revenue,
    waste value, available stock, sell-through rate, waste rate,
    average inventory, and stock turnover.
    """

    prepared_df = standardize_column_names(df)
    _ensure_required_columns(prepared_df)

    prepared_df = prepared_df.copy()

    prepared_df["date"] = pd.to_datetime(prepared_df["date"], errors="coerce")

    numeric_columns = [
        "opening_stock",
        "purchased_quantity",
        "sold_quantity",
        "wasted_quantity",
        "closing_stock",
        "unit_price",
    ]

    for col in numeric_columns:
        prepared_df[col] = pd.to_numeric(prepared_df[col], errors="coerce").fillna(0)

    prepared_df["available_stock"] = (
        prepared_df["opening_stock"] + prepared_df["purchased_quantity"]
    )

    prepared_df["revenue"] = prepared_df["sold_quantity"] * prepared_df["unit_price"]

    prepared_df["waste_value"] = (
        prepared_df["wasted_quantity"] * prepared_df["unit_price"]
    )

    prepared_df["closing_inventory_value"] = (
        prepared_df["closing_stock"] * prepared_df["unit_price"]
    )

    prepared_df["average_inventory"] = (
        prepared_df["opening_stock"] + prepared_df["closing_stock"]
    ) / 2

    prepared_df["sell_through_rate"] = _safe_divide(
        prepared_df["sold_quantity"], prepared_df["available_stock"]
    )

    prepared_df["waste_rate"] = _safe_divide(
        prepared_df["wasted_quantity"],
        prepared_df["sold_quantity"] + prepared_df["wasted_quantity"],
    )

    prepared_df["stock_turnover"] = _safe_divide(
        prepared_df["sold_quantity"], prepared_df["average_inventory"]
    )

    return prepared_df


def get_latest_inventory_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the latest inventory record for each store-product combination.

    This prevents overcounting inventory value across multiple dates.
    """

    latest_df = df.sort_values("date").drop_duplicates(
        subset=["store_id", "product_id"], keep="last"
    )

    return latest_df


def calculate_product_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate product-level performance metrics.

    Output includes revenue, units sold, waste, current stock,
    estimated days of stock left, stockout risk, overstock risk,
    and product health score.
    """

    product_group_cols = ["product_id", "product_name", "category"]

    product_metrics = (
        df.groupby(product_group_cols)
        .agg(
            total_revenue=("revenue", "sum"),
            total_units_sold=("sold_quantity", "sum"),
            total_units_wasted=("wasted_quantity", "sum"),
            total_waste_value=("waste_value", "sum"),
            total_available_stock=("available_stock", "sum"),
            average_inventory=("average_inventory", "mean"),
            avg_unit_price=("unit_price", "mean"),
            active_days=("date", "nunique"),
        )
        .reset_index()
    )

    latest_snapshot = get_latest_inventory_snapshot(df)

    latest_stock = (
        latest_snapshot.groupby(product_group_cols)
        .agg(current_stock=("closing_stock", "sum"))
        .reset_index()
    )

    product_metrics = product_metrics.merge(
        latest_stock, on=product_group_cols, how="left"
    )

    product_metrics["sell_through_rate"] = _safe_divide(
        product_metrics["total_units_sold"],
        product_metrics["total_available_stock"],
    )

    product_metrics["waste_rate"] = _safe_divide(
        product_metrics["total_units_wasted"],
        product_metrics["total_units_sold"] + product_metrics["total_units_wasted"],
    )

    product_metrics["stock_turnover"] = _safe_divide(
        product_metrics["total_units_sold"],
        product_metrics["average_inventory"],
    )

    product_metrics["avg_daily_demand"] = _safe_divide(
        product_metrics["total_units_sold"],
        product_metrics["active_days"],
    )

    product_metrics["days_of_stock_left"] = np.where(
        product_metrics["avg_daily_demand"] > 0,
        product_metrics["current_stock"] / product_metrics["avg_daily_demand"],
        np.inf,
    )

    product_metrics["stockout_risk"] = product_metrics["days_of_stock_left"].apply(
        _classify_stockout_risk
    )

    product_metrics["overstock_risk"] = product_metrics.apply(
        _classify_overstock_risk, axis=1
    )

    product_metrics["product_health_score"] = product_metrics.apply(
        _calculate_product_health_score, axis=1
    )

    product_metrics = product_metrics.sort_values(
        by="total_revenue", ascending=False
    ).reset_index(drop=True)

    return product_metrics


def _classify_stockout_risk(days_of_stock_left: float) -> str:
    """
    Classify stockout risk based on estimated days of stock left.
    """

    if np.isinf(days_of_stock_left):
        return "Low"

    if days_of_stock_left <= 7:
        return "High"

    if days_of_stock_left <= 14:
        return "Medium"

    return "Low"


def _classify_overstock_risk(row: pd.Series) -> str:
    """
    Classify overstock risk based on current stock and demand speed.
    """

    avg_daily_demand = row["avg_daily_demand"]
    current_stock = row["current_stock"]
    days_of_stock_left = row["days_of_stock_left"]

    if avg_daily_demand == 0 and current_stock > 0:
        return "High"

    if days_of_stock_left > 45:
        return "High"

    if days_of_stock_left > 30:
        return "Medium"

    return "Low"


def _calculate_product_health_score(row: pd.Series) -> int:
    """
    Calculate a simple product health score from 0 to 100.

    Higher score means the product is healthier from an inventory perspective.
    """

    score = 100

    if row["stockout_risk"] == "High":
        score -= 25
    elif row["stockout_risk"] == "Medium":
        score -= 12

    if row["overstock_risk"] == "High":
        score -= 20
    elif row["overstock_risk"] == "Medium":
        score -= 10

    if row["waste_rate"] > 0.10:
        score -= 15
    elif row["waste_rate"] > 0.05:
        score -= 7

    if row["sell_through_rate"] < 0.20:
        score -= 10

    return max(min(int(score), 100), 0)


def calculate_category_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate category-level performance metrics.
    """

    category_metrics = (
        df.groupby("category")
        .agg(
            total_revenue=("revenue", "sum"),
            total_units_sold=("sold_quantity", "sum"),
            total_units_wasted=("wasted_quantity", "sum"),
            total_waste_value=("waste_value", "sum"),
            total_available_stock=("available_stock", "sum"),
            product_count=("product_id", "nunique"),
        )
        .reset_index()
    )

    category_metrics["sell_through_rate"] = _safe_divide(
        category_metrics["total_units_sold"],
        category_metrics["total_available_stock"],
    )

    category_metrics["waste_rate"] = _safe_divide(
        category_metrics["total_units_wasted"],
        category_metrics["total_units_sold"]
        + category_metrics["total_units_wasted"],
    )

    category_metrics = category_metrics.sort_values(
        by="total_revenue", ascending=False
    ).reset_index(drop=True)

    return category_metrics


def calculate_risk_summary(product_performance: pd.DataFrame) -> Dict[str, int]:
    """
    Summarize product risk counts.
    """

    return {
        "high_stockout_risk_products": int(
            (product_performance["stockout_risk"] == "High").sum()
        ),
        "medium_stockout_risk_products": int(
            (product_performance["stockout_risk"] == "Medium").sum()
        ),
        "high_overstock_risk_products": int(
            (product_performance["overstock_risk"] == "High").sum()
        ),
        "medium_overstock_risk_products": int(
            (product_performance["overstock_risk"] == "Medium").sum()
        ),
        "high_waste_products": int((product_performance["waste_rate"] > 0.10).sum()),
    }


def calculate_inventory_health_score(
    summary_metrics: Dict[str, Any],
    risk_summary: Dict[str, int],
) -> int:
    """
    Calculate overall inventory health score from 0 to 100.
    """

    score = 100

    avg_sell_through_rate = summary_metrics["average_sell_through_rate"]
    avg_waste_rate = summary_metrics["average_waste_rate"]
    product_count = max(summary_metrics["unique_products"], 1)

    if avg_sell_through_rate < 0.30:
        score -= 15
    elif avg_sell_through_rate < 0.50:
        score -= 8

    score -= min(int(avg_waste_rate * 200), 25)

    score -= int(
        (risk_summary["high_stockout_risk_products"] / product_count) * 25
    )

    score -= int(
        (risk_summary["high_overstock_risk_products"] / product_count) * 20
    )

    score -= int((risk_summary["high_waste_products"] / product_count) * 15)

    return max(min(int(score), 100), 0)


def calculate_summary_metrics(
    df: pd.DataFrame,
    risk_summary: Dict[str, int],
) -> Dict[str, Any]:
    """
    Calculate overall business summary KPIs.
    """

    latest_snapshot = get_latest_inventory_snapshot(df)

    total_available_stock = df["available_stock"].sum()
    total_units_sold = df["sold_quantity"].sum()
    total_units_wasted = df["wasted_quantity"].sum()

    summary = {
        "date_start": str(df["date"].min().date()),
        "date_end": str(df["date"].max().date()),
        "unique_stores": int(df["store_id"].nunique()),
        "unique_products": int(df["product_id"].nunique()),
        "total_revenue": round(float(df["revenue"].sum()), 2),
        "total_units_sold": int(total_units_sold),
        "total_units_wasted": int(total_units_wasted),
        "total_waste_value": round(float(df["waste_value"].sum()), 2),
        "latest_inventory_value": round(
            float(latest_snapshot["closing_inventory_value"].sum()), 2
        ),
        "average_sell_through_rate": round(
            float(total_units_sold / total_available_stock)
            if total_available_stock > 0
            else 0,
            4,
        ),
        "average_waste_rate": round(
            float(total_units_wasted / (total_units_sold + total_units_wasted))
            if (total_units_sold + total_units_wasted) > 0
            else 0,
            4,
        ),
    }

    summary["inventory_health_score"] = calculate_inventory_health_score(
        summary, risk_summary
    )

    return summary


def calculate_inventory_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Main function to calculate all inventory KPIs.

    Returns:
        Dictionary containing:
        - summary_metrics
        - product_performance
        - category_performance
        - risk_summary
        - enriched_data
    """

    enriched_data = prepare_kpi_data(df)

    product_performance = calculate_product_performance(enriched_data)
    category_performance = calculate_category_performance(enriched_data)
    risk_summary = calculate_risk_summary(product_performance)
    summary_metrics = calculate_summary_metrics(enriched_data, risk_summary)

    return {
        "summary_metrics": summary_metrics,
        "product_performance": product_performance,
        "category_performance": category_performance,
        "risk_summary": risk_summary,
        "enriched_data": enriched_data,
    }