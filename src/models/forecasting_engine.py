from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import List

import numpy as np
import pandas as pd


@dataclass
class ProductForecast:
    """
    Represents one product demand forecast.
    """

    product_id: str
    product_name: str
    category: str
    forecast_horizon_days: int
    current_stock: int
    avg_daily_demand: float
    recent_avg_daily_demand: float
    predicted_demand: int
    safety_stock: int
    recommended_reorder_quantity: int
    trend_direction: str
    trend_percentage: float
    stock_coverage_days: float
    forecast_stockout_risk: str
    confidence: str


def _safe_divide(numerator: float, denominator: float) -> float:
    """
    Safely divide two numbers.
    """

    if denominator == 0:
        return 0.0

    return numerator / denominator


def _classify_trend(trend_percentage: float) -> str:
    """
    Classify demand trend.
    """

    if trend_percentage >= 0.10:
        return "Increasing"

    if trend_percentage <= -0.10:
        return "Decreasing"

    return "Stable"


def _classify_forecast_stockout_risk(
    current_stock: int,
    predicted_demand: int,
    safety_stock: int,
) -> str:
    """
    Classify future stockout risk using predicted demand and safety stock.
    """

    required_stock = predicted_demand + safety_stock

    if current_stock < predicted_demand:
        return "High"

    if current_stock < required_stock:
        return "Medium"

    return "Low"


def _classify_confidence(history_days: int, demand_volatility: float) -> str:
    """
    Classify forecast confidence.

    More history and lower volatility means higher confidence.
    """

    if history_days >= 60 and demand_volatility <= 0.40:
        return "High"

    if history_days >= 30 and demand_volatility <= 0.75:
        return "Medium"

    return "Low"


def _prepare_daily_product_demand(enriched_data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare daily demand per product.

    Demand is represented by sold_quantity.
    """

    required_columns = [
        "date",
        "product_id",
        "product_name",
        "category",
        "sold_quantity",
    ]

    missing_columns = [col for col in required_columns if col not in enriched_data.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns for forecasting: {missing_columns}")

    data = enriched_data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    daily_demand = (
        data.groupby(["date", "product_id", "product_name", "category"], as_index=False)
        .agg(sold_quantity=("sold_quantity", "sum"))
        .sort_values(["product_id", "date"])
    )

    return daily_demand


def generate_product_forecasts(
    enriched_data: pd.DataFrame,
    product_performance: pd.DataFrame,
    forecast_horizon_days: int = 30,
    recent_window_days: int = 14,
) -> List[ProductForecast]:
    """
    Generate product-level demand forecasts.

    This is an explainable baseline forecasting engine using:
    - Historical average daily demand
    - Recent moving average demand
    - Previous period comparison
    - Trend adjustment
    - Demand volatility
    - Safety stock
    - Current stock position
    """

    daily_demand = _prepare_daily_product_demand(enriched_data)

    forecasts: List[ProductForecast] = []

    for product_id, product_history in daily_demand.groupby("product_id"):
        product_history = product_history.sort_values("date").copy()

        min_date = product_history["date"].min()
        max_date = product_history["date"].max()

        full_dates = pd.date_range(start=min_date, end=max_date, freq="D")

        product_history = (
            product_history.set_index("date")
            .reindex(full_dates)
            .rename_axis("date")
            .reset_index()
        )

        product_history["product_id"] = product_history["product_id"].ffill().bfill()
        product_history["product_name"] = product_history["product_name"].ffill().bfill()
        product_history["category"] = product_history["category"].ffill().bfill()
        product_history["sold_quantity"] = product_history["sold_quantity"].fillna(0)

        product_name = str(product_history["product_name"].iloc[0])
        category = str(product_history["category"].iloc[0])

        history_days = len(product_history)
        all_demand = product_history["sold_quantity"]

        recent_demand = all_demand.tail(recent_window_days)
        previous_demand = all_demand.tail(recent_window_days * 2).head(recent_window_days)

        avg_daily_demand = float(all_demand.mean())
        recent_avg_daily_demand = float(recent_demand.mean())
        previous_avg_daily_demand = float(previous_demand.mean()) if len(previous_demand) > 0 else avg_daily_demand

        trend_percentage = _safe_divide(
            recent_avg_daily_demand - previous_avg_daily_demand,
            previous_avg_daily_demand,
        )

        trend_direction = _classify_trend(trend_percentage)

        # Limit trend effect so one unusual period does not overinflate forecast.
        capped_trend_adjustment = max(min(trend_percentage, 0.25), -0.25)

        forecast_daily_demand = max(
            recent_avg_daily_demand * (1 + capped_trend_adjustment),
            0,
        )

        predicted_demand = int(round(forecast_daily_demand * forecast_horizon_days))

        demand_std = float(all_demand.std(ddof=0))
        safety_stock = int(round(demand_std * sqrt(forecast_horizon_days)))

        performance_row = product_performance[
            product_performance["product_id"] == product_id
        ]

        if performance_row.empty:
            current_stock = 0
        else:
            current_stock = int(performance_row["current_stock"].iloc[0])

        stock_coverage_days = (
            current_stock / recent_avg_daily_demand
            if recent_avg_daily_demand > 0
            else float("inf")
        )

        demand_volatility = _safe_divide(demand_std, avg_daily_demand)

        forecast_stockout_risk = _classify_forecast_stockout_risk(
            current_stock=current_stock,
            predicted_demand=predicted_demand,
            safety_stock=safety_stock,
        )

        confidence = _classify_confidence(
            history_days=history_days,
            demand_volatility=demand_volatility,
        )

        recommended_reorder_quantity = max(
            predicted_demand + safety_stock - current_stock,
            0,
        )

        forecasts.append(
            ProductForecast(
                product_id=str(product_id),
                product_name=product_name,
                category=category,
                forecast_horizon_days=forecast_horizon_days,
                current_stock=current_stock,
                avg_daily_demand=round(avg_daily_demand, 2),
                recent_avg_daily_demand=round(recent_avg_daily_demand, 2),
                predicted_demand=predicted_demand,
                safety_stock=safety_stock,
                recommended_reorder_quantity=int(recommended_reorder_quantity),
                trend_direction=trend_direction,
                trend_percentage=round(float(trend_percentage), 4),
                stock_coverage_days=round(float(stock_coverage_days), 2)
                if not np.isinf(stock_coverage_days)
                else float("inf"),
                forecast_stockout_risk=forecast_stockout_risk,
                confidence=confidence,
            )
        )

    risk_order = {"High": 1, "Medium": 2, "Low": 3}

    forecasts = sorted(
        forecasts,
        key=lambda forecast: (
            risk_order.get(forecast.forecast_stockout_risk, 99),
            -forecast.recommended_reorder_quantity,
        ),
    )

    return forecasts


def forecasts_to_dataframe(forecasts: List[ProductForecast]) -> pd.DataFrame:
    """
    Convert product forecasts into a DataFrame.
    """

    rows = []

    for forecast in forecasts:
        rows.append(
            {
                "product_id": forecast.product_id,
                "product_name": forecast.product_name,
                "category": forecast.category,
                "forecast_horizon_days": forecast.forecast_horizon_days,
                "current_stock": forecast.current_stock,
                "avg_daily_demand": forecast.avg_daily_demand,
                "recent_avg_daily_demand": forecast.recent_avg_daily_demand,
                "predicted_demand": forecast.predicted_demand,
                "safety_stock": forecast.safety_stock,
                "recommended_reorder_quantity": forecast.recommended_reorder_quantity,
                "trend_direction": forecast.trend_direction,
                "trend_percentage": forecast.trend_percentage,
                "stock_coverage_days": forecast.stock_coverage_days,
                "forecast_stockout_risk": forecast.forecast_stockout_risk,
                "confidence": forecast.confidence,
            }
        )

    return pd.DataFrame(rows)


def summarize_forecasts(forecast_df: pd.DataFrame) -> dict:
    """
    Summarize forecast results.
    """

    if forecast_df.empty:
        return {
            "total_products_forecasted": 0,
            "total_predicted_demand": 0,
            "total_recommended_reorder": 0,
            "high_forecast_stockout_risk": 0,
            "medium_forecast_stockout_risk": 0,
        }

    return {
        "total_products_forecasted": int(forecast_df["product_id"].nunique()),
        "total_predicted_demand": int(forecast_df["predicted_demand"].sum()),
        "total_recommended_reorder": int(
            forecast_df["recommended_reorder_quantity"].sum()
        ),
        "high_forecast_stockout_risk": int(
            (forecast_df["forecast_stockout_risk"] == "High").sum()
        ),
        "medium_forecast_stockout_risk": int(
            (forecast_df["forecast_stockout_risk"] == "Medium").sum()
        ),
    }