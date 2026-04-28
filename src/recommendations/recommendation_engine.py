from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass
class InventoryRecommendation:
    """
    Represents one business recommendation for an inventory product.
    """

    product_id: str
    product_name: str
    category: str
    recommendation_type: str
    priority: str
    action: str
    reason: str
    confidence: str
    supporting_metrics: Dict[str, float | int | str]


def _format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _format_days(value: float) -> str:
    if value == float("inf"):
        return "Not available"
    return f"{value:.1f} days"


def generate_recommendations(product_performance: pd.DataFrame) -> List[InventoryRecommendation]:
    """
    Generate inventory recommendations from product-level KPI data.

    This function uses business rules to recommend actions such as:
    - Increase stock
    - Reduce stock
    - Monitor product
    - Investigate high waste
    - Maintain current stock
    """

    recommendations: List[InventoryRecommendation] = []

    for _, row in product_performance.iterrows():
        product_recommendations = _generate_product_recommendations(row)
        recommendations.extend(product_recommendations)

    priority_order = {"High": 1, "Medium": 2, "Low": 3}

    recommendations = sorted(
        recommendations,
        key=lambda rec: priority_order.get(rec.priority, 99),
    )

    return recommendations


def _generate_product_recommendations(row: pd.Series) -> List[InventoryRecommendation]:
    """
    Generate recommendations for one product.
    """

    recommendations: List[InventoryRecommendation] = []

    product_id = str(row["product_id"])
    product_name = str(row["product_name"])
    category = str(row["category"])

    stockout_risk = row["stockout_risk"]
    overstock_risk = row["overstock_risk"]
    waste_rate = float(row["waste_rate"])
    sell_through_rate = float(row["sell_through_rate"])
    days_of_stock_left = float(row["days_of_stock_left"])
    current_stock = int(row["current_stock"])
    avg_daily_demand = float(row["avg_daily_demand"])
    product_health_score = int(row["product_health_score"])

    supporting_metrics = {
        "current_stock": current_stock,
        "avg_daily_demand": round(avg_daily_demand, 2),
        "days_of_stock_left": round(days_of_stock_left, 2)
        if days_of_stock_left != float("inf")
        else "Not available",
        "sell_through_rate": round(sell_through_rate, 4),
        "waste_rate": round(waste_rate, 4),
        "product_health_score": product_health_score,
    }

    if stockout_risk == "High":
        recommendations.append(
            InventoryRecommendation(
                product_id=product_id,
                product_name=product_name,
                category=category,
                recommendation_type="Stockout Prevention",
                priority="High",
                action=f"Increase stock for {product_name}.",
                reason=(
                    f"{product_name} has only {_format_days(days_of_stock_left)} of stock left "
                    f"based on average daily demand. This creates a high stockout risk."
                ),
                confidence="High" if avg_daily_demand > 0 else "Medium",
                supporting_metrics=supporting_metrics,
            )
        )

    elif stockout_risk == "Medium":
        recommendations.append(
            InventoryRecommendation(
                product_id=product_id,
                product_name=product_name,
                category=category,
                recommendation_type="Stock Monitoring",
                priority="Medium",
                action=f"Monitor {product_name} and prepare reorder if demand continues.",
                reason=(
                    f"{product_name} has around {_format_days(days_of_stock_left)} of stock left. "
                    f"It is not critical yet, but should be monitored."
                ),
                confidence="Medium",
                supporting_metrics=supporting_metrics,
            )
        )

    if overstock_risk == "High":
        recommendations.append(
            InventoryRecommendation(
                product_id=product_id,
                product_name=product_name,
                category=category,
                recommendation_type="Overstock Reduction",
                priority="High",
                action=f"Reduce future purchase quantity for {product_name}.",
                reason=(
                    f"{product_name} has high overstock risk with {_format_days(days_of_stock_left)} "
                    f"of estimated stock coverage. Excess stock can lock capital and increase holding cost."
                ),
                confidence="High",
                supporting_metrics=supporting_metrics,
            )
        )

    elif overstock_risk == "Medium":
        recommendations.append(
            InventoryRecommendation(
                product_id=product_id,
                product_name=product_name,
                category=category,
                recommendation_type="Overstock Monitoring",
                priority="Medium",
                action=f"Review purchase plan for {product_name}.",
                reason=(
                    f"{product_name} has medium overstock risk with {_format_days(days_of_stock_left)} "
                    f"of stock coverage."
                ),
                confidence="Medium",
                supporting_metrics=supporting_metrics,
            )
        )

    if waste_rate > 0.10:
        recommendations.append(
            InventoryRecommendation(
                product_id=product_id,
                product_name=product_name,
                category=category,
                recommendation_type="Waste Reduction",
                priority="High",
                action=f"Investigate and reduce wastage for {product_name}.",
                reason=(
                    f"{product_name} has a high waste rate of {_format_percent(waste_rate)}. "
                    f"This may indicate over-ordering, expiry issues, poor handling, or demand mismatch."
                ),
                confidence="High",
                supporting_metrics=supporting_metrics,
            )
        )

    elif waste_rate > 0.05:
        recommendations.append(
            InventoryRecommendation(
                product_id=product_id,
                product_name=product_name,
                category=category,
                recommendation_type="Waste Monitoring",
                priority="Medium",
                action=f"Monitor wastage pattern for {product_name}.",
                reason=(
                    f"{product_name} has a moderate waste rate of {_format_percent(waste_rate)}. "
                    f"It should be watched before it becomes a larger loss."
                ),
                confidence="Medium",
                supporting_metrics=supporting_metrics,
            )
        )

    if (
        stockout_risk == "Low"
        and overstock_risk == "Low"
        and waste_rate <= 0.05
        and sell_through_rate >= 0.30
        and product_health_score >= 80
    ):
        recommendations.append(
            InventoryRecommendation(
                product_id=product_id,
                product_name=product_name,
                category=category,
                recommendation_type="Maintain Strategy",
                priority="Low",
                action=f"Maintain current inventory strategy for {product_name}.",
                reason=(
                    f"{product_name} is performing well with healthy sell-through, low waste, "
                    f"and no major stock risk."
                ),
                confidence="High",
                supporting_metrics=supporting_metrics,
            )
        )

    if product_health_score < 60:
        recommendations.append(
            InventoryRecommendation(
                product_id=product_id,
                product_name=product_name,
                category=category,
                recommendation_type="Product Review",
                priority="High",
                action=f"Review overall inventory strategy for {product_name}.",
                reason=(
                    f"{product_name} has a low product health score of {product_health_score}/100. "
                    f"It may require pricing, purchasing, promotion, or supplier review."
                ),
                confidence="Medium",
                supporting_metrics=supporting_metrics,
            )
        )

    return recommendations


def recommendations_to_dataframe(
    recommendations: List[InventoryRecommendation],
) -> pd.DataFrame:
    """
    Convert recommendations into a DataFrame for dashboard display.
    """

    rows = []

    for rec in recommendations:
        rows.append(
            {
                "priority": rec.priority,
                "product_name": rec.product_name,
                "category": rec.category,
                "recommendation_type": rec.recommendation_type,
                "action": rec.action,
                "reason": rec.reason,
                "confidence": rec.confidence,
                "current_stock": rec.supporting_metrics["current_stock"],
                "avg_daily_demand": rec.supporting_metrics["avg_daily_demand"],
                "days_of_stock_left": rec.supporting_metrics["days_of_stock_left"],
                "sell_through_rate": rec.supporting_metrics["sell_through_rate"],
                "waste_rate": rec.supporting_metrics["waste_rate"],
                "product_health_score": rec.supporting_metrics["product_health_score"],
            }
        )

    return pd.DataFrame(rows)


def summarize_recommendations(recommendations: List[InventoryRecommendation]) -> Dict[str, int]:
    """
    Summarize recommendation counts by priority.
    """

    return {
        "total_recommendations": len(recommendations),
        "high_priority": sum(1 for rec in recommendations if rec.priority == "High"),
        "medium_priority": sum(1 for rec in recommendations if rec.priority == "Medium"),
        "low_priority": sum(1 for rec in recommendations if rec.priority == "Low"),
    }