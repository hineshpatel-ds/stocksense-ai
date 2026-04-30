from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class ToolResult:
    """
    Standard output format for every chatbot tool.

    This makes the agent reliable because every tool returns:
    - success status
    - message
    - structured data
    """

    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


def find_product_name(question: str, product_performance: pd.DataFrame) -> Optional[str]:
    """
    Find product name from user question using simple case-insensitive matching.

    Example:
    Question: "Should I reorder veggie burger?"
    Product found: "Veggie Burger"
    """

    question_lower = question.lower()

    for product_name in product_performance["product_name"].dropna().unique():
        if str(product_name).lower() in question_lower:
            return str(product_name)

    return None


def get_inventory_summary(context: Dict[str, Any]) -> ToolResult:
    """
    Return high-level inventory summary.
    """

    summary = context["summary_metrics"]
    risk_summary = context["risk_summary"]

    message = (
        f"Inventory health score is {summary['inventory_health_score']}/100. "
        f"Total revenue is ${summary['total_revenue']:,.2f}, "
        f"units sold are {summary['total_units_sold']:,}, "
        f"and waste value is ${summary['total_waste_value']:,.2f}. "
        f"There are {risk_summary['high_stockout_risk_products']} high stockout-risk products "
        f"and {risk_summary['high_overstock_risk_products']} high overstock-risk products."
    )

    return ToolResult(
        success=True,
        message=message,
        data={
            "summary_metrics": summary,
            "risk_summary": risk_summary,
        },
    )


def get_top_products(context: Dict[str, Any], limit: int = 5) -> ToolResult:
    """
    Return top products by revenue.
    """

    product_df = context["product_performance"]

    top_products = product_df.sort_values("total_revenue", ascending=False).head(limit)

    if top_products.empty:
        return ToolResult(False, "No product data is available.")

    product_lines = []

    for _, row in top_products.iterrows():
        product_lines.append(
            f"{row['product_name']} generated ${row['total_revenue']:,.2f} "
            f"from {int(row['total_units_sold']):,} units sold."
        )

    message = "Top products by revenue:\n\n" + "\n".join(
        f"{index + 1}. {line}" for index, line in enumerate(product_lines)
    )

    return ToolResult(
        success=True,
        message=message,
        data={"top_products": top_products.to_dict(orient="records")},
    )


def get_stockout_risk_products(context: Dict[str, Any]) -> ToolResult:
    """
    Return products with high or medium stockout risk.
    """

    product_df = context["product_performance"]

    risk_df = product_df[
        product_df["stockout_risk"].isin(["High", "Medium"])
    ].sort_values(["stockout_risk", "days_of_stock_left"])

    if risk_df.empty:
        return ToolResult(
            success=True,
            message="No high or medium stockout-risk products were detected.",
            data={"products": []},
        )

    lines = []

    for _, row in risk_df.iterrows():
        lines.append(
            f"{row['product_name']} has {row['stockout_risk']} stockout risk "
            f"with approximately {row['days_of_stock_left']:.1f} days of stock left."
        )

    message = "Stockout risk products:\n\n" + "\n".join(
        f"- {line}" for line in lines
    )

    return ToolResult(
        success=True,
        message=message,
        data={"products": risk_df.to_dict(orient="records")},
    )


def get_overstock_risk_products(context: Dict[str, Any]) -> ToolResult:
    """
    Return products with high or medium overstock risk.
    """

    product_df = context["product_performance"]

    risk_df = product_df[
        product_df["overstock_risk"].isin(["High", "Medium"])
    ].sort_values(["overstock_risk", "days_of_stock_left"], ascending=[True, False])

    if risk_df.empty:
        return ToolResult(
            success=True,
            message="No high or medium overstock-risk products were detected.",
            data={"products": []},
        )

    lines = []

    for _, row in risk_df.iterrows():
        lines.append(
            f"{row['product_name']} has {row['overstock_risk']} overstock risk "
            f"with approximately {row['days_of_stock_left']:.1f} days of stock coverage."
        )

    message = "Overstock risk products:\n\n" + "\n".join(
        f"- {line}" for line in lines
    )

    return ToolResult(
        success=True,
        message=message,
        data={"products": risk_df.to_dict(orient="records")},
    )


def get_waste_analysis(context: Dict[str, Any], limit: int = 5) -> ToolResult:
    """
    Return products with highest waste value.
    """

    product_df = context["product_performance"]

    waste_df = product_df.sort_values("total_waste_value", ascending=False).head(limit)

    if waste_df.empty:
        return ToolResult(False, "No waste analysis data is available.")

    lines = []

    for _, row in waste_df.iterrows():
        lines.append(
            f"{row['product_name']} has waste value of ${row['total_waste_value']:,.2f} "
            f"and waste rate of {row['waste_rate'] * 100:.1f}%."
        )

    message = "Highest waste products:\n\n" + "\n".join(
        f"- {line}" for line in lines
    )

    return ToolResult(
        success=True,
        message=message,
        data={"products": waste_df.to_dict(orient="records")},
    )


def get_general_recommendations(context: Dict[str, Any], limit: int = 5) -> ToolResult:
    """
    Return top recommendations.
    """

    recommendation_df = context["recommendation_df"]

    if recommendation_df.empty:
        return ToolResult(
            success=True,
            message="No major recommendation is needed. Inventory appears stable based on current rules.",
            data={"recommendations": []},
        )

    priority_order = {"High": 1, "Medium": 2, "Low": 3}
    recommendation_df = recommendation_df.copy()
    recommendation_df["priority_rank"] = recommendation_df["priority"].map(priority_order)

    top_recommendations = (
        recommendation_df.sort_values("priority_rank")
        .drop(columns=["priority_rank"])
        .head(limit)
    )

    lines = []

    for _, row in top_recommendations.iterrows():
        lines.append(
            f"[{row['priority']}] {row['action']} Reason: {row['reason']}"
        )

    message = "Top inventory recommendations:\n\n" + "\n".join(
        f"- {line}" for line in lines
    )

    return ToolResult(
        success=True,
        message=message,
        data={"recommendations": top_recommendations.to_dict(orient="records")},
    )


def get_product_recommendation(
    context: Dict[str, Any],
    product_name: str,
) -> ToolResult:
    """
    Return recommendation and forecast for a specific product.
    """

    recommendation_df = context["recommendation_df"]
    forecast_df = context["forecast_df"]
    product_df = context["product_performance"]

    product_rows = product_df[
        product_df["product_name"].str.lower() == product_name.lower()
    ]

    if product_rows.empty:
        return ToolResult(
            success=False,
            message=f"I could not find product '{product_name}' in the uploaded dataset.",
        )

    product_row = product_rows.iloc[0]

    product_recommendations = recommendation_df[
        recommendation_df["product_name"].str.lower() == product_name.lower()
    ]

    product_forecasts = forecast_df[
        forecast_df["product_name"].str.lower() == product_name.lower()
    ]

    answer_parts = [
        f"Product analysis for {product_name}:",
        f"- Current stock: {int(product_row['current_stock'])}",
        f"- Stockout risk: {product_row['stockout_risk']}",
        f"- Overstock risk: {product_row['overstock_risk']}",
        f"- Waste rate: {product_row['waste_rate'] * 100:.1f}%",
        f"- Product health score: {int(product_row['product_health_score'])}/100",
    ]

    if not product_forecasts.empty:
        forecast_row = product_forecasts.iloc[0]
        answer_parts.extend(
            [
                f"- Predicted 30-day demand: {int(forecast_row['predicted_demand'])}",
                f"- Safety stock: {int(forecast_row['safety_stock'])}",
                f"- Recommended reorder quantity: {int(forecast_row['recommended_reorder_quantity'])}",
                f"- Forecast stockout risk: {forecast_row['forecast_stockout_risk']}",
                f"- Forecast confidence: {forecast_row['confidence']}",
            ]
        )

    if not product_recommendations.empty:
        answer_parts.append("\nRecommended actions:")

        for _, row in product_recommendations.iterrows():
            answer_parts.append(
                f"- [{row['priority']}] {row['action']} Reason: {row['reason']}"
            )
    else:
        answer_parts.append(
            "\nNo urgent recommendation was generated for this product based on current rules."
        )

    return ToolResult(
        success=True,
        message="\n".join(answer_parts),
        data={
            "product": product_row.to_dict(),
            "recommendations": product_recommendations.to_dict(orient="records"),
            "forecast": product_forecasts.to_dict(orient="records"),
        },
    )


def get_forecast_summary(context: Dict[str, Any]) -> ToolResult:
    """
    Return overall forecast summary.
    """

    forecast_summary = context["forecast_summary"]

    message = (
        f"Forecast summary for the next 30 days: "
        f"{forecast_summary['total_products_forecasted']} products were forecasted. "
        f"Total predicted demand is {forecast_summary['total_predicted_demand']:,} units. "
        f"Recommended reorder quantity is {forecast_summary['total_recommended_reorder']:,} units. "
        f"{forecast_summary['high_forecast_stockout_risk']} products have high forecast stockout risk."
    )

    return ToolResult(
        success=True,
        message=message,
        data={"forecast_summary": forecast_summary},
    )


def get_category_performance(context: Dict[str, Any]) -> ToolResult:
    """
    Return category-level performance.
    """

    category_df = context["category_performance"]

    if category_df.empty:
        return ToolResult(False, "No category performance data is available.")

    lines = []

    for _, row in category_df.iterrows():
        lines.append(
            f"{row['category']} generated ${row['total_revenue']:,.2f} "
            f"from {int(row['total_units_sold']):,} units sold."
        )

    message = "Category performance:\n\n" + "\n".join(
        f"- {line}" for line in lines
    )

    return ToolResult(
        success=True,
        message=message,
        data={"categories": category_df.to_dict(orient="records")},
    )