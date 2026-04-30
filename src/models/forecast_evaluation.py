from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class ForecastEvaluationResult:
    """
    Stores forecasting evaluation results.

    mae:
        Mean Absolute Error.

    rmse:
        Root Mean Squared Error.

    mape:
        Mean Absolute Percentage Error.

    evaluated_products:
        Number of products included in evaluation.

    evaluation_data:
        Product-date level actual vs predicted results.
    """

    mae: float
    rmse: float
    mape: float
    evaluated_products: int
    evaluation_data: pd.DataFrame


def prepare_daily_demand(enriched_data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare daily product demand from enriched inventory data.

    Demand is represented using sold_quantity.
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
        raise ValueError(f"Missing required columns for forecast evaluation: {missing_columns}")

    data = enriched_data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    daily_demand = (
        data.groupby(["date", "product_id", "product_name", "category"], as_index=False)
        .agg(actual_demand=("sold_quantity", "sum"))
        .sort_values(["product_id", "date"])
    )

    return daily_demand


def evaluate_moving_average_forecast(
    enriched_data: pd.DataFrame,
    recent_window_days: int = 14,
    test_window_days: int = 14,
) -> ForecastEvaluationResult:
    """
    Evaluate a moving average forecasting baseline.

    Method:
    - For each product, reserve the last test_window_days as test data.
    - Use the previous recent_window_days from training history to predict test demand.
    - Compare predicted demand with actual demand.
    """

    if recent_window_days <= 0:
        raise ValueError("recent_window_days must be greater than 0.")

    if test_window_days <= 0:
        raise ValueError("test_window_days must be greater than 0.")

    daily_demand = prepare_daily_demand(enriched_data)

    evaluation_rows = []

    for product_id, product_history in daily_demand.groupby("product_id"):
        product_history = product_history.sort_values("date").copy()

        if len(product_history) <= recent_window_days + test_window_days:
            continue

        train_data = product_history.iloc[:-test_window_days]
        test_data = product_history.iloc[-test_window_days:]

        recent_train_demand = train_data["actual_demand"].tail(recent_window_days)
        predicted_daily_demand = float(recent_train_demand.mean())

        for _, row in test_data.iterrows():
            evaluation_rows.append(
                {
                    "date": row["date"],
                    "product_id": row["product_id"],
                    "product_name": row["product_name"],
                    "category": row["category"],
                    "actual_demand": float(row["actual_demand"]),
                    "predicted_demand": round(predicted_daily_demand, 2),
                    "absolute_error": abs(float(row["actual_demand"]) - predicted_daily_demand),
                }
            )

    evaluation_df = pd.DataFrame(evaluation_rows)

    if evaluation_df.empty:
        return ForecastEvaluationResult(
            mae=0.0,
            rmse=0.0,
            mape=0.0,
            evaluated_products=0,
            evaluation_data=evaluation_df,
        )

    actual = evaluation_df["actual_demand"]
    predicted = evaluation_df["predicted_demand"]

    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

    non_zero_actual = actual.replace(0, np.nan)
    mape = float(
        np.mean(np.abs((actual - predicted) / non_zero_actual).replace([np.inf, -np.inf], np.nan).dropna())
        * 100
    )

    if np.isnan(mape):
        mape = 0.0

    evaluated_products = int(evaluation_df["product_id"].nunique())

    return ForecastEvaluationResult(
        mae=round(mae, 4),
        rmse=round(rmse, 4),
        mape=round(mape, 4),
        evaluated_products=evaluated_products,
        evaluation_data=evaluation_df,
    )


def get_evaluation_metrics_dict(
    result: ForecastEvaluationResult,
) -> Dict[str, float | int]:
    """
    Convert evaluation result into MLflow-friendly metrics dictionary.
    """

    return {
        "mae": result.mae,
        "rmse": result.rmse,
        "mape": result.mape,
        "evaluated_products": result.evaluated_products,
    }