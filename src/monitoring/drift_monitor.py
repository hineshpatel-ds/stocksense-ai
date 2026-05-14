from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd


@dataclass
class DriftMonitoringResult:
    """
    Stores monitoring result.

    reference_period:
        Older historical period used as baseline.

    current_period:
        Recent period compared against baseline.

    numeric_drift:
        Drift report for numerical columns.

    categorical_drift:
        Drift report for categorical columns.

    demand_shift:
        Product-level demand change report.

    summary:
        High-level monitoring summary.
    """

    reference_period: Dict[str, str]
    current_period: Dict[str, str]
    numeric_drift: pd.DataFrame
    categorical_drift: pd.DataFrame
    demand_shift: pd.DataFrame
    summary: Dict[str, Any]


def classify_drift_score(score: float) -> str:
    """
    Classify drift score into Low, Medium, or High.

    We use simple thresholds:
    - Low: score < 0.10
    - Medium: 0.10 to 0.20
    - High: >= 0.20
    """

    if score >= 0.20:
        return "High"

    if score >= 0.10:
        return "Medium"

    return "Low"


def calculate_population_stability_index(
    reference_values: pd.Series,
    current_values: pd.Series,
    bins: int = 10,
) -> float:
    """
    Calculate a PSI-like drift score for numerical values.

    PSI compares distribution differences between reference and current data.
    Higher score means stronger drift.
    """

    reference_values = pd.to_numeric(reference_values, errors="coerce").dropna()
    current_values = pd.to_numeric(current_values, errors="coerce").dropna()

    if reference_values.empty or current_values.empty:
        return 0.0

    if reference_values.nunique() <= 1:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.unique(reference_values.quantile(quantiles).values)

    if len(bin_edges) < 3:
        min_value = float(reference_values.min())
        max_value = float(reference_values.max())

        if min_value == max_value:
            return 0.0

        bin_edges = np.linspace(min_value, max_value, bins + 1)

    reference_counts, _ = np.histogram(reference_values, bins=bin_edges)
    current_counts, _ = np.histogram(current_values, bins=bin_edges)

    reference_percents = reference_counts / max(reference_counts.sum(), 1)
    current_percents = current_counts / max(current_counts.sum(), 1)

    epsilon = 0.0001
    reference_percents = np.where(reference_percents == 0, epsilon, reference_percents)
    current_percents = np.where(current_percents == 0, epsilon, current_percents)

    psi = np.sum(
        (current_percents - reference_percents)
        * np.log(current_percents / reference_percents)
    )

    return round(float(psi), 4)


def calculate_numeric_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_columns: List[str],
) -> pd.DataFrame:
    """
    Calculate numerical drift for selected columns.
    """

    rows = []

    for column in numeric_columns:
        if column not in reference_df.columns or column not in current_df.columns:
            continue

        drift_score = calculate_population_stability_index(
            reference_df[column],
            current_df[column],
        )

        reference_mean = pd.to_numeric(reference_df[column], errors="coerce").mean()
        current_mean = pd.to_numeric(current_df[column], errors="coerce").mean()

        rows.append(
            {
                "feature": column,
                "reference_mean": round(float(reference_mean), 4)
                if not np.isnan(reference_mean)
                else 0.0,
                "current_mean": round(float(current_mean), 4)
                if not np.isnan(current_mean)
                else 0.0,
                "drift_score": drift_score,
                "drift_level": classify_drift_score(drift_score),
            }
        )

    return pd.DataFrame(rows)


def calculate_categorical_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    categorical_columns: List[str],
) -> pd.DataFrame:
    """
    Calculate distribution drift for categorical columns.

    Uses total variation distance:
    0 means no distribution change.
    Higher values mean stronger category distribution shift.
    """

    rows = []

    for column in categorical_columns:
        if column not in reference_df.columns or column not in current_df.columns:
            continue

        reference_distribution = reference_df[column].value_counts(normalize=True)
        current_distribution = current_df[column].value_counts(normalize=True)

        all_categories = sorted(
            set(reference_distribution.index).union(set(current_distribution.index))
        )

        total_difference = 0.0

        for category in all_categories:
            reference_percent = float(reference_distribution.get(category, 0.0))
            current_percent = float(current_distribution.get(category, 0.0))
            total_difference += abs(current_percent - reference_percent)

        drift_score = round(total_difference / 2, 4)

        rows.append(
            {
                "feature": column,
                "unique_reference_values": int(reference_df[column].nunique()),
                "unique_current_values": int(current_df[column].nunique()),
                "drift_score": drift_score,
                "drift_level": classify_drift_score(drift_score),
            }
        )

    return pd.DataFrame(rows)


def calculate_product_demand_shift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate product-level demand change between reference and current periods.
    """

    required_columns = ["product_id", "product_name", "sold_quantity"]

    for column in required_columns:
        if column not in reference_df.columns or column not in current_df.columns:
            raise ValueError(f"Missing required column for demand shift: {column}")

    reference_days = max(reference_df["date"].nunique(), 1)
    current_days = max(current_df["date"].nunique(), 1)

    reference_demand = (
        reference_df.groupby(["product_id", "product_name"], as_index=False)
        .agg(reference_units_sold=("sold_quantity", "sum"))
    )

    current_demand = (
        current_df.groupby(["product_id", "product_name"], as_index=False)
        .agg(current_units_sold=("sold_quantity", "sum"))
    )

    demand_shift = reference_demand.merge(
        current_demand,
        on=["product_id", "product_name"],
        how="outer",
    ).fillna(0)

    demand_shift["reference_avg_daily_demand"] = (
        demand_shift["reference_units_sold"] / reference_days
    )

    demand_shift["current_avg_daily_demand"] = (
        demand_shift["current_units_sold"] / current_days
    )

    demand_shift["demand_change_percent"] = np.where(
        demand_shift["reference_avg_daily_demand"] > 0,
        (
            demand_shift["current_avg_daily_demand"]
            - demand_shift["reference_avg_daily_demand"]
        )
        / demand_shift["reference_avg_daily_demand"],
        0,
    )

    demand_shift["demand_shift_level"] = demand_shift["demand_change_percent"].apply(
        classify_demand_shift
    )

    demand_shift = demand_shift.sort_values(
        by="demand_change_percent",
        key=lambda series: series.abs(),
        ascending=False,
    ).reset_index(drop=True)

    return demand_shift[
        [
            "product_id",
            "product_name",
            "reference_avg_daily_demand",
            "current_avg_daily_demand",
            "demand_change_percent",
            "demand_shift_level",
        ]
    ]


def classify_demand_shift(change_percent: float) -> str:
    """
    Classify product demand shift.
    """

    if change_percent >= 0.25:
        return "Strong Increase"

    if change_percent >= 0.10:
        return "Moderate Increase"

    if change_percent <= -0.25:
        return "Strong Decrease"

    if change_percent <= -0.10:
        return "Moderate Decrease"

    return "Stable"


def split_reference_current_periods(
    df: pd.DataFrame,
    current_window_days: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into reference period and current period.

    The current period is the most recent N days.
    The reference period is everything before that.
    """

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).sort_values("date")

    if data.empty:
        return data.copy(), data.copy()

    max_date = data["date"].max()
    current_start_date = max_date - pd.Timedelta(days=current_window_days - 1)

    reference_df = data[data["date"] < current_start_date].copy()
    current_df = data[data["date"] >= current_start_date].copy()

    return reference_df, current_df


def monitor_inventory_drift(
    enriched_data: pd.DataFrame,
    current_window_days: int = 30,
) -> DriftMonitoringResult:
    """
    Run drift monitoring on inventory data.

    Compares older reference data with recent current data.
    """

    reference_df, current_df = split_reference_current_periods(
        enriched_data,
        current_window_days=current_window_days,
    )

    if reference_df.empty or current_df.empty:
        empty_df = pd.DataFrame()

        return DriftMonitoringResult(
            reference_period={},
            current_period={},
            numeric_drift=empty_df,
            categorical_drift=empty_df,
            demand_shift=empty_df,
            summary={
                "monitoring_status": "Insufficient Data",
                "message": "Not enough historical data to compare reference and current periods.",
                "high_drift_features": 0,
                "medium_drift_features": 0,
                "products_with_strong_demand_shift": 0,
            },
        )

    numeric_columns = [
        "sold_quantity",
        "wasted_quantity",
        "closing_stock",
        "revenue",
        "waste_value",
        "sell_through_rate",
        "waste_rate",
    ]

    categorical_columns = [
        "category",
        "store_id",
    ]

    numeric_drift = calculate_numeric_drift(
        reference_df=reference_df,
        current_df=current_df,
        numeric_columns=numeric_columns,
    )

    categorical_drift = calculate_categorical_drift(
        reference_df=reference_df,
        current_df=current_df,
        categorical_columns=categorical_columns,
    )

    demand_shift = calculate_product_demand_shift(
        reference_df=reference_df,
        current_df=current_df,
    )

    all_drift = pd.concat(
        [
            numeric_drift[["feature", "drift_score", "drift_level"]],
            categorical_drift[["feature", "drift_score", "drift_level"]],
        ],
        ignore_index=True,
    )

    high_drift_features = int((all_drift["drift_level"] == "High").sum())
    medium_drift_features = int((all_drift["drift_level"] == "Medium").sum())

    strong_demand_shift = int(
        demand_shift["demand_shift_level"].isin(
            ["Strong Increase", "Strong Decrease"]
        ).sum()
    )

    if high_drift_features > 0 or strong_demand_shift > 0:
        monitoring_status = "Attention Needed"
    elif medium_drift_features > 0:
        monitoring_status = "Watch"
    else:
        monitoring_status = "Stable"

    summary = {
        "monitoring_status": monitoring_status,
        "reference_rows": int(len(reference_df)),
        "current_rows": int(len(current_df)),
        "reference_start": str(reference_df["date"].min().date()),
        "reference_end": str(reference_df["date"].max().date()),
        "current_start": str(current_df["date"].min().date()),
        "current_end": str(current_df["date"].max().date()),
        "high_drift_features": high_drift_features,
        "medium_drift_features": medium_drift_features,
        "products_with_strong_demand_shift": strong_demand_shift,
    }

    return DriftMonitoringResult(
        reference_period={
            "start": summary["reference_start"],
            "end": summary["reference_end"],
        },
        current_period={
            "start": summary["current_start"],
            "end": summary["current_end"],
        },
        numeric_drift=numeric_drift,
        categorical_drift=categorical_drift,
        demand_shift=demand_shift,
        summary=summary,
    )