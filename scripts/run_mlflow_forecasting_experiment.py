from __future__ import annotations

from pathlib import Path

import mlflow

from src.analytics.kpi_engine import calculate_inventory_kpis
from src.data.validation import validate_inventory_file
from src.models.forecast_evaluation import (
    evaluate_moving_average_forecast,
    get_evaluation_metrics_dict,
)


def run_experiment() -> None:
    """
    Run and track a baseline forecasting experiment using MLflow.
    """

    experiment_name = "stocksense-forecasting-baseline"
    file_path = "data/sample/sample_inventory.csv"
    recent_window_days = 14
    test_window_days = 14
    forecast_method = "moving_average_baseline"

    mlflow.set_experiment(experiment_name)

    validation_result = validate_inventory_file(file_path)

    if not validation_result.is_valid:
        raise ValueError(f"Data validation failed: {validation_result.errors}")

    kpi_result = calculate_inventory_kpis(validation_result.cleaned_data)

    with mlflow.start_run(run_name="moving-average-baseline-14d-window"):
        mlflow.log_param("forecast_method", forecast_method)
        mlflow.log_param("data_file", file_path)
        mlflow.log_param("recent_window_days", recent_window_days)
        mlflow.log_param("test_window_days", test_window_days)
        mlflow.log_param("data_quality_score", validation_result.data_quality_score)

        evaluation_result = evaluate_moving_average_forecast(
            enriched_data=kpi_result["enriched_data"],
            recent_window_days=recent_window_days,
            test_window_days=test_window_days,
        )

        metrics = get_evaluation_metrics_dict(evaluation_result)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        artifact_dir = Path("artifacts")
        artifact_dir.mkdir(exist_ok=True)

        evaluation_artifact_path = artifact_dir / "forecast_evaluation_results.csv"
        evaluation_result.evaluation_data.to_csv(evaluation_artifact_path, index=False)

        mlflow.log_artifact(str(evaluation_artifact_path))

        print("MLflow forecasting experiment completed.")
        print(f"Experiment name: {experiment_name}")
        print("Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"- {metric_name}: {metric_value}")

        print(f"Artifact saved: {evaluation_artifact_path}")


if __name__ == "__main__":
    run_experiment()