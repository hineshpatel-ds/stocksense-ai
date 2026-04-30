from __future__ import annotations

import io
import math
from typing import Any, Dict

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.analytics.kpi_engine import calculate_inventory_kpis
from src.chatbot.agent import InventoryAIAgent
from src.data.validation import validate_inventory_data
from src.models.forecasting_engine import (
    forecasts_to_dataframe,
    generate_product_forecasts,
    summarize_forecasts,
)
from src.recommendations.recommendation_engine import (
    generate_recommendations,
    recommendations_to_dataframe,
    summarize_recommendations,
)


app = FastAPI(
    title="StockSense AI API",
    description=(
        "Backend API for inventory data validation, KPI analytics, "
        "forecasting, recommendations, and AI agent responses."
    ),
    version="0.1.0",
)

# CORS allows the frontend dashboard or future React app to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only. Restrict this in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_uploaded_inventory_file(file: UploadFile) -> pd.DataFrame:
    """
    Read uploaded inventory file into a pandas DataFrame.

    Supports CSV and Excel files.
    """

    filename = file.filename or ""

    try:
        file_bytes = file.file.read()

        if filename.endswith(".csv"):
            return pd.read_csv(io.BytesIO(file_bytes))

        if filename.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(file_bytes))

        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload CSV or Excel file.",
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read uploaded file: {exc}",
        ) from exc


def make_json_safe(value: Any) -> Any:
    """
    Convert pandas/numpy objects into JSON-safe Python objects.

    APIs cannot safely return NaN, infinity, numpy types, or pandas timestamps.
    This helper cleans those values.
    """

    if isinstance(value, dict):
        return {key: make_json_safe(val) for key, val in value.items()}

    if isinstance(value, list):
        return [make_json_safe(item) for item in value]

    if isinstance(value, pd.DataFrame):
        return make_json_safe(value.to_dict(orient="records"))

    if isinstance(value, pd.Series):
        return make_json_safe(value.to_dict())

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    return value


def run_full_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run complete StockSense AI analysis pipeline.

    Pipeline:
    1. Validate data
    2. Calculate KPIs
    3. Generate forecasts
    4. Generate recommendations
    5. Prepare agent context
    """

    validation_result = validate_inventory_data(df)

    if not validation_result.is_valid:
        return {
            "is_valid": False,
            "data_quality_score": validation_result.data_quality_score,
            "errors": validation_result.errors,
            "warnings": validation_result.warnings,
        }

    kpi_result = calculate_inventory_kpis(validation_result.cleaned_data)

    recommendations = generate_recommendations(kpi_result["product_performance"])
    recommendation_df = recommendations_to_dataframe(recommendations)
    recommendation_summary = summarize_recommendations(recommendations)

    forecasts = generate_product_forecasts(
        enriched_data=kpi_result["enriched_data"],
        product_performance=kpi_result["product_performance"],
        forecast_horizon_days=30,
    )
    forecast_df = forecasts_to_dataframe(forecasts)
    forecast_summary = summarize_forecasts(forecast_df)

    return {
        "is_valid": True,
        "data_quality_score": validation_result.data_quality_score,
        "errors": validation_result.errors,
        "warnings": validation_result.warnings,
        "summary_metrics": kpi_result["summary_metrics"],
        "risk_summary": kpi_result["risk_summary"],
        "product_performance": kpi_result["product_performance"],
        "category_performance": kpi_result["category_performance"],
        "forecast_summary": forecast_summary,
        "forecast_results": forecast_df,
        "recommendation_summary": recommendation_summary,
        "recommendations": recommendation_df,
        "agent_context": {
            "summary_metrics": kpi_result["summary_metrics"],
            "risk_summary": kpi_result["risk_summary"],
            "product_performance": kpi_result["product_performance"],
            "category_performance": kpi_result["category_performance"],
            "recommendation_df": recommendation_df,
            "forecast_df": forecast_df,
            "forecast_summary": forecast_summary,
        },
    }


@app.get("/health")
def health_check() -> Dict[str, str]:
    """
    Health check endpoint.

    Used to confirm that the backend service is running.
    """

    return {
        "status": "healthy",
        "service": "stocksense-ai-api",
        "version": "0.1.0",
    }


@app.post("/validate")
def validate_inventory_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Validate uploaded inventory file.
    """

    df = read_uploaded_inventory_file(file)
    validation_result = validate_inventory_data(df)

    return make_json_safe(
        {
            "is_valid": validation_result.is_valid,
            "data_quality_score": validation_result.data_quality_score,
            "errors": validation_result.errors,
            "warnings": validation_result.warnings,
            "row_count": len(validation_result.cleaned_data),
            "columns": list(validation_result.cleaned_data.columns),
        }
    )


@app.post("/analyze")
def analyze_inventory_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Run complete inventory analysis on uploaded file.
    """

    df = read_uploaded_inventory_file(file)
    analysis_result = run_full_analysis(df)

    if not analysis_result["is_valid"]:
        return make_json_safe(analysis_result)

    response = {
        "is_valid": True,
        "data_quality_score": analysis_result["data_quality_score"],
        "warnings": analysis_result["warnings"],
        "summary_metrics": analysis_result["summary_metrics"],
        "risk_summary": analysis_result["risk_summary"],
        "forecast_summary": analysis_result["forecast_summary"],
        "recommendation_summary": analysis_result["recommendation_summary"],
        "product_performance": analysis_result["product_performance"],
        "category_performance": analysis_result["category_performance"],
        "forecast_results": analysis_result["forecast_results"],
        "recommendations": analysis_result["recommendations"],
    }

    return make_json_safe(response)


@app.post("/ask")
def ask_inventory_agent(
    question: str = Form(...),
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """
    Ask the AI inventory agent a question about uploaded inventory data.
    """

    df = read_uploaded_inventory_file(file)
    analysis_result = run_full_analysis(df)

    if not analysis_result["is_valid"]:
        return make_json_safe(
            {
                "is_valid": False,
                "data_quality_score": analysis_result["data_quality_score"],
                "errors": analysis_result["errors"],
                "warnings": analysis_result["warnings"],
                "answer": "I cannot answer questions until the uploaded data passes validation.",
            }
        )

    agent = InventoryAIAgent()
    response = agent.answer_question(
        question=question,
        context=analysis_result["agent_context"],
    )

    return make_json_safe(
        {
            "is_valid": True,
            "question": question,
            "answer": response.answer,
            "intent": response.intent,
            "tools_used": response.tools_used,
            "confidence": response.confidence,
        }
    )