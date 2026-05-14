from __future__ import annotations

import os
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
from src.database.inventory_repository import InventoryRepository
from src.security.upload_security import validate_upload_metadata


app = FastAPI(
    title="StockSense AI API",
    description=(
        "Backend API for inventory data validation, KPI analytics, "
        "forecasting, recommendations, and AI agent responses."
    ),
    version="0.1.0",
)

# CORS allows the frontend dashboard or future React app to call this API.
def get_allowed_cors_origins() -> list[str]:
    """
    Get allowed CORS origins from environment variable.

    Development default:
        *

    Production example:
        https://stocksense-ai.com,https://app.stocksense-ai.com
    """

    raw_origins = os.getenv("STOCKSENSE_CORS_ORIGINS", "*")

    if raw_origins.strip() == "*":
        return ["*"]

    return [
        origin.strip()
        for origin in raw_origins.split(",")
        if origin.strip()
    ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_uploaded_inventory_file(file: UploadFile) -> pd.DataFrame:
    """
    Read uploaded inventory file into a pandas DataFrame.

    Supports CSV and Excel files.

    Security checks:
    - filename must exist
    - extension must be allowed
    - file size must be within configured limit
    """

    filename = file.filename or ""

    try:
        file_bytes = file.file.read()

        security_result = validate_upload_metadata(
            filename=filename,
            file_size_bytes=len(file_bytes),
        )

        if not security_result.is_allowed:
            raise HTTPException(
                status_code=400,
                detail=security_result.message,
            )

        extension = security_result.extension

        if extension == ".csv":
            return pd.read_csv(io.BytesIO(file_bytes))

        if extension in [".xlsx", ".xls"]:
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

def run_analysis_from_database_batch(batch_id: int) -> Dict[str, Any]:
    """
    Load inventory records from SQLite by batch_id and run full analysis.

    This allows the API to analyze previously uploaded data without requiring
    the user to upload the same file again.
    """

    repository = InventoryRepository()
    stored_df = repository.load_inventory_records(batch_id=batch_id)

    if stored_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No inventory records found for batch_id={batch_id}",
        )

    return run_full_analysis(stored_df)


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
    

@app.post("/uploads/save")
def save_inventory_upload(
    file: UploadFile = File(...),
    company_name: str = Form("Demo Company"),
    industry: str = Form("General Inventory"),
) -> Dict[str, Any]:
    """
    Validate uploaded inventory file and save clean records into SQLite.
    """

    df = read_uploaded_inventory_file(file)
    validation_result = validate_inventory_data(df)

    if not validation_result.is_valid:
        return make_json_safe(
            {
                "is_saved": False,
                "is_valid": False,
                "data_quality_score": validation_result.data_quality_score,
                "errors": validation_result.errors,
                "warnings": validation_result.warnings,
                "message": "Upload was not saved because validation failed.",
            }
        )

    repository = InventoryRepository()

    batch_id = repository.save_inventory_dataframe(
        df=validation_result.cleaned_data,
        company_name=company_name,
        industry=industry,
        source_filename=file.filename or "uploaded_inventory.csv",
        data_quality_score=validation_result.data_quality_score,
    )

    return make_json_safe(
        {
            "is_saved": True,
            "is_valid": True,
            "batch_id": batch_id,
            "company_name": company_name,
            "industry": industry,
            "source_filename": file.filename,
            "row_count": len(validation_result.cleaned_data),
            "data_quality_score": validation_result.data_quality_score,
            "warnings": validation_result.warnings,
            "message": "Inventory upload saved successfully.",
        }
    )


@app.get("/uploads")
def list_upload_batches(limit: int = 10) -> Dict[str, Any]:
    """
    List recent inventory upload batches saved in SQLite.
    """

    repository = InventoryRepository()
    batches_df = repository.get_upload_batches(limit=limit)

    return make_json_safe(
        {
            "count": len(batches_df),
            "batches": batches_df,
        }
    )


@app.get("/uploads/{batch_id}/analyze")
def analyze_saved_upload(batch_id: int) -> Dict[str, Any]:
    """
    Run inventory analysis on a previously saved upload batch.
    """

    analysis_result = run_analysis_from_database_batch(batch_id=batch_id)

    if not analysis_result["is_valid"]:
        return make_json_safe(analysis_result)

    return make_json_safe(
        {
            "is_valid": True,
            "batch_id": batch_id,
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
    )


@app.post("/uploads/{batch_id}/ask")
def ask_agent_about_saved_upload(
    batch_id: int,
    question: str = Form(...),
) -> Dict[str, Any]:
    """
    Ask the AI inventory agent a question about a previously saved upload batch.
    """

    analysis_result = run_analysis_from_database_batch(batch_id=batch_id)

    if not analysis_result["is_valid"]:
        return make_json_safe(
            {
                "is_valid": False,
                "batch_id": batch_id,
                "data_quality_score": analysis_result["data_quality_score"],
                "errors": analysis_result["errors"],
                "warnings": analysis_result["warnings"],
                "answer": "I cannot answer questions until the stored data passes validation.",
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
            "batch_id": batch_id,
            "question": question,
            "answer": response.answer,
            "intent": response.intent,
            "tools_used": response.tools_used,
            "confidence": response.confidence,
        }
    )