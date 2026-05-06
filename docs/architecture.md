# StockSense AI Architecture

## Overview

StockSense AI is designed as a modular AI inventory intelligence platform.

The system separates user interface, backend API, and core business logic.

```text
User
 |
 v
Streamlit Dashboard
 |
 v
FastAPI Backend
 |
 v
Core Services
 |
 |-- Data Validation
 |-- KPI Analytics
 |-- Forecasting
 |-- Recommendation Engine
 |-- AI Agent
 |-- LLM Adapter
 |-- MLflow Tracking
```

---

## Main Components

### 1. Streamlit Dashboard

The dashboard provides the user interface.

Responsibilities:

- Upload inventory files
- Display data quality score
- Show KPI cards
- Display charts and tables
- Show forecasts
- Show recommendations
- Provide AI agent chat interface

---

### 2. FastAPI Backend

The backend exposes API endpoints.

Responsibilities:

- Accept uploaded files
- Validate input data
- Run the full analysis pipeline
- Return JSON results
- Answer AI agent questions
- Provide API documentation through Swagger UI

---

### 3. Data Validation Engine

Location:

```text
src/data/validation.py
```

Responsibilities:

- Standardize column names
- Check required columns
- Validate dates
- Validate numeric columns
- Detect missing values
- Detect negative values
- Check inventory equation
- Generate data quality score

---

### 4. KPI Analytics Engine

Location:

```text
src/analytics/kpi_engine.py
```

Responsibilities:

- Calculate revenue
- Calculate waste value
- Calculate sell-through rate
- Calculate stock turnover
- Calculate stockout risk
- Calculate overstock risk
- Calculate inventory health score

---

### 5. Forecasting Engine

Location:

```text
src/models/forecasting_engine.py
```

Responsibilities:

- Prepare product-level daily demand
- Calculate average daily demand
- Detect demand trend
- Predict future demand
- Calculate safety stock
- Recommend reorder quantity
- Estimate forecast stockout risk

---

### 6. Recommendation Engine

Location:

```text
src/recommendations/recommendation_engine.py
```

Responsibilities:

- Generate stockout prevention recommendations
- Generate overstock reduction recommendations
- Generate waste reduction recommendations
- Assign priority
- Assign confidence
- Provide business explanation

---

### 7. AI Agent Layer

Location:

```text
src/chatbot/agent.py
src/chatbot/tools.py
```

Responsibilities:

- Detect user intent
- Select correct tool
- Call trusted analytics functions
- Return grounded business answers

The agent does not invent numbers. It uses verified outputs from the analytics, forecasting, and recommendation modules.

---

### 8. LLM Adapter

Location:

```text
src/chatbot/llm_adapter.py
```

Responsibilities:

- Provide no-LLM fallback mode
- Support optional local Ollama mode
- Keep LLM integration separate from agent logic
- Rewrite verified answers into natural language when LLM is available

---

### 9. MLflow Tracking

Location:

```text
scripts/run_mlflow_forecasting_experiment.py
src/models/forecast_evaluation.py
```

Responsibilities:

- Evaluate forecasting baseline
- Track parameters
- Track metrics
- Log artifacts
- Support reproducible model experimentation

---

## Data Flow

```text
CSV/Excel Upload
   |
   v
Data Validation
   |
   v
Clean DataFrame
   |
   v
KPI Analytics
   |
   v
Forecasting + Recommendations
   |
   v
Dashboard + API + AI Agent
```

---

## Design Principles

### Modularity

Each major responsibility is placed in its own module.

### Reusability

The same core logic is used by both Streamlit and FastAPI.

### Explainability

Recommendations are based on transparent business rules and supporting metrics.

### Safety

The AI agent uses trusted tools instead of guessing from raw data.

### MLOps Readiness

The project includes experiment tracking, tests, Docker, and CI.

---

## Production Considerations

Before using StockSense AI with real enterprise data, the following should be added:

- Authentication and authorization
- Role-based access control
- Tenant-level data isolation
- Secure file storage
- Database persistence
- Request size limits
- Rate limiting
- Audit logging
- Monitoring and alerting
- Cloud deployment hardening
