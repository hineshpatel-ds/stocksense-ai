# StockSense AI

StockSense AI is a production-style AI inventory intelligence platform that helps businesses upload inventory data, validate data quality, analyze inventory performance, forecast demand, generate business recommendations, and ask inventory questions through an AI agent.

![CI](https://github.com/hineshpatel-ds/stocksense-ai/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## Project Purpose

Many businesses manage inventory using spreadsheets, but raw inventory data does not automatically tell managers what to do next.

StockSense AI converts inventory data into decision-ready intelligence:

- What products are performing well?
- Which products are at risk of stockout?
- Which products are overstocked?
- Where is waste or loss happening?
- What should be reordered?
- What is the expected future demand?
- What actions should a manager take?

---

## Key Features

### Data Validation

- Validates uploaded CSV or Excel files
- Checks required columns
- Detects missing values
- Detects invalid dates
- Detects negative stock values
- Detects inventory equation mismatch
- Generates a data quality score

### KPI Analytics

- Total revenue
- Units sold
- Waste value
- Latest inventory value
- Sell-through rate
- Waste rate
- Stock turnover
- Inventory health score
- Product-level performance
- Category-level performance

### Demand Forecasting

- Baseline moving-average forecasting
- 30-day demand prediction
- Safety stock calculation
- Recommended reorder quantity
- Forecast stockout risk
- Forecast confidence level

### Recommendation Engine

- Stockout prevention recommendations
- Overstock reduction recommendations
- Waste reduction recommendations
- Product review recommendations
- Priority and confidence levels
- Business-friendly explanations

### AI Inventory Agent

- Tool-based AI agent
- Answers inventory questions using trusted analytics tools
- Supports summary, risk, forecast, waste, recommendation, and product-specific questions
- LLM-ready adapter architecture
- Works with no external LLM by default
- Optional local Ollama support

### MLOps

- MLflow experiment tracking
- Forecasting metrics: MAE, RMSE, MAPE
- Experiment parameters and artifacts
- Dockerized API and dashboard
- GitHub Actions CI
- Automated tests with Pytest

### Data Persistence

- SQLite-backed inventory repository
- Saves validated upload batches for later reuse
- List and re-analyze past uploads without re-uploading files
- Ask the AI agent questions about a previously saved batch

### Drift Monitoring

- Compares current inventory data against a reference window
- Numeric feature drift scoring
- Categorical feature drift scoring
- Drift level classification (low, medium, high)
- Dashboard tab for monitoring status at a glance

### Upload Security

- File extension and MIME type checks
- Upload size limits
- Metadata validation before parsing untrusted files

---

## Architecture

```text
User / Manager
   |
   v
Streamlit Dashboard
   |
   v
FastAPI Backend
   |
   v
Core Python Services
   |
   |-- Data Validation Engine
   |-- Upload Security Layer
   |-- KPI Analytics Engine
   |-- Forecasting Engine
   |-- Recommendation Engine
   |-- Drift Monitoring Engine
   |-- AI Agent Layer
   |-- LLM Adapter
   |-- MLflow Tracking
   |-- SQLite Persistence Layer
```

Detailed architecture is available in:

```text
docs/architecture.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Dashboard | Streamlit |
| Backend API | FastAPI |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn baseline approach |
| Visualization | Plotly |
| AI Agent | Tool-based Python agent |
| LLM Adapter | No-LLM mode, optional Ollama |
| Experiment Tracking | MLflow |
| Testing | Pytest |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |

---

## Project Structure

```text
stocksense-ai/
├── api/
│   └── main.py
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
├── docs/
│   ├── architecture.md
│   ├── api_documentation.md
│   ├── model_card.md
│   └── user_guide.md
├── notebooks/
├── scripts/
│   ├── generate_sample_data.py
│   ├── run_database_demo.py
│   ├── run_forecasting_demo.py
│   ├── run_kpi_demo.py
│   ├── run_mlflow_forecasting_experiment.py
│   ├── run_monitoring_demo.py
│   └── run_recommendation_demo.py
├── src/
│   ├── analytics/
│   ├── chatbot/
│   ├── data/
│   ├── database/
│   ├── models/
│   ├── monitoring/
│   ├── recommendations/
│   └── security/
├── tests/
├── .github/
│   └── workflows/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/hineshpatel-ds/stocksense-ai.git
cd stocksense-ai
```

### 2. Create virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Mac/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Generate sample data

```bash
python scripts/generate_sample_data.py
```

---

## Run Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

---

## Run FastAPI Backend

```bash
uvicorn api.main:app --reload
```

Open API docs:

```text
http://localhost:8000/docs
```

Health check:

```text
http://localhost:8000/health
```

---

## Run with Docker

Make sure Docker Desktop is running.

```bash
docker compose up --build
```

Open:

```text
Dashboard: http://localhost:8501
API:       http://localhost:8000
API Docs:  http://localhost:8000/docs
```

Stop containers:

```bash
docker compose down
```

Run tests inside Docker:

```bash
docker compose run --rm api pytest
```

---

## Run Tests

```bash
pytest
```

---

## Run MLflow Experiment

```bash
python scripts/run_mlflow_forecasting_experiment.py
```

Start MLflow UI:

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

---

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check API health |
| POST | `/validate` | Validate uploaded inventory file |
| POST | `/analyze` | Run full inventory analysis |
| POST | `/ask` | Ask AI agent a business question |
| POST | `/uploads/save` | Validate and save an upload batch to SQLite |
| GET | `/uploads` | List recent saved upload batches |
| GET | `/uploads/{batch_id}/analyze` | Run full analysis on a saved batch |
| POST | `/uploads/{batch_id}/ask` | Ask the AI agent about a saved batch |

More details are available in:

```text
docs/api_documentation.md
```

---

## Example Questions for AI Agent

```text
Give me inventory summary
Which products are at stockout risk?
Which products have highest waste?
What are the top products by revenue?
What recommendations do you have?
Give me forecast summary
Should I reorder Veggie Burger next month?
```

---

## Input Data Format

Required columns:

```text
date
store_id
product_id
product_name
category
opening_stock
purchased_quantity
sold_quantity
wasted_quantity
closing_stock
unit_price
```

Inventory equation:

```text
closing_stock = opening_stock + purchased_quantity - sold_quantity - wasted_quantity
```

---

## MLOps Highlights

- Modular ML project structure
- Experiment tracking with MLflow
- Automated testing with Pytest
- CI pipeline with GitHub Actions
- Dockerized API and dashboard
- LLM-ready architecture
- Clean separation between frontend, backend, and core services

---

## Resume Highlights

This project demonstrates:

- End-to-end AI product development
- Data validation and analytics engineering
- Demand forecasting
- Recommendation systems
- AI agent design
- MLOps practices
- FastAPI backend development
- Streamlit dashboard development
- Dockerization
- CI/CD with GitHub Actions

---

## Current Status

MVP modules completed:

- Project setup
- Data validation engine
- KPI analytics engine
- Professional Streamlit dashboard
- Demand forecasting engine
- Recommendation engine
- Tool-based AI agent
- Pluggable LLM adapter
- MLflow tracking
- FastAPI backend
- Docker setup
- GitHub Actions CI
- SQLite persistence layer for uploaded inventory batches
- Inventory drift monitoring
- Upload security hardening

---

## Roadmap

Future improvements:

- Production database (PostgreSQL) for multi-tenant scale
- User authentication
- Multi-company workspace support
- Advanced forecasting models
- Model registry
- Scheduled retraining
- Cloud deployment
- Report export as PDF
- React frontend
- Expanded production-grade security controls (auth, rate limiting, audit logging)

---

## Disclaimer

This project is a portfolio and proof-of-concept system. It is designed to demonstrate production-style architecture and MLOps practices. Before using it with real enterprise data, additional security, privacy, compliance, monitoring, and deployment hardening would be required.
