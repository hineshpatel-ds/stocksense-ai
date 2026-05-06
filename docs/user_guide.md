# StockSense AI User Guide

## Overview

StockSense AI helps users analyze inventory data through a dashboard, API, forecasting engine, recommendation engine, and AI agent.

---

## Using the Dashboard

Run:

```bash
streamlit run app/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

---

## Uploading Data

Use the sidebar to upload a CSV or Excel file.

The file must include these required columns:

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

If no file is uploaded, the app can use the sample demo dataset.

---

## Dashboard Tabs

### Overview

Shows:

- Data quality status
- Executive summary
- KPI cards
- Revenue trend
- Top products
- Waste analysis
- Business insights

### Forecasting

Shows:

- Predicted demand
- Safety stock
- Recommended reorder quantity
- Forecast stockout risk
- Forecast confidence

### Recommendations

Shows:

- High priority recommendations
- Medium priority recommendations
- Low priority recommendations
- Product-specific action reasons

### AI Agent

Allows users to ask questions such as:

```text
Give me inventory summary
Which products are at stockout risk?
Which products have highest waste?
What are the top products by revenue?
What recommendations do you have?
Give me forecast summary
Should I reorder Veggie Burger next month?
```

### Products

Shows product-level performance.

### Risk Center

Shows stockout and overstock risk.

### Categories

Shows category-level performance.

### Data Quality

Shows validation errors, warnings, and cleaned data preview.

---

## Using the API

Run:

```bash
uvicorn api.main:app --reload
```

Open:

```text
http://localhost:8000/docs
```

Use Swagger UI to test:

- `/health`
- `/validate`
- `/analyze`
- `/ask`

---

## Using Docker

Run:

```bash
docker compose up --build
```

Open:

```text
Dashboard: http://localhost:8501
API Docs:  http://localhost:8000/docs
```

Stop:

```bash
docker compose down
```

---

## Running Tests

```bash
pytest
```

---

## Running MLflow

```bash
python scripts/run_mlflow_forecasting_experiment.py
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

---

## Data Privacy Note

Do not commit real company inventory data to GitHub.

Use:

```text
data/raw/
data/processed/
```

for local/private data.

These folders are ignored by Git.

---

## Troubleshooting

### Sample data is missing

Run:

```bash
python scripts/generate_sample_data.py
```

### Streamlit port already in use

Stop the old Streamlit process with `Ctrl + C`, or use another port:

```bash
streamlit run app/streamlit_app.py --server.port=8502
```

### FastAPI port already in use

Stop the previous backend process or change the port:

```bash
uvicorn api.main:app --reload --port 8001
```

### Docker containers are already running

Run:

```bash
docker compose down
docker compose up --build
```

---

## Recommended Workflow

1. Generate or upload inventory data
2. Check data quality score
3. Review executive dashboard
4. Check forecasting tab
5. Review recommendations
6. Ask the AI agent follow-up questions
7. Use insights to make inventory decisions
