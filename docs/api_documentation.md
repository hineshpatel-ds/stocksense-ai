# StockSense AI API Documentation

The backend API is built using FastAPI.

Run locally:

```bash
uvicorn api.main:app --reload
```

Open Swagger UI:

```text
http://localhost:8000/docs
```

---

## GET /health

Checks whether the API service is running.

### Example Response

```json
{
  "status": "healthy",
  "service": "stocksense-ai-api",
  "version": "0.1.0"
}
```

---

## POST /validate

Validates an uploaded inventory CSV or Excel file.

### Input

Multipart form data:

```text
file: CSV or Excel inventory file
```

### Output

```json
{
  "is_valid": true,
  "data_quality_score": 100,
  "errors": [],
  "warnings": [],
  "row_count": 1440,
  "columns": ["date", "store_id", "product_id"]
}
```

### Purpose

This endpoint is useful when a user only wants to check whether their inventory file is valid before running full analysis.

---

## POST /analyze

Runs the full StockSense AI analysis pipeline.

### Pipeline

```text
File upload
 → Data validation
 → KPI analytics
 → Forecasting
 → Recommendations
 → JSON response
```

### Input

Multipart form data:

```text
file: CSV or Excel inventory file
```

### Output Includes

- Data quality score
- Summary metrics
- Risk summary
- Forecast summary
- Recommendation summary
- Product performance
- Category performance
- Forecast results
- Recommendations

---

## POST /ask

Allows a user to ask the AI agent a business question about uploaded inventory data.

### Input

Multipart form data:

```text
question: business question
file: CSV or Excel inventory file
```

### Example Questions

```text
Give me inventory summary
Which products are at stockout risk?
Which products have highest waste?
What recommendations do you have?
Should I reorder Veggie Burger next month?
```

### Example Response

```json
{
  "is_valid": true,
  "question": "Give me inventory summary",
  "answer": "Here is the current inventory overview...",
  "intent": "inventory_summary",
  "tools_used": ["inventory_summary"],
  "confidence": "High"
}
```

---

## Error Handling

Unsupported file formats return:

```json
{
  "detail": "Unsupported file format. Please upload CSV or Excel file."
}
```

Invalid inventory data returns validation errors and prevents full analysis.

---

## Manual Testing with curl

Health check:

```bash
curl -X GET http://127.0.0.1:8000/health
```

Validate file:

```bash
curl -X POST "http://127.0.0.1:8000/validate" \
  -F "file=@data/sample/sample_inventory.csv"
```

Ask agent:

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -F "question=Give me inventory summary" \
  -F "file=@data/sample/sample_inventory.csv"
```

On Windows PowerShell, use `curl.exe`:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/ask" `
  -F "question=Give me inventory summary" `
  -F "file=@data/sample/sample_inventory.csv"
```

---

## Production Notes

For production:

- Restrict CORS origins
- Add authentication
- Add request size limits
- Add rate limiting
- Store uploaded files securely
- Add audit logging
- Use a production database
- Validate file size and MIME type
- Use HTTPS
