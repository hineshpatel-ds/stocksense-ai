# Forecasting Model Card

## Model Name

Moving Average Baseline Forecasting Engine

---

## Purpose

The forecasting engine predicts future product demand and estimates reorder quantities for inventory planning.

The goal is to help managers answer questions such as:

- How much demand should we expect next month?
- Which products may run out soon?
- How much should we reorder?
- Is demand increasing, decreasing, or stable?

---

## Current Model Type

The current version uses an explainable moving-average baseline.

This is not the final advanced model. It is used as a baseline so future models can be compared against it.

---

## Inputs

Required columns:

```text
date
product_id
product_name
category
sold_quantity
closing_stock
```

The forecasting engine also uses product-level KPI outputs such as current stock.

---

## Outputs

For each product:

- Forecast horizon
- Current stock
- Average daily demand
- Recent average daily demand
- Predicted demand
- Safety stock
- Recommended reorder quantity
- Trend direction
- Forecast stockout risk
- Confidence level

---

## Forecasting Logic

The baseline forecasting model:

1. Aggregates daily demand by product
2. Calculates average daily demand
3. Calculates recent moving average demand
4. Compares recent demand with previous demand
5. Applies capped trend adjustment
6. Predicts future demand over a selected horizon
7. Calculates safety stock using demand volatility
8. Calculates reorder quantity

Formula:

```text
recommended_reorder_quantity = predicted_demand + safety_stock - current_stock
```

If the value is negative, reorder quantity is set to zero.

---

## Evaluation Metrics

The project evaluates forecasting using:

- MAE
- RMSE
- MAPE

### MAE

Mean Absolute Error shows average unit-level forecast error.

### RMSE

Root Mean Squared Error penalizes large forecast errors more strongly.

### MAPE

Mean Absolute Percentage Error shows forecast error as a percentage.

---

## Experiment Tracking

MLflow is used to track:

- Forecasting method
- Recent window size
- Test window size
- Data quality score
- MAE
- RMSE
- MAPE
- Evaluated products
- Forecast evaluation artifacts

---

## Strengths

- Simple and explainable
- Easy to debug
- Good baseline for comparison
- Works without expensive infrastructure
- Suitable for early MVP development

---

## Limitations

- Does not currently use external factors such as promotions, holidays, weather, or local events
- Does not use advanced ML models yet
- May not perform well with very irregular demand
- Requires enough historical data for better reliability
- Does not yet support probabilistic forecasting
- Does not yet model different stores separately in an advanced way

---

## Future Improvements

- Random Forest forecasting
- XGBoost or LightGBM models
- Store-level forecasting
- Promotion-aware forecasting
- Seasonality-aware forecasting
- Model registry
- Drift monitoring
- Scheduled retraining
- Probabilistic confidence intervals

---

## Responsible Use Notes

Forecasting outputs should support human decision-making, not replace business judgment completely.

Managers should consider:

- Supplier delays
- Promotions
- Holidays
- Weather
- Local events
- Product expiry
- Business strategy

before making final purchasing decisions.
