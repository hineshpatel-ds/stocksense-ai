import pandas as pd

from src.chatbot.agent import InventoryAIAgent


def create_agent_context():
    product_performance = pd.DataFrame(
        {
            "product_id": ["P001", "P002"],
            "product_name": ["Veggie Burger", "Cola 500ml"],
            "category": ["Food", "Beverage"],
            "total_revenue": [1000.0, 500.0],
            "total_units_sold": [200, 80],
            "total_units_wasted": [5, 20],
            "total_waste_value": [25.0, 50.0],
            "current_stock": [20, 500],
            "avg_daily_demand": [6.5, 2.0],
            "days_of_stock_left": [3.0, 250.0],
            "sell_through_rate": [0.66, 0.20],
            "waste_rate": [0.02, 0.20],
            "stockout_risk": ["High", "Low"],
            "overstock_risk": ["Low", "High"],
            "product_health_score": [75, 45],
        }
    )

    category_performance = pd.DataFrame(
        {
            "category": ["Food", "Beverage"],
            "total_revenue": [1000.0, 500.0],
            "total_units_sold": [200, 80],
        }
    )

    recommendation_df = pd.DataFrame(
        {
            "priority": ["High", "High"],
            "product_name": ["Veggie Burger", "Cola 500ml"],
            "category": ["Food", "Beverage"],
            "recommendation_type": ["Stockout Prevention", "Overstock Reduction"],
            "action": ["Increase stock for Veggie Burger.", "Reduce future purchase quantity for Cola 500ml."],
            "reason": [
                "Veggie Burger has high stockout risk.",
                "Cola 500ml has high overstock risk.",
            ],
            "confidence": ["High", "High"],
            "current_stock": [20, 500],
            "avg_daily_demand": [6.5, 2.0],
            "days_of_stock_left": [3.0, 250.0],
            "product_health_score": [75, 45],
        }
    )

    forecast_df = pd.DataFrame(
        {
            "product_name": ["Veggie Burger", "Cola 500ml"],
            "predicted_demand": [190, 60],
            "safety_stock": [20, 10],
            "recommended_reorder_quantity": [190, 0],
            "forecast_stockout_risk": ["High", "Low"],
            "confidence": ["Medium", "Medium"],
        }
    )

    return {
        "summary_metrics": {
            "inventory_health_score": 78,
            "total_revenue": 1500.0,
            "total_units_sold": 280,
            "total_waste_value": 75.0,
        },
        "risk_summary": {
            "high_stockout_risk_products": 1,
            "high_overstock_risk_products": 1,
        },
        "product_performance": product_performance,
        "category_performance": category_performance,
        "recommendation_df": recommendation_df,
        "forecast_df": forecast_df,
        "forecast_summary": {
            "total_products_forecasted": 2,
            "total_predicted_demand": 250,
            "total_recommended_reorder": 190,
            "high_forecast_stockout_risk": 1,
        },
    }


def test_agent_answers_inventory_summary():
    agent = InventoryAIAgent()
    context = create_agent_context()

    response = agent.answer_question("Give me inventory summary", context)

    assert response.intent == "inventory_summary"
    assert "Inventory health score" in response.answer


def test_agent_answers_top_products():
    agent = InventoryAIAgent()
    context = create_agent_context()

    response = agent.answer_question("Which are the top products?", context)

    assert response.intent == "top_products"
    assert "Veggie Burger" in response.answer


def test_agent_answers_stockout_risk():
    agent = InventoryAIAgent()
    context = create_agent_context()

    response = agent.answer_question("Which products are at stockout risk?", context)

    assert response.intent == "stockout_risk"
    assert "Veggie Burger" in response.answer


def test_agent_answers_product_recommendation():
    agent = InventoryAIAgent()
    context = create_agent_context()

    response = agent.answer_question("Should I reorder Veggie Burger next month?", context)

    assert response.intent == "product_recommendation"
    assert "Veggie Burger" in response.answer
    assert "Recommended reorder quantity" in response.answer


def test_agent_answers_waste_analysis():
    agent = InventoryAIAgent()
    context = create_agent_context()

    response = agent.answer_question("Which product has highest waste?", context)

    assert response.intent == "waste_analysis"
    assert "waste" in response.answer.lower()


def test_agent_handles_empty_question():
    agent = InventoryAIAgent()
    context = create_agent_context()

    response = agent.answer_question("", context)

    assert response.intent == "empty_question"
    assert response.confidence == "Low"