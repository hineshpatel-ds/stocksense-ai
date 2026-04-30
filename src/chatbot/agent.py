from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
from src.chatbot.llm_adapter import enhance_answer_with_llm

from src.chatbot.tools import (
    ToolResult,
    find_product_name,
    get_category_performance,
    get_forecast_summary,
    get_general_recommendations,
    get_inventory_summary,
    get_overstock_risk_products,
    get_product_recommendation,
    get_stockout_risk_products,
    get_top_products,
    get_waste_analysis,
)


@dataclass
class AgentResponse:
    """
    Final response returned by the AI agent.
    """

    answer: str
    intent: str
    tools_used: List[str]
    confidence: str


class InventoryAIAgent:
    """
    Tool-based inventory AI agent.

    This is the first version of our chatbot/agent.

    Important:
    The agent does not invent numbers.
    It routes the user question to trusted analytics tools.
    """

    def answer_question(self, question: str, context: Dict[str, Any]) -> AgentResponse:
        """
        Answer a user question using trusted tools.
        """

        normalized_question = question.lower().strip()

        if not normalized_question:
            return AgentResponse(
                answer="Please ask a question about inventory, products, forecast, risk, waste, or recommendations.",
                intent="empty_question",
                tools_used=[],
                confidence="Low",
            )

        product_name = find_product_name(
            question=question,
            product_performance=context["product_performance"],
        )

        intent = self._detect_intent(normalized_question, product_name)

        tool_result = self._run_tool(
            intent=intent,
            question=question,
            context=context,
            product_name=product_name,
        )

        tool_based_answer = self._format_final_answer(
            tool_result=tool_result,
            intent=intent,
        )

        llm_response = enhance_answer_with_llm(
            question=question,
            intent=intent,
            tool_answer=tool_based_answer,
        )

        final_answer = llm_response.text

        return AgentResponse(
            answer=final_answer,
            intent=intent,
            tools_used=[intent],
            confidence="High" if tool_result.success else "Low",
        )

    def _detect_intent(self, question: str, product_name: str | None) -> str:
        """
        Detect what the user is asking.

        This is a rule-based intent router.
        Later, we can replace or enhance this with an LLM.
        """

        if product_name and any(
            keyword in question
            for keyword in [
                "should",
                "recommend",
                "keep",
                "increase",
                "reduce",
                "order",
                "reorder",
                "buy",
                "next month",
                "next week",
                "forecast",
                "demand",
            ]
        ):
            return "product_recommendation"

        if any(keyword in question for keyword in ["summary", "overview", "health", "performance"]):
            return "inventory_summary"

        if any(keyword in question for keyword in ["top", "best", "highest revenue", "most revenue", "sold most"]):
            return "top_products"

        if any(keyword in question for keyword in ["stockout", "run out", "low stock", "shortage"]):
            return "stockout_risk"

        if any(keyword in question for keyword in ["overstock", "excess", "too much stock", "high stock"]):
            return "overstock_risk"

        if any(keyword in question for keyword in ["waste", "wastage", "loss", "spoiled"]):
            return "waste_analysis"

        if any(keyword in question for keyword in ["forecast", "future demand", "next month", "next week", "predicted demand"]):
            return "forecast_summary"

        if any(keyword in question for keyword in ["recommend", "recommendation", "action", "what should i do"]):
            return "general_recommendations"

        if any(keyword in question for keyword in ["category", "categories"]):
            return "category_performance"

        return "inventory_summary"

    def _run_tool(
        self,
        intent: str,
        question: str,
        context: Dict[str, Any],
        product_name: str | None,
    ) -> ToolResult:
        """
        Run the correct tool based on detected intent.
        """

        if intent == "inventory_summary":
            return get_inventory_summary(context)

        if intent == "top_products":
            return get_top_products(context)

        if intent == "stockout_risk":
            return get_stockout_risk_products(context)

        if intent == "overstock_risk":
            return get_overstock_risk_products(context)

        if intent == "waste_analysis":
            return get_waste_analysis(context)

        if intent == "forecast_summary":
            return get_forecast_summary(context)

        if intent == "general_recommendations":
            return get_general_recommendations(context)

        if intent == "category_performance":
            return get_category_performance(context)

        if intent == "product_recommendation":
            if product_name is None:
                return ToolResult(
                    success=False,
                    message=(
                        "Please mention the exact product name so I can analyze its forecast, "
                        "stock level, and recommendation."
                    ),
                )

            return get_product_recommendation(
                context=context,
                product_name=product_name,
            )

        return get_inventory_summary(context)

    def _format_final_answer(self, tool_result: ToolResult, intent: str) -> str:
        """
        Format final chatbot answer.

        Later, an LLM can rewrite this response more naturally.
        """

        if not tool_result.success:
            return f"I could not complete the analysis. {tool_result.message}"

        intro = self._get_intent_intro(intent)

        return f"{intro}\n\n{tool_result.message}"

    def _get_intent_intro(self, intent: str) -> str:
        """
        Add short business-friendly intro based on intent.
        """

        intros = {
            "inventory_summary": "Here is the current inventory overview based on uploaded data.",
            "top_products": "Here are the strongest products based on revenue performance.",
            "stockout_risk": "I checked which products may run out soon.",
            "overstock_risk": "I checked which products may have excess stock.",
            "waste_analysis": "I analyzed products contributing most to waste or loss.",
            "forecast_summary": "Here is the demand forecast summary.",
            "general_recommendations": "Here are the most important inventory actions to consider.",
            "category_performance": "Here is the category-level performance breakdown.",
            "product_recommendation": "I checked this product using KPIs, forecast, and recommendation rules.",
        }

        return intros.get(intent, "Here is the analysis based on uploaded inventory data.")