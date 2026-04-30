from src.chatbot.llm_adapter import (
    NoLLMProvider,
    build_grounded_explanation_prompt,
    build_llm_provider,
    enhance_answer_with_llm,
)


def test_no_llm_provider_returns_tool_answer():
    provider = NoLLMProvider()

    response = provider.enhance_answer(
        question="Give me inventory summary",
        intent="inventory_summary",
        tool_answer="Inventory health score is 80/100.",
    )

    assert response.success is True
    assert response.provider == "none"
    assert response.text == "Inventory health score is 80/100."


def test_build_grounded_prompt_contains_safety_rules():
    prompt = build_grounded_explanation_prompt(
        question="Should I reorder Veggie Burger?",
        intent="product_recommendation",
        tool_answer="Recommended reorder quantity is 100.",
    )

    assert "Do not invent new numbers" in prompt
    assert "Veggie Burger" in prompt
    assert "Recommended reorder quantity is 100" in prompt


def test_build_llm_provider_defaults_to_no_llm(monkeypatch):
    monkeypatch.delenv("STOCKSENSE_LLM_PROVIDER", raising=False)

    provider = build_llm_provider()

    assert provider.provider_name == "none"


def test_enhance_answer_with_llm_defaults_to_tool_answer(monkeypatch):
    monkeypatch.setenv("STOCKSENSE_LLM_PROVIDER", "none")

    response = enhance_answer_with_llm(
        question="Give me forecast summary",
        intent="forecast_summary",
        tool_answer="Total predicted demand is 500 units.",
    )

    assert response.success is True
    assert response.text == "Total predicted demand is 500 units."
    assert response.provider == "none"