from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResponse:
    """
    Standard response format for any LLM provider.
    """

    success: bool
    text: str
    provider: str
    error: str | None = None


class LLMProvider(Protocol):
    """
    Common interface for all LLM providers.

    Any future LLM provider should implement this method.
    """

    provider_name: str

    def enhance_answer(
        self,
        question: str,
        intent: str,
        tool_answer: str,
    ) -> LLMResponse:
        ...


class NoLLMProvider:
    """
    Default provider.

    This does not call any external model.
    It simply returns the deterministic tool-based answer.
    """

    provider_name = "none"

    def enhance_answer(
        self,
        question: str,
        intent: str,
        tool_answer: str,
    ) -> LLMResponse:
        return LLMResponse(
            success=True,
            text=tool_answer,
            provider=self.provider_name,
        )


class LocalOllamaProvider:
    """
    Local LLM provider using Ollama.

    This is optional. If Ollama is not running, the system will safely fall back
    to the original tool-based answer.
    """

    provider_name = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2:1b",
        timeout_seconds: int = 30,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def enhance_answer(
        self,
        question: str,
        intent: str,
        tool_answer: str,
    ) -> LLMResponse:
        prompt = build_grounded_explanation_prompt(
            question=question,
            intent=intent,
            tool_answer=tool_answer,
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
            },
        }

        request = urllib.request.Request(
            url=f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                response_body = response.read().decode("utf-8")
                response_json = json.loads(response_body)

            generated_text = response_json.get("response", "").strip()

            if not generated_text:
                return LLMResponse(
                    success=False,
                    text=tool_answer,
                    provider=self.provider_name,
                    error="Ollama returned an empty response.",
                )

            return LLMResponse(
                success=True,
                text=generated_text,
                provider=self.provider_name,
            )

        except urllib.error.URLError as exc:
            return LLMResponse(
                success=False,
                text=tool_answer,
                provider=self.provider_name,
                error=f"Could not connect to Ollama: {exc}",
            )

        except Exception as exc:
            return LLMResponse(
                success=False,
                text=tool_answer,
                provider=self.provider_name,
                error=f"Unexpected LLM error: {exc}",
            )


def build_grounded_explanation_prompt(
    question: str,
    intent: str,
    tool_answer: str,
) -> str:
    """
    Build prompt for the LLM.

    The LLM is only allowed to rewrite and explain the verified tool answer.
    It must not invent new numbers or unsupported recommendations.
    """

    return f"""
You are StockSense AI, a professional inventory intelligence assistant.

Your task:
Rewrite the verified tool-based answer into clear, professional business language.

Strict rules:
1. Use only the information provided in the tool answer.
2. Do not invent new numbers, products, forecasts, or recommendations.
3. Do not mention calculations that are not already present.
4. If the tool answer says something is unavailable, keep that limitation clear.
5. Be concise, practical, and manager-friendly.

User question:
{question}

Detected intent:
{intent}

Verified tool answer:
{tool_answer}

Final business-friendly answer:
""".strip()


def build_llm_provider() -> LLMProvider:
    """
    Build LLM provider from environment variables.

    Default is NoLLMProvider, so the project works without any external AI setup.
    """

    provider_name = os.getenv("STOCKSENSE_LLM_PROVIDER", "none").strip().lower()

    if provider_name == "ollama":
        return LocalOllamaProvider(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "llama3.2:1b"),
        )

    return NoLLMProvider()


def enhance_answer_with_llm(
    question: str,
    intent: str,
    tool_answer: str,
) -> LLMResponse:
    """
    Enhance an answer using the configured LLM provider.

    If the LLM fails, return the original deterministic answer.
    """

    provider = build_llm_provider()
    llm_response = provider.enhance_answer(
        question=question,
        intent=intent,
        tool_answer=tool_answer,
    )

    if not llm_response.success:
        return LLMResponse(
            success=True,
            text=tool_answer,
            provider=llm_response.provider,
            error=llm_response.error,
        )

    return llm_response