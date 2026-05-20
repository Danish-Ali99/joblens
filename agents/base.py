"""Shared LLM factory used by every agent.

Provider is selected via the ``LLM_PROVIDER`` env var (``groq`` or ``openai``).
``max_tokens`` is capped to keep the parallel pipeline under ~5 seconds.
"""
import os

MAX_TOKENS = 800


def get_llm(temperature: float = 0.6):
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        from langchain_groq import ChatGroq
        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        return ChatGroq(model=model, temperature=temperature, max_tokens=MAX_TOKENS)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=temperature, max_tokens=MAX_TOKENS)

    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r}. Use 'groq' or 'openai'."
    )
