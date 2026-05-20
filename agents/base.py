"""Shared LLM factory and RAG-retrieval helper used by every agent.

Provider is selected via the ``LLM_PROVIDER`` env var (``groq`` or ``openai``).
``max_tokens`` is capped to keep the parallel pipeline under ~5 seconds.

``retrieve_resume_context`` is the RAG entry point each agent uses to fetch
only the resume chunks most relevant to its role — instead of stuffing the
full resume into every prompt.
"""
import os

MAX_TOKENS = 350


def get_llm(temperature: float = 0.6):
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        from langchain_groq import ChatGroq
        base = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=temperature,
            max_tokens=MAX_TOKENS,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        base = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
            max_tokens=MAX_TOKENS,
        )
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER={provider!r}. Use 'groq' or 'openai'."
        )

    # Retry once with exponential-jitter backoff on transient 429 / rate-limit
    # responses — important for Groq's free tier when 5 agents fan out.
    return base.with_retry(
        stop_after_attempt=2,
        wait_exponential_jitter=True,
    )


def retrieve_resume_context(state, query: str, k: int = 4) -> str:
    """RAG: return top-k resume chunks for an agent's role.

    Falls back to the full resume text if the retriever is missing or fails —
    keeps the demo robust on unusual resume formats.
    """
    retriever = state.get("retriever")
    if retriever is None:
        return state["resume_text"]
    try:
        chunks = retriever.retrieve(query, k=k)
        return "\n\n---\n\n".join(chunks)
    except Exception:
        return state["resume_text"]
