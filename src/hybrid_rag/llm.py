from __future__ import annotations

from collections.abc import Sequence

from .models import SearchResult

DEFAULT_LLM_MODEL = "gpt-4o-mini"


def _client(api_key: str, base_url: str | None = None):
    if not api_key.strip():
        raise ValueError("Enter an LLM API key to generate an answer.")
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("LLM support requires the openai package. Install requirements.txt.") from exc
    kwargs = {"api_key": api_key.strip()}
    if base_url and base_url.strip():
        kwargs["base_url"] = base_url.strip()
    return OpenAI(**kwargs)


def source_label(result: SearchResult) -> str:
    chunk = result.chunk
    location = []
    if chunk.section:
        location.append(f"section {chunk.section}")
    if chunk.page is not None:
        location.append(f"page {chunk.page}")
    location_text = ", ".join(location) or "location not specified"
    return f"{chunk.source} — {location_text} — chunk {chunk.chunk_id}"


def answer_with_openai(
    question: str,
    results: Sequence[SearchResult],
    api_key: str,
    model: str = DEFAULT_LLM_MODEL,
    base_url: str | None = None,
) -> str:
    """Generate a grounded answer; every factual claim must cite [S#]."""
    if not results:
        return "I don't know based on the provided documents."

    passages = []
    for position, result in enumerate(results[:5], start=1):
        passages.append(
            f"[S{position}] {source_label(result)}\n{result.chunk.text}"
        )

    system_prompt = (
        "You answer questions using only the supplied document passages. "
        "Cite every factual claim with one or more passage IDs in square brackets, "
        "for example [S1] or [S1][S3]. Do not use outside knowledge. "
        "If the passages do not support an answer, reply exactly: "
        "I don't know based on the provided documents. Do not invent citations."
    )
    user_prompt = (
        f"Question: {question.strip()}\n\n"
        "Document passages:\n\n"
        + "\n\n---\n\n".join(passages)
        + "\n\nAnswer concisely. Put citations next to the sentence or claim they support."
    )

    response = _client(api_key, base_url).chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content
    return (content or "I don't know based on the provided documents.").strip()


def rewrite_query_with_openai(
    question: str,
    api_key: str,
    model: str = DEFAULT_LLM_MODEL,
    base_url: str | None = None,
) -> str:
    """Optionally clarify a vague retrieval query without answering it."""
    response = _client(api_key, base_url).chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Rewrite a vague document-search query into one concise search query. "
                    "Preserve the user's intent and named entities. Do not answer the query, "
                    "add assumptions, or introduce facts not present in the original. "
                    "Return only the rewritten query."
                ),
            },
            {"role": "user", "content": question.strip()},
        ],
    )
    return (response.choices[0].message.content or question).strip()
