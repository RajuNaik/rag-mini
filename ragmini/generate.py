from __future__ import annotations

from .retrieve import SearchResult


SYSTEM_PROMPT = """You answer questions using only the provided context.
If the answer is not in the context, say you do not know from the provided documents.
Include brief source citations using the chunk source names."""


def build_context(results: list[SearchResult]) -> str:
    parts = []
    for result in results:
        parts.append(f"Source: {result.chunk.source}\n{result.chunk.text}")
    return "\n\n---\n\n".join(parts)


def answer_with_openai(question: str, results: list[SearchResult], *, model: str = "gpt-4.1-mini") -> str:
    from openai import OpenAI

    client = OpenAI()
    context = build_context(results)
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response.output_text
