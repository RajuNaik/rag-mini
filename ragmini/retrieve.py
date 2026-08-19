from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .ingest import DEFAULT_INDEX_PATH, Chunk


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def load_chunks(index_path: Path = DEFAULT_INDEX_PATH) -> list[Chunk]:
    chunks: list[Chunk] = []
    with index_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunks.append(Chunk(**json.loads(line)))
    return chunks


def idf_by_term(chunks: list[Chunk]) -> dict[str, float]:
    document_frequency: Counter[str] = Counter()
    for chunk in chunks:
        document_frequency.update(set(tokenize(chunk.text)))

    total = len(chunks)
    return {
        term: math.log((1 + total) / (1 + frequency)) + 1
        for term, frequency in document_frequency.items()
    }


def vector_for(text: str, idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(tokenize(text))
    if not counts:
        return {}
    total = sum(counts.values())
    return {
        term: (count / total) * idf.get(term, 0.0)
        for term, count in counts.items()
    }


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    shared = set(left).intersection(right)
    numerator = sum(left[term] * right[term] for term in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def search(query: str, *, index_path: Path = DEFAULT_INDEX_PATH, limit: int = 4) -> list[SearchResult]:
    chunks = load_chunks(index_path)
    idf = idf_by_term(chunks)
    query_vector = vector_for(query, idf)

    results = []
    for chunk in chunks:
        score = cosine_similarity(query_vector, vector_for(chunk.text, idf))
        if score > 0:
            results.append(SearchResult(chunk=chunk, score=score))

    return sorted(results, key=lambda result: result.score, reverse=True)[:limit]
