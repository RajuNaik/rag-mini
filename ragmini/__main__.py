from __future__ import annotations

import argparse
from pathlib import Path

from .ingest import DEFAULT_INDEX_PATH, ingest
from .retrieve import search


def cmd_ingest(args: argparse.Namespace) -> None:
    count = ingest(args.source_dir, args.index)
    print(f"Indexed {count} chunks into {args.index}")


def cmd_ask(args: argparse.Namespace) -> None:
    results = search(args.question, index_path=args.index, limit=args.limit)
    if not results:
        print("No matching context found.")
        return

    if args.generate:
        from .generate import answer_with_openai

        print(answer_with_openai(args.question, results, model=args.model))
        return

    for position, result in enumerate(results, start=1):
        preview = result.chunk.text.replace("\n", " ")
        if len(preview) > 320:
            preview = preview[:317] + "..."
        print(f"{position}. {result.chunk.source} score={result.score:.3f}")
        print(f"   {preview}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Small local RAG starter.")
    subparsers = parser.add_subparsers(required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Index local documents.")
    ingest_parser.add_argument("source_dir", type=Path)
    ingest_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    ingest_parser.set_defaults(func=cmd_ingest)

    ask_parser = subparsers.add_parser("ask", help="Ask a question against the index.")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    ask_parser.add_argument("--limit", type=int, default=4)
    ask_parser.add_argument("--generate", action="store_true", help="Use OpenAI to generate an answer.")
    ask_parser.add_argument("--model", default="gpt-4.1-mini")
    ask_parser.set_defaults(func=cmd_ask)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
