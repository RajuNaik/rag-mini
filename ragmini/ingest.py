from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS = {".md", ".txt"}
DEFAULT_INDEX_PATH = Path("data/index/chunks.jsonl")


@dataclass
class Chunk:
    id: str
    source: str
    text: str


def read_documents(root: Path) -> list[tuple[Path, str]]:
    documents: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                documents.append((path, text))
    return documents


def chunk_text(text: str, *, chunk_size: int = 900, overlap: int = 160) -> list[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def build_chunks(documents: list[tuple[Path, str]], source_root: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path, text in documents:
        relative = path.relative_to(source_root).as_posix()
        for index, chunk in enumerate(chunk_text(text), start=1):
            chunks.append(Chunk(id=f"{relative}#{index}", source=relative, text=chunk))
    return chunks


def write_index(chunks: list[Chunk], index_path: Path = DEFAULT_INDEX_PATH) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def ingest(source_dir: Path, index_path: Path = DEFAULT_INDEX_PATH) -> int:
    documents = read_documents(source_dir)
    chunks = build_chunks(documents, source_dir)
    write_index(chunks, index_path)
    return len(chunks)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a RAG Mini chunk index.")
    parser.add_argument("source_dir", type=Path, help="Folder containing .txt and .md files.")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH, help="Index output path.")
    args = parser.parse_args(argv)

    count = ingest(args.source_dir, args.index)
    print(f"Indexed {count} chunks into {args.index}")


if __name__ == "__main__":
    main()
