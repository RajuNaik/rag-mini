# RAG Mini

A small Retrieval-Augmented Generation starter project.

It indexes local `.txt` and `.md` files, retrieves the most relevant chunks with a TF-IDF scorer, and can optionally ask an OpenAI model to answer using only the retrieved context.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Add documents to `data/docs`, then build the index:

```powershell
python -m ragmini ingest data/docs
```

Ask a retrieval-only question:

```powershell
python -m ragmini ask "What do these documents say about the project?"
```

Ask with an OpenAI-generated answer:

```powershell
$env:OPENAI_API_KEY = "sk-..."
python -m ragmini ask "What do these documents say about the project?" --generate
```

## Project Layout

```text
ragmini/
  __main__.py      CLI entrypoint
  ingest.py        document loading, chunking, and index writing
  retrieve.py      TF-IDF retrieval
  generate.py      optional OpenAI answer generation
data/
  docs/            put source documents here
  index/           generated index files
```

## Notes

- This is intentionally simple: no database, no server, no background services.
- The index is stored as JSONL in `data/index/chunks.jsonl`.
- OpenAI generation is optional. Retrieval works without any API key.
