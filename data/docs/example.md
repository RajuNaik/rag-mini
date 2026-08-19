# Example Knowledge Base

RAG Mini is a compact starter project for retrieval-augmented generation.

The ingestion step reads Markdown and text files, splits them into overlapping chunks, and stores those chunks in a JSONL index.

The retrieval step uses TF-IDF scoring to find chunks that match a user question.

The optional generation step sends the retrieved context to an OpenAI model and asks it to answer only from that context.
