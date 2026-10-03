"""Local semantic code search: Ollama embeddings in FAISS, chunk metadata in SQLite."""


class CodeIndexError(Exception):
    """An expected failure the CLI reports as a message instead of a traceback."""
