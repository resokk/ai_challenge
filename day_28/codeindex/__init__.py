"""Local semantic code search: transformers embeddings in FAISS, chunk metadata in SQLite."""

import os

# faiss-cpu and torch each bundle their own OpenMP runtime; on macOS the second one to load aborts the
# process (or segfaults) unless this is set before either loads. store.py keeps faiss single-threaded,
# so only torch runs OpenMP code.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


class CodeIndexError(Exception):
    """An expected failure the CLI reports as a message instead of a traceback."""
