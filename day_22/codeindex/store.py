"""Per-mode index folder: chunk metadata in meta.db (SQLite), vectors in vectors.faiss.

Another database name keeps its vectors beside it under the same stem (docs.db -> docs.faiss).

The FAISS vector id is chunks.id, the only link between the two stores.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import subprocess
from pathlib import Path, PurePosixPath
from typing import Callable

import faiss
import numpy as np

from . import CodeIndexError
from .chunker import Chunk

SKIP_DIRS = {".git", "venv", ".venv", "node_modules", "__pycache__"}
MAX_FILE_BYTES = 1_000_000
DEFAULT_DB = "meta.db"
REBUILD_HINT = "run: python -m codeindex index <repo_path> --chunking <mode> --rebuild"

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY, path TEXT UNIQUE, mtime REAL, sha256 TEXT);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY, file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
    language TEXT, kind TEXT, class_name TEXT, symbol_name TEXT, qualified_name TEXT, part INTEGER,
    start_line INTEGER, end_line INTEGER, content TEXT, content_hash TEXT);
CREATE INDEX IF NOT EXISTS chunks_file ON chunks(file_id);
CREATE TABLE IF NOT EXISTS summaries (content_hash TEXT PRIMARY KEY, model TEXT, summary TEXT);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def list_files(root: Path) -> list[str]:
    """Repo-relative paths: git ls-files in a git work tree, otherwise a directory walk."""
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--cached", "--others",
                              "--exclude-standard"], capture_output=True, check=True).stdout
        paths = [p for p in out.decode().split("\0") if p]
    except (OSError, subprocess.CalledProcessError):
        paths = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            paths += [(Path(dirpath) / f).relative_to(root).as_posix() for f in filenames]
    return sorted(p for p in paths if not SKIP_DIRS & set(PurePosixPath(p).parts[:-1]))


class Store:
    def __init__(self, directory: Path, db_name: str = DEFAULT_DB):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        db_path = self.dir / db_name
        self.faiss_path = self.dir / "vectors.faiss" if db_name == DEFAULT_DB else db_path.with_suffix(".faiss")
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.index = faiss.read_index(str(self.faiss_path)) if self.faiss_path.exists() else None

    def meta(self) -> dict[str, str]:
        return {r["key"]: r["value"] for r in self.conn.execute("SELECT key, value FROM meta")}

    def check(self, **expected: str) -> None:
        """Stop if the index was built with other settings, or its two stores disagree."""
        stored = self.meta()
        diffs = [f"{k} is {stored[k]!r}, not {v!r}" for k, v in expected.items()
                 if k in stored and stored[k] != str(v)]
        if diffs:
            raise CodeIndexError(f"{self.dir} was built with other settings ({'; '.join(diffs)}); {REBUILD_HINT}")
        chunks = self.conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        vectors = self.index.ntotal if self.index is not None else 0
        if chunks != vectors:
            raise CodeIndexError(f"{self.dir} is inconsistent ({chunks} chunks, {vectors} vectors); {REBUILD_HINT}")

    def sync(self, root: Path, mode: str, chunk_fn: Callable[[str, str], list[Chunk]], embedder,
             summarizer=None, rebuild: bool = False) -> dict[str, int]:
        """Bring the index up to date with the files under root.

        All chunking, summary and embedding calls happen before any write; the FAISS file is
        replaced before the SQLite transaction commits, and any failure rolls the transaction back.
        """
        root = Path(root).resolve()
        if not rebuild:
            self.check(chunking=mode, model=embedder.model, repo_root=str(root))
        try:
            if rebuild:  # summaries survive, so a rebuild reuses them
                self.conn.execute("DELETE FROM files")
                self.conn.execute("DELETE FROM meta")
                self.index = None
            known = {r["path"]: r for r in self.conn.execute("SELECT id, path, mtime, sha256 FROM files")}
            seen, touched, todo = set(), [], []
            for rel in list_files(root):
                try:
                    st = (root / rel).stat()
                except OSError:  # listed by git but deleted from the work tree
                    continue
                if st.st_size > MAX_FILE_BYTES:
                    continue
                row = known.get(rel)
                if row and row["mtime"] == st.st_mtime:
                    seen.add(rel)
                    continue
                data = (root / rel).read_bytes()
                if b"\0" in data[:8192]:  # binary
                    continue
                seen.add(rel)
                sha = hashlib.sha256(data).hexdigest()
                if row and row["sha256"] == sha:
                    touched.append((st.st_mtime, row["id"]))
                    continue
                chunks = chunk_fn(rel, data.decode("utf-8", errors="replace"))
                todo.append((rel, row["id"] if row else None, st.st_mtime, sha, chunks))
            gone = [row["id"] for rel, row in known.items() if rel not in seen]
            chunks = [c for *_, cs in todo for c in cs]
            texts = summarizer.embed_texts(self.conn, chunks) if summarizer else [c.embed_text() for c in chunks]
            vectors = embedder.embed_documents(texts) if texts else None
            stored_dim = self.meta().get("dim")
            if vectors is not None and stored_dim and int(stored_dim) != embedder.dim:
                raise CodeIndexError(f"{self.dir} has dimension {stored_dim}, the model returns "
                                     f"{embedder.dim}; {REBUILD_HINT}")

            # Everything is computed; now write.
            replaced = [fid for _, fid, *_ in todo if fid is not None] + gone
            old_ids = [r[0] for r in self.conn.execute(
                f"SELECT id FROM chunks WHERE file_id IN ({','.join('?' * len(replaced))})", replaced)]
            if old_ids:
                self.index.remove_ids(np.array(old_ids, dtype="int64"))
            self.conn.executemany("DELETE FROM files WHERE id = ?", [(i,) for i in gone])
            self.conn.executemany("UPDATE files SET mtime = ? WHERE id = ?", touched)
            new_ids = []
            for rel, fid, mtime, sha, file_chunks in todo:
                if fid is None:
                    fid = self.conn.execute("INSERT INTO files (path, mtime, sha256) VALUES (?, ?, ?)",
                                            (rel, mtime, sha)).lastrowid
                else:
                    self.conn.execute("DELETE FROM chunks WHERE file_id = ?", (fid,))
                    self.conn.execute("UPDATE files SET mtime = ?, sha256 = ? WHERE id = ?", (mtime, sha, fid))
                for c in file_chunks:
                    new_ids.append(self.conn.execute(
                        "INSERT INTO chunks (file_id, language, kind, class_name, symbol_name, qualified_name,"
                        " part, start_line, end_line, content, content_hash) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (fid, c.language, c.kind, c.class_name, c.symbol_name, c.qualified_name, c.part,
                         c.start_line, c.end_line, c.content, c.content_hash)).lastrowid)
            if vectors is not None:
                if self.index is None:
                    self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(embedder.dim))
                self.index.add_with_ids(vectors, np.array(new_ids, dtype="int64"))
            meta = {"chunking": mode, "model": embedder.model, "repo_root": str(root)}
            if self.index is not None:
                meta["dim"] = str(self.index.d)
            self.conn.executemany("INSERT OR REPLACE INTO meta VALUES (?, ?)", meta.items())
            if rebuild or old_ids or new_ids:
                self._write_index()
            self.conn.commit()
        except BaseException:
            self.conn.rollback()
            self.index = faiss.read_index(str(self.faiss_path)) if self.faiss_path.exists() else None
            raise
        return {"files_indexed": len(todo), "files_deleted": len(gone), "chunks_embedded": len(chunks),
                "chunks_removed": len(old_ids)}

    def _write_index(self) -> None:
        if self.index is None:  # a rebuild of a tree with no chunks
            self.faiss_path.unlink(missing_ok=True)
            return
        tmp = self.faiss_path.with_name(self.faiss_path.name + ".tmp")
        faiss.write_index(self.index, str(tmp))
        os.replace(tmp, self.faiss_path)

    def search(self, query_vector: np.ndarray, k: int, path_prefix: str | None = None,
               kind: str | None = None, class_name: str | None = None,
               merge_parts: bool = False) -> list[dict]:
        if self.index is None or self.index.ntotal == 0:
            return []
        filtered = bool(path_prefix or kind or class_name)
        fetch = min(self.index.ntotal, k * 5 if filtered or merge_parts else k)
        scores, ids = self.index.search(query_vector.reshape(1, -1).astype("float32"), fetch)
        score_of = {int(i): float(s) for s, i in zip(scores[0], ids[0]) if i != -1}
        sql = (f"SELECT c.*, f.path AS file_path FROM chunks c JOIN files f ON f.id = c.file_id"
               f" WHERE c.id IN ({','.join('?' * len(score_of))})")
        params: list = list(score_of)
        if path_prefix:
            sql += " AND substr(f.path, 1, ?) = ?"
            params += [len(path_prefix), path_prefix]
        if kind:
            sql += " AND c.kind = ?"
            params.append(kind)
        if class_name:  # the class itself, or a nested class by its last name
            sql += " AND (c.class_name = ? OR substr(c.class_name, -?) = ?)"
            params += [class_name, len(class_name) + 1, "." + class_name]
        rows = sorted(({**dict(r), "score": score_of[r["id"]]} for r in self.conn.execute(sql, params)),
                      key=lambda r: -r["score"])
        if merge_parts:  # one result per split unit, at its best-scoring part
            seen, merged = set(), []
            for r in rows:
                key = (r["file_path"], r["qualified_name"]) if r["part"] is not None else r["id"]
                if key not in seen:
                    seen.add(key)
                    merged.append(r)
            rows = merged
        return rows[:k]

    def stats(self) -> dict:
        count = lambda sql: self.conn.execute(sql).fetchone()[0]  # noqa: E731
        return {
            **self.meta(),
            "files": count("SELECT count(*) FROM files"),
            "chunks": count("SELECT count(*) FROM chunks"),
            "vectors": self.index.ntotal if self.index is not None else 0,
            "kinds": {r[0]: r[1] for r in self.conn.execute("SELECT kind, count(*) FROM chunks GROUP BY kind")},
            "summaries": count("SELECT count(*) FROM summaries"),
        }
