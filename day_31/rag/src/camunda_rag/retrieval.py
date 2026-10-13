"""Hybrid retrieval: exact identifiers + SQLite FTS5 (BM25) + BGE-M3/FAISS, fused with RRF."""

from __future__ import annotations

import re
import sqlite3
from typing import Any

import numpy as np

RRF_K = 60
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_$.#]*|[А-Яа-яЁё]+|\d+")


def _fts_query(q: str) -> str | None:
    toks = [t for t in re.findall(r"[\w]+", q, flags=re.UNICODE) if len(t) > 1]
    if not toks:
        return None
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(toks))


class HybridRetriever:
    def __init__(self, db: sqlite3.Connection, faiss_index=None, embedder=None) -> None:
        self.db = db
        self.index = faiss_index
        self.embedder = embedder

    def _exact(self, query: str, limit: int) -> list[int]:
        docs: list[int] = []
        for tok in TOKEN_RE.findall(query):
            if len(tok) < 3:
                continue
            up = tok.upper()
            rows = self.db.execute(
                """SELECT d.id FROM retrieval_documents d JOIN entities e ON d.subject_kind='entity' AND d.subject_id=e.id
                   WHERE e.name_upper=? ORDER BY d.doc_type='entity' DESC""", (up,)).fetchall()
            rows += self.db.execute(
                """SELECT d.id FROM retrieval_documents d JOIN queries q ON d.subject_kind='query' AND d.subject_id=q.id
                   WHERE q.statement_id=? OR q.symbol=? OR q.namespace=?""", (tok, tok, tok)).fetchall()
            rows += self.db.execute(
                """SELECT d.id FROM retrieval_documents d JOIN symbols s ON d.subject_kind='symbol' AND d.subject_id=s.id
                   WHERE s.name=? OR s.fqn=?""", (tok, tok)).fetchall()
            if "_" in tok or tok.isupper():
                rows += self.db.execute(
                    """SELECT DISTINCT d.id FROM columns c JOIN retrieval_documents d ON d.subject_kind='entity' AND d.subject_id=c.entity_id
                       WHERE c.name_upper=? AND d.doc_type='entity' LIMIT 20""", (up,)).fetchall()
            for (did,) in rows:
                if did not in docs:
                    docs.append(did)
        return docs[:limit]

    def _fts(self, query: str, limit: int) -> list[int]:
        fq = _fts_query(query)
        if not fq:
            return []
        rows = self.db.execute(
            """SELECT c.document_id FROM chunks_fts f JOIN chunks c ON c.id=f.rowid
               WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts, 1.0, 3.0) LIMIT ?""", (fq, limit * 3)).fetchall()
        return list(dict.fromkeys(r[0] for r in rows))[:limit]

    def _dense(self, query: str, limit: int) -> list[int]:
        if self.index is None or self.embedder is None:
            return []
        vec = self.embedder.embed([query])
        _, ids = self.index.search(np.asarray(vec, dtype="float32"), limit * 3)
        chunk_ids = [int(i) for i in ids[0] if i >= 0]
        if not chunk_ids:
            return []
        marks = ",".join("?" * len(chunk_ids))
        doc_of = dict(self.db.execute(f"SELECT id, document_id FROM chunks WHERE id IN ({marks})", chunk_ids).fetchall())
        return list(dict.fromkeys(doc_of[c] for c in chunk_ids if c in doc_of))[:limit]

    def search(self, query: str, k: int = 10, storage_kind: str | None = None, component: str | None = None,
               doc_types: list[str] | None = None, include_tests: bool = False) -> dict[str, Any]:
        pool = max(50, k * 5)
        channels = {"exact": self._exact(query, pool), "fts": self._fts(query, pool), "dense": self._dense(query, pool)}
        scores: dict[int, float] = {}
        ranks: dict[int, dict[str, int]] = {}
        for name, docs in channels.items():
            for r, did in enumerate(docs, 1):
                scores[did] = scores.get(did, 0.0) + 1.0 / (RRF_K + r) * (1.5 if name == "exact" else 1.0)
                ranks.setdefault(did, {})[name] = r
        hits = []
        for did in sorted(scores, key=scores.get, reverse=True):
            d = self.db.execute("SELECT * FROM retrieval_documents WHERE id=?", (did,)).fetchone()
            if d is None:
                continue
            if not include_tests and d["is_test"]:
                continue
            if storage_kind and d["storage_kind"] != storage_kind:
                continue
            if component and not (d["component"] or "").startswith(component):
                continue
            if doc_types and d["doc_type"] not in doc_types:
                continue
            hits.append({
                "score": round(scores[did], 5), "channels": ranks[did], "doc_type": d["doc_type"], "title": d["title"],
                "subject": {"kind": d["subject_kind"], "id": d["subject_id"]}, "storage_kind": d["storage_kind"],
                "component": d["component"], "is_test": bool(d["is_test"]),
                "provenance": {"path": d["path"], "line_start": d["line_start"], "line_end": d["line_end"]},
                "snippet": d["text"][:600],
            })
            if len(hits) >= k:
                break
        return {"query": query, "channels_used": [n for n in channels if n != "dense" or self.index is not None], "hits": hits}
