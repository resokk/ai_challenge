"""Elasticsearch / OpenSearch index and index-template mappings (JSON).

Each mapping file becomes a non-relational entity with its own storage_kind; fields are the
flattened property paths with their ES/OS types. They are never validated with SQLGlot.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

from camunda_rag.models import ColumnDef, Diagnostic, EntityDef, FactStatus, Provenance, StorageKind

EXTRACTOR = "search-mapping"


def storage_kind_of(rel: str, doc: dict[str, Any]) -> StorageKind | None:
    low = rel.lower()
    engine = "opensearch" if "opensearch" in low else "elasticsearch" if "elasticsearch" in low else None
    if engine is None:
        return None
    template = "/template" in low or "template" in PurePosixPath(low).name or "index_patterns" in doc
    return StorageKind(f"{engine}_index_template" if template else f"{engine}_index")


def _line_of(text: str, key: str) -> int | None:
    idx = text.find(f'"{key}"')
    return text.count("\n", 0, idx) + 1 if idx >= 0 else None


def _flatten(props: dict[str, Any], prefix: str, out: list[tuple[str, dict[str, Any]]]) -> None:
    for name, spec in props.items():
        path = f"{prefix}{name}"
        if not isinstance(spec, dict):
            continue
        out.append((path, spec))
        if isinstance(spec.get("properties"), dict):
            _flatten(spec["properties"], path + ".", out)
        for sub, sspec in (spec.get("fields") or {}).items():
            if isinstance(sspec, dict):
                out.append((f"{path}.{sub}", {**sspec, "_multi_field": True}))


def extract(repo: Path, rel: str, is_test: bool) -> tuple[EntityDef | None, list[Diagnostic]]:
    text = (repo / rel).read_text("utf-8")
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as e:
        return None, [Diagnostic(path=rel, extractor=EXTRACTOR, severity="error", code="json_error", message=str(e))]
    kind = storage_kind_of(rel, doc)
    if kind is None:
        return None, [Diagnostic(path=rel, extractor=EXTRACTOR, severity="info", code="unknown_engine",
                                 message="mapping-like JSON outside an elasticsearch/opensearch path; engine not determinable")]
    tmpl = doc.get("template") if isinstance(doc.get("template"), dict) else {}
    mappings = doc.get("mappings") or tmpl.get("mappings") or {}
    if not isinstance(mappings, dict) or not isinstance(mappings.get("properties"), dict):
        return None, [Diagnostic(path=rel, extractor=EXTRACTOR, severity="info", code="no_properties",
                                 message="JSON has no mappings.properties")]
    fields: list[tuple[str, dict[str, Any]]] = []
    _flatten(mappings["properties"], "", fields)
    component = rel.split("/src/", 1)[0]
    ent = EntityDef(
        storage_kind=kind, component=component, name=PurePosixPath(rel).stem, is_test=is_test,
        provenance=Provenance(path=rel, line_start=1, line_end=text.count("\n") + 1, symbol=PurePosixPath(rel).stem,
                              extractor=EXTRACTOR, evidence_kind="search_mapping_json"),
        attributes={"dynamic": mappings.get("dynamic"), "index_patterns": doc.get("index_patterns"),
                    "composed_of": doc.get("composed_of"), "settings": doc.get("settings") or tmpl.get("settings"),
                    "completeness": "complete"},
    )
    for i, (path, spec) in enumerate(fields):
        line = _line_of(text, path.rsplit(".", 1)[-1])
        dtype = spec.get("type") or ("object" if "properties" in spec else None)
        extras = {k: v for k, v in spec.items() if k not in {"type", "properties", "fields", "_multi_field"}}
        ent.columns.append(ColumnDef(
            name=path, data_type=dtype, type_source="es_mapping", position=i, status=FactStatus.declared,
            remarks=("multi-field; " if spec.get("_multi_field") else "") + (json.dumps(extras) if extras else "") or None,
            provenance=Provenance(path=rel, line_start=line, line_end=line, symbol=f"{ent.name}.{path}", extractor=EXTRACTOR,
                                  evidence_kind="search_mapping_field"),
        ))
    return ent, []
