"""Runs the JVM analyzer (JavaParser + Symbol Solver) and converts its versioned JSONL.

Holes (unresolved expressions) stay holes: a variant with holes is `partial`. The only
substitution is a hole that is literally the `prefix` parameter/variable: it is replaced by the
profile table prefix and recorded as an *assumed* parameter.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterator

from camunda_rag.config import PROJECT_DIR, Settings
from camunda_rag.models import Diagnostic, Provenance, QueryDef, QueryParam, QueryVariant, SymbolDef, FactStatus

EXTRACTOR = "java-analyzer"
SUPPORTED_SCHEMA_VERSIONS = {1}
DIALECT_HINTS = [
    (re.compile(r"postgres", re.I), "postgresql"), (re.compile(r"oracle", re.I), "oracle"),
    (re.compile(r"mssql|sqlserver", re.I), "mssql"), (re.compile(r"mariadb", re.I), "mariadb"),
    (re.compile(r"mysql", re.I), "mysql"), (re.compile(r"(^|[^a-z])h2", re.I), "h2"),
]


def ensure_jar(settings: Settings, rebuild: bool = False) -> Path:
    jar = settings.java_analyzer_jar
    if jar.exists() and not rebuild:
        return jar
    mvn = shutil.which(settings.maven_cmd)
    if mvn is None:
        raise RuntimeError(f"Maven ({settings.maven_cmd}) is required to build the Java analyzer; set maven_cmd in config.toml")
    subprocess.run([mvn, "-q", "-f", str(PROJECT_DIR / "java-analyzer" / "pom.xml"), "package", "-DskipTests"], check=True)
    return jar


def source_roots(java_files: list[str]) -> list[str]:
    roots = set()
    for f in java_files:
        m = re.match(r"(.*?/src/(?:main|test)/java)/", f)
        if m:
            roots.add(m.group(1))
    return sorted(roots)


def run_analyzer(settings: Settings, repo: Path, files: list[str], all_java: list[str], namespaces: list[str],
                 work: Path) -> Path:
    jar = ensure_jar(settings)
    work.mkdir(parents=True, exist_ok=True)
    (work / "files.txt").write_text("\n".join(files))
    (work / "namespaces.txt").write_text("\n".join(namespaces))
    (work / "roots.txt").write_text("\n".join(source_roots(all_java)))
    out = work / "java-analysis.jsonl"
    lim = settings.limits
    cmd = [settings.java_cmd, "-Xss16m", "-Xmx6g", "-jar", str(jar), "--repo", str(repo), "--files", str(work / "files.txt"),
           "--namespaces", str(work / "namespaces.txt"), "--source-roots", str(work / "roots.txt"), "--out", str(out),
           "--max-depth", str(lim.java_max_depth), "--max-variants", str(lim.java_max_variants), "--max-loop", str(lim.java_max_loop_unroll)]
    subprocess.run(cmd, check=True)
    return out


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        first = json.loads(fh.readline())
        if first.get("type") != "header" or first.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported analyzer output header: {first}")
        yield first
        for line in fh:
            if line.strip():
                yield json.loads(line)


def dialect_from_evidence(path: str, cls: str | None) -> tuple[str | None, str]:
    names = ([("class", cls.rsplit(".", 1)[-1])] if cls else []) + [("file", path.rsplit("/", 1)[-1])]
    for where, name in names:
        for rx, d in DIALECT_HINTS:
            if rx.search(name):
                return d, f"name hint in {where} {name}"
    return None, "no dialect evidence in Java source; analysed with the default dialect as an assumption"


def convert(records: Iterator[dict[str, Any]], is_test_path, table_prefix: str) -> tuple[
        list[QueryDef], list[SymbolDef], dict[str, dict[str, Any]], list[Diagnostic], dict[str, Any]]:
    queries: list[QueryDef] = []
    symbols: list[SymbolDef] = []
    files: dict[str, dict[str, Any]] = {}
    diags: list[Diagnostic] = []
    header: dict[str, Any] = {}
    for r in records:
        t = r["type"]
        if t == "header":
            header = r
        elif t == "file":
            files[r["path"]] = r
            if r.get("status") == "parse_error":
                diags.append(Diagnostic(path=r["path"], extractor=EXTRACTOR, severity="error", code="java_parse_error",
                                        message=r.get("message", "")[:1000]))
            for d in r.get("diagnostics", []):
                diags.append(Diagnostic(path=r["path"], extractor=EXTRACTOR, code="java", message=str(d)))
        elif t == "query":
            queries.append(_query(r, is_test_path(r["path"]), table_prefix))
            for d in r.get("diagnostics", []):
                diags.append(Diagnostic(path=r["path"], line=r.get("line_start"), extractor=EXTRACTOR, code=str(d).split(" ")[0],
                                        message=str(d)[:1000], subject=queries[-1].key))
        elif t == "type":
            symbols.extend(_symbols(r, is_test_path(r["path"])))
    return queries, symbols, files, diags, header


def _query(r: dict[str, Any], is_test: bool, table_prefix: str) -> QueryDef:
    dialect, evidence = dialect_from_evidence(r["path"], r.get("class"))
    variants = []
    for v in r.get("variants", []):
        parts, unresolved, params = [], [], []
        for s in v["segments"]:
            if "text" in s:
                parts.append(s["text"])
            elif s.get("hole") == "prefix":
                parts.append(table_prefix)
                params.append(QueryParam(name="prefix", kind="java_hole", resolved_value=table_prefix))
            else:
                parts.append(f"⟪{s['hole']}⟫")
                unresolved.append(f"{s['hole']}: {s.get('reason')}")
                params.append(QueryParam(name=s["hole"], kind="java_hole"))
        variants.append(QueryVariant(dialect=dialect, dialect_evidence=evidence, conditions=v.get("conditions", []),
                                     sql="".join(parts), partial=bool(unresolved), unresolved=unresolved, parameters=params))
    symbol = r.get("class", "") + ("#" + r["method"] if r.get("method") else ("." + r["field"] if r.get("field") else ""))
    lang = r.get("language_hint", "unknown")
    return QueryDef(
        key=f"java:{r['path']}:{r.get('line_start')}:{r['kind']}", language=lang, origin="java", statement_kind=r["kind"],
        raw_text=r.get("expression", ""), description="\n".join(x for x in [r.get("javadoc"), r.get("comment")] if x) or None,
        is_test=is_test, variants=variants,
        template={"type": "java_expression", "kind": r["kind"], "sink": r.get("sink"), "expression": r.get("expression")},
        provenance=Provenance(path=r["path"], line_start=r.get("line_start"), line_end=r.get("line_end"), symbol=symbol,
                              extractor=EXTRACTOR, evidence_kind=f"java_{r['kind']}",
                              confidence=1.0 if not any(v.partial for v in variants) else 0.6),
        diagnostics=[str(d) for d in r.get("diagnostics", [])],
    )


def _symbols(r: dict[str, Any], is_test: bool) -> list[SymbolDef]:
    prov = Provenance(path=r["path"], line_start=r.get("line_start"), line_end=r.get("line_end"), symbol=r["fqn"],
                      extractor=EXTRACTOR, evidence_kind="java_type")
    out = [SymbolDef(kind=r["kind"], fqn=r["fqn"], name=r["name"], javadoc=r.get("javadoc"), comment=r.get("comment"),
                     attributes={"fields": r.get("fields", []), "enum_constants": r.get("enum_constants", []),
                                 "mapper_interface": r.get("mapper_interface", False), "annotations": r.get("annotations", [])},
                     provenance=prov, is_test=is_test)]
    for m in r.get("methods", []):
        out.append(SymbolDef(
            kind="method", fqn=f"{r['fqn']}#{m['name']}", name=m["name"], javadoc=m.get("javadoc"), signature=m.get("signature"),
            attributes={"params": m.get("params", []), "return_type": m.get("return_type"), "annotations": m.get("annotations", []),
                        "parent": r["fqn"]},
            provenance=Provenance(path=r["path"], line_start=m.get("line_start"), line_end=m.get("line_end"),
                                  symbol=f"{r['fqn']}#{m['name']}", extractor=EXTRACTOR, evidence_kind="java_mapper_method"),
            is_test=is_test))
    return out


def rocksdb_entities(symbols: list[SymbolDef]):
    """ZbColumnFamilies-like enums -> rocksdb_column_family entities (declared names only)."""
    from camunda_rag.models import EntityDef, StorageKind

    out = []
    for s in symbols:
        if s.kind != "enum" or not s.name.endswith("ColumnFamilies"):
            continue
        component = s.provenance.path.split("/src/", 1)[0]
        for c in s.attributes.get("enum_constants", []):
            args = c.get("arguments", [])
            out.append(EntityDef(
                storage_kind=StorageKind.rocksdb_column_family, component=component, name=c["name"], is_test=s.is_test,
                remarks=c.get("comment"),
                attributes={"enum": s.fqn, "arguments": args, "deprecated": c.get("deprecated", False),
                            "completeness": "partial",
                            "note": "key/value types are defined at usage sites and are not extracted"},
                provenance=Provenance(path=s.provenance.path, line_start=c.get("line"), line_end=c.get("line"),
                                      symbol=f"{s.fqn}.{c['name']}", extractor=EXTRACTOR, evidence_kind="java_enum_constant",
                                      status=FactStatus.declared),
            ))
    return out
