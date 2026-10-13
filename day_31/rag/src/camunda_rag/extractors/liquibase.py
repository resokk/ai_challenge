"""Liquibase changelog extractor (XML / YAML / JSON).

XML is converted into the same nested structure that the YAML/JSON changelog formats use, so a
single interpreter walks all three. include/includeAll are resolved in Liquibase order
(includeAll: alphabetical), producing ChangeRecords with a global order per root changelog.
The extractor does not decide what applies; schema reconstruction evaluates dbms, contexts,
labels and preconditions for a concrete profile.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from lxml import etree

from camunda_rag.models import ChangelogProperty, ChangeRecord, Diagnostic, Provenance

EXTRACTOR = "liquibase"
NON_CHANGE_TAGS = {"preConditions", "comment", "rollback", "validCheckSum", "tagDatabase"}


def _local(tag: Any) -> str:
    return etree.QName(tag).localname if isinstance(tag, str) else ""


def _el_to_dict(el: etree._Element) -> dict[str, Any]:
    d: dict[str, Any] = dict(el.attrib)
    d["_line"] = el.sourceline
    d["_line_end"] = max((x.sourceline or 0 for x in el.iter() if isinstance(x.tag, str)), default=el.sourceline)
    for child in el:
        tag = _local(child.tag)
        if not tag:
            continue
        cd = _el_to_dict(child)
        if tag == "column":
            d.setdefault("columns", []).append({"column": cd})
        elif tag == "constraints":
            d["constraints"] = cd
        else:
            d.setdefault("_children", []).append({tag: cd})
    text = (el.text or "").strip()
    if text:
        d["_text"] = text
    return d


def parse_xml(path: Path) -> tuple[list[dict[str, Any]], dict[int, str]]:
    """Returns top-level entries ({tag: dict}) and XML comments keyed by the line of the next element."""
    root = etree.parse(str(path), etree.XMLParser(remove_blank_text=False, resolve_entities=False, no_network=True)).getroot()
    entries: list[dict[str, Any]] = []
    comments: dict[int, str] = {}
    pending: list[str] = []
    for child in root:
        if isinstance(child, etree._Comment):
            pending.append((child.text or "").strip())
            continue
        tag = _local(child.tag)
        if not tag:
            continue
        entries.append({tag: _el_to_dict(child)})
        if pending:
            comments[child.sourceline] = "\n".join(pending)
            pending = []
    return entries, comments


def _normalize_structured(obj: Any) -> list[dict[str, Any]]:
    """YAML/JSON: {'databaseChangeLog': [ {changeSet: {..., changes: [...]}} ]} -> entries like XML."""
    items = obj.get("databaseChangeLog", []) if isinstance(obj, dict) else []
    out = []
    for item in items or []:
        if not isinstance(item, dict) or len(item) != 1:
            continue
        tag, body = next(iter(item.items()))
        body = dict(body or {})
        if tag == "changeSet":
            children = []
            pre = body.pop("preConditions", None)
            if pre is not None:
                children.append({"preConditions": _yaml_preconditions(pre)})
            for ch in body.pop("changes", []) or []:
                if isinstance(ch, dict) and len(ch) == 1:
                    ctag, cbody = next(iter(ch.items()))
                    children.append({ctag: dict(cbody or {})})
            if "comment" in body:
                children.append({"comment": {"_text": body.pop("comment")}})
            body["_children"] = children
        out.append({tag: body})
    return out


def _yaml_preconditions(pre: Any) -> dict[str, Any]:
    # YAML: list of single-key dicts, where onFail/onError appear as items
    d: dict[str, Any] = {"_children": []}
    for item in pre if isinstance(pre, list) else [pre]:
        if not isinstance(item, dict):
            continue
        for k, v in item.items():
            if k in {"onFail", "onError", "onFailMessage", "onErrorMessage"}:
                d[k] = v
            elif k in {"not", "and", "or"}:
                d["_children"].append({k: _yaml_preconditions(v)})
            else:
                d["_children"].append({k: dict(v or {})})
    return d


def _condition_tree(d: dict[str, Any]) -> dict[str, Any]:
    children = []
    for c in d.get("_children", []):
        tag, body = next(iter(c.items()))
        if tag in {"not", "and", "or"}:
            children.append({"op": tag, "children": _condition_tree(body)["children"]})
        else:
            attrs = {k: v for k, v in body.items() if not k.startswith("_")}
            if "_text" in body:
                attrs["sql"] = body["_text"]
            children.append({"op": tag, "attrs": attrs})
    return {"op": "and", "children": children}


def _split_list(v: Any) -> list[str]:
    if not v:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v]
    return [x.strip() for x in str(v).split(",") if x.strip()]


@dataclass
class ChangelogResult:
    root: str
    component: str
    is_test: bool
    changes: list[ChangeRecord] = field(default_factory=list)
    properties: list[ChangelogProperty] = field(default_factory=list)
    files: list[str] = field(default_factory=list)  # every file reached through includes
    diagnostics: list[Diagnostic] = field(default_factory=list)


def resource_roots(rel: str) -> list[str]:
    """Classpath roots the file can see: <module>/src/{main,test}/resources."""
    p = PurePosixPath(rel)
    parts = p.parts
    for i in range(len(parts) - 2):
        if parts[i] == "src" and parts[i + 2] == "resources":
            module = PurePosixPath(*parts[:i]) if i else PurePosixPath(".")
            own = str(PurePosixPath(*parts[: i + 3]))
            others = [str(module / "src" / s / "resources") for s in ("main", "test")]
            return [own] + [o for o in others if o != own]
    return []


def component_of(rel: str) -> str:
    """<module>:<root changelog path inside its resource root, without extension>; unique per root."""
    module = rel.split("/src/", 1)[0] if "/src/" in rel else str(PurePosixPath(rel).parent)
    inner = rel.split("/resources/", 1)[1] if "/resources/" in rel else PurePosixPath(rel).name
    return f"{module}:{inner.rsplit('.', 1)[0]}"


class LiquibaseExtractor:
    def __init__(self, repo: Path, known_files: set[str], is_test_path) -> None:
        self.repo = repo
        self.known = known_files
        self.is_test_path = is_test_path
        self._cache: dict[str, tuple[list[dict[str, Any]], dict[int, str]]] = {}

    def load(self, rel: str) -> tuple[list[dict[str, Any]], dict[int, str]]:
        if rel not in self._cache:
            path = self.repo / rel
            if rel.endswith(".xml"):
                self._cache[rel] = parse_xml(path)
            else:
                text = path.read_text("utf-8")
                obj = json.loads(text) if rel.endswith(".json") else yaml.safe_load(text)
                self._cache[rel] = (_normalize_structured(obj), {})
        return self._cache[rel]

    def includes_of(self, rel: str) -> list[str]:
        out: list[str] = []
        try:
            entries, _ = self.load(rel)
        except Exception:  # noqa: BLE001 - reported when the file is walked as a root
            return out
        for e in entries:
            tag, body = next(iter(e.items()))
            if tag == "include":
                r = self._resolve(rel, body.get("file", ""), str(body.get("relativeToChangelogFile", "false")).lower() == "true")
                if r:
                    out.append(r)
            elif tag == "includeAll":
                out.extend(self._resolve_dir(rel, body))
        return out

    def _resolve(self, current: str, target: str, relative: bool) -> str | None:
        target = target.strip()
        if relative:
            cand = str(PurePosixPath(current).parent / target)
            return cand if cand in self.known else None
        for root in resource_roots(current):
            cand = str(PurePosixPath(root) / target.lstrip("/"))
            if cand in self.known:
                return cand
        return target.lstrip("/") if target.lstrip("/") in self.known else None

    def _resolve_dir(self, current: str, body: dict[str, Any]) -> list[str]:
        path = body.get("path", "").strip().rstrip("/")
        relative = str(body.get("relativeToChangelogFile", "false")).lower() == "true"
        bases = [str(PurePosixPath(current).parent / path)] if relative else [
            str(PurePosixPath(r) / path.lstrip("/")) for r in resource_roots(current)
        ]
        for base in bases:
            files = sorted(f for f in self.known if f.startswith(base + "/") and f.endswith((".xml", ".yaml", ".yml", ".json")))
            if files:
                return files
        return []

    def roots(self, changelog_files: list[str]) -> list[str]:
        included = {inc for f in changelog_files for inc in self.includes_of(f)}
        return [f for f in changelog_files if f not in included]

    def extract_root(self, root: str) -> ChangelogResult:
        res = ChangelogResult(root=root, component=component_of(root), is_test=self.is_test_path(root))
        counter = [0]
        self._walk(root, res, counter, stack=[])
        return res

    def _walk(self, rel: str, res: ChangelogResult, counter: list[int], stack: list[str]) -> None:
        if rel in stack:
            res.diagnostics.append(Diagnostic(path=rel, extractor=EXTRACTOR, code="include_cycle", message=" -> ".join(stack + [rel])))
            return
        res.files.append(rel)
        try:
            entries, comments = self.load(rel)
        except Exception as e:  # noqa: BLE001
            res.diagnostics.append(Diagnostic(path=rel, extractor=EXTRACTOR, severity="error", code="parse_error", message=str(e)))
            return
        for e in entries:
            tag, body = next(iter(e.items()))
            if tag == "include":
                target = self._resolve(rel, body.get("file", ""), str(body.get("relativeToChangelogFile", "false")).lower() == "true")
                if target is None:
                    res.diagnostics.append(Diagnostic(path=rel, line=body.get("_line"), extractor=EXTRACTOR, code="unresolved_include",
                                                      message=f"include {body.get('file')!r} not found in snapshot"))
                else:
                    self._walk(target, res, counter, stack + [rel])
            elif tag == "includeAll":
                targets = self._resolve_dir(rel, body)
                if not targets:
                    res.diagnostics.append(Diagnostic(path=rel, line=body.get("_line"), extractor=EXTRACTOR, code="unresolved_include_all",
                                                      message=f"includeAll {body.get('path')!r} matched no files"))
                for t in targets:
                    self._walk(t, res, counter, stack + [rel])
            elif tag == "property":
                if "name" in body and "value" in body:
                    res.properties.append(ChangelogProperty(name=body["name"], value=str(body["value"]), dbms=_split_list(body.get("dbms")),
                                                            context=body.get("context"), path=rel))
            elif tag == "changeSet":
                self._changeset(rel, body, comments.get(body.get("_line", -1)), res, counter)
            else:
                res.diagnostics.append(Diagnostic(path=rel, line=body.get("_line"), extractor=EXTRACTOR, severity="info",
                                                  code="unsupported_tag", message=f"top-level <{tag}> ignored"))

    def _changeset(self, rel: str, cs: dict[str, Any], xml_comment: str | None, res: ChangelogResult, counter: list[int]) -> None:
        cs_id = str(cs.get("id"))
        pre, on_fail, comment = None, None, xml_comment
        changes: list[tuple[str, dict[str, Any]]] = []
        for c in cs.get("_children", []):
            tag, body = next(iter(c.items()))
            if tag == "preConditions":
                pre = _condition_tree(body)
                on_fail = body.get("onFail", "HALT")
            elif tag == "comment":
                comment = "\n".join(x for x in [comment, body.get("_text")] if x)
            elif tag not in NON_CHANGE_TAGS:
                changes.append((tag, body))
        if not changes:
            res.diagnostics.append(Diagnostic(path=rel, line=cs.get("_line"), extractor=EXTRACTOR, severity="info",
                                              code="empty_changeset", message=f"changeSet {cs_id} has no changes"))
        for tag, body in changes:
            counter[0] += 1
            details = {k: v for k, v in body.items() if not k.startswith("_") or k in {"_children"}}
            if "_text" in body:
                details["sql" if tag in {"sql", "createProcedure", "createView"} else "text"] = body["_text"]
            res.changes.append(ChangeRecord(
                order=counter[0], component=res.component, changelog_path=rel, changeset_id=cs_id,
                author=cs.get("author"), change_type=tag, table=body.get("tableName") or body.get("oldTableName"),
                details=details, dbms=_split_list(body.get("dbms") or cs.get("dbms")),
                contexts=cs.get("contextFilter") or cs.get("context") or cs.get("contexts"), labels=cs.get("labels"),
                preconditions=pre, on_fail=on_fail, comment=comment,
                provenance=Provenance(path=rel, line_start=body.get("_line") or cs.get("_line"),
                                      line_end=body.get("_line_end") or cs.get("_line_end"),
                                      symbol=f"changeSet:{cs_id}", extractor=EXTRACTOR, evidence_kind=f"liquibase_{tag}"),
            ))
