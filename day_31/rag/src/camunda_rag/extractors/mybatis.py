"""MyBatis mapper XML extractor.

1. Structural parse of every mapper into a template tree (text, placeholder, if, choose/when/
   otherwise, foreach, trim/where/set, include, bind) — stored as is.
2. Bounded rendering per dialect: statements and <sql> fragments are selected with MyBatis
   databaseId rules; includes resolve across files; `${}` is resolved only from include
   properties and configuration variables (vendor properties, table prefix). Any other `${}`
   is a runtime substitution: the variant is `partial` and is never handed to SQLGlot with
   an invented identifier.
3. Variant plan (no blind cartesian product): all-if-true/first-when, all-if-false/otherwise,
   then one variant per alternative branch of each <choose>, capped by `max_variants`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lxml import etree

from camunda_rag.extractors.vendor_properties import VendorProperties
from camunda_rag.models import Diagnostic, Provenance, QueryDef, QueryParam, QueryVariant, ResultMapping

EXTRACTOR = "mybatis"
STATEMENT_TAGS = {"select", "insert", "update", "delete"}
DYNAMIC_TAGS = {"if", "choose", "foreach", "trim", "where", "set", "include", "bind"}
PLACEHOLDER_RE = re.compile(r"([#$])\{([^}]*)\}")
EQ_ATOM = re.compile(r"^\s*([\w.()]+?)\s*(?:==\s*'([^']*)'|\.equals\('([^']*)'\))\s*$")


def _eq_atoms(test: str) -> tuple[str, frozenset[str]] | None:
    """`x == 'A' or x == 'B'` / `x.equals('A')` -> (x, {A, B}); anything else -> None."""
    expr, consts = None, set()
    for atom in re.split(r"\s+or\s+|\s*\|\|\s*", test.strip()):
        m = EQ_ATOM.match(atom)
        if not m or (expr is not None and m.group(1) != expr):
            return None
        expr = m.group(1)
        consts.add(m.group(2) if m.group(2) is not None else m.group(3))
    return (expr, frozenset(consts)) if expr else None


def group_exclusive_ifs(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sibling <if>s testing one expression for equality with disjoint constants are mutually
    exclusive; they are rendered like a <choose> (one member or none), never all together."""
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(nodes):
        n = nodes[i]
        atoms = _eq_atoms(n.get("test") or "") if n["type"] == "if" else None
        if atoms is None:
            out.append(n)
            i += 1
            continue
        members, seen, j, last = [n], set(atoms[1]), i + 1, i
        while j < len(nodes):
            m = nodes[j]
            if m["type"] == "text" and not m["text"].strip():
                j += 1
                continue
            a = _eq_atoms(m.get("test") or "") if m["type"] == "if" else None
            if a is None or a[0] != atoms[0] or a[1] & seen:
                break
            members.append(m)
            seen |= a[1]
            last = j
            j += 1
        if len(members) > 1:
            out.append({"type": "_exclusive_ifs", "line": n["line"], "expr": atoms[0], "members": members})
            i = last + 1
        else:
            out.append(n)
            i += 1
    return out


# ------------------------------------------------------------------ template tree


def _text_nodes(text: str | None) -> list[dict[str, Any]]:
    if not text:
        return []
    out: list[dict[str, Any]] = []
    pos = 0
    for m in PLACEHOLDER_RE.finditer(text):
        if m.start() > pos:
            out.append({"type": "text", "text": text[pos:m.start()]})
        out.append(_placeholder(m.group(1), m.group(2)))
        pos = m.end()
    if pos < len(text):
        out.append({"type": "text", "text": text[pos:]})
    return out


def _placeholder(syntax: str, body: str) -> dict[str, Any]:
    parts = [p.strip() for p in body.split(",")]
    node: dict[str, Any] = {"type": "placeholder", "syntax": syntax, "expr": parts[0]}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            node[k.strip()] = v.strip()
    return node


def build_tree(el: etree._Element) -> list[dict[str, Any]]:
    nodes = _text_nodes(el.text)
    for child in el:
        if isinstance(child, etree._Comment) or not isinstance(child.tag, str):
            nodes.extend(_text_nodes(child.tail))
            continue
        tag = etree.QName(child.tag).localname
        a = dict(child.attrib)
        line = child.sourceline
        if tag == "if":
            nodes.append({"type": "if", "test": a.get("test", ""), "line": line, "children": build_tree(child)})
        elif tag == "choose":
            branches = []
            for b in child:
                if not isinstance(b.tag, str):
                    continue
                btag = etree.QName(b.tag).localname
                if btag in {"when", "otherwise"}:
                    branches.append({"type": btag, "test": b.get("test"), "line": b.sourceline, "children": build_tree(b)})
            nodes.append({"type": "choose", "line": line, "branches": branches})
        elif tag == "foreach":
            nodes.append({"type": "foreach", "line": line, **{k: a.get(k) for k in ("collection", "item", "index", "open", "close", "separator")},
                          "children": build_tree(child)})
        elif tag in {"trim", "where", "set"}:
            nodes.append({"type": tag, "line": line, **{k: a.get(k) for k in ("prefix", "suffix", "prefixOverrides", "suffixOverrides")},
                          "children": build_tree(child)})
        elif tag == "include":
            props = {p.get("name"): p.get("value") for p in child if isinstance(p.tag, str) and etree.QName(p.tag).localname == "property"}
            nodes.append({"type": "include", "line": line, "refid": a.get("refid", ""), "properties": props})
        elif tag == "bind":
            nodes.append({"type": "bind", "line": line, "name": a.get("name"), "value": a.get("value")})
        elif tag == "selectKey":
            nodes.append({"type": "selectKey", "line": line, "keyProperty": a.get("keyProperty"), "children": build_tree(child)})
        else:
            nodes.append({"type": "unknown", "tag": tag, "line": line, "children": build_tree(child)})
        nodes.extend(_text_nodes(child.tail))
    return nodes


# ------------------------------------------------------------------ registry


@dataclass
class Element:
    namespace: str
    id: str
    tag: str
    database_id: str | None
    attrs: dict[str, str]
    tree: list[dict[str, Any]]
    raw: str
    path: str
    line_start: int
    line_end: int
    comment: str | None
    is_test: bool


@dataclass
class Registry:
    statements: list[Element] = field(default_factory=list)
    fragments: dict[tuple[str, str], list[Element]] = field(default_factory=dict)
    result_maps: list[ResultMapping] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    namespaces: dict[str, str] = field(default_factory=dict)  # namespace -> path


def parse_mapper(repo: Path, rel: str, is_test: bool, reg: Registry) -> None:
    try:
        tree = etree.parse(str(repo / rel), etree.XMLParser(load_dtd=False, no_network=True, resolve_entities=False))
    except etree.XMLSyntaxError as e:
        reg.diagnostics.append(Diagnostic(path=rel, extractor=EXTRACTOR, severity="error", code="parse_error", message=str(e)))
        return
    root = tree.getroot()
    ns = root.get("namespace", "")
    reg.namespaces[ns] = rel
    pending_comment: list[str] = []
    for el in root:
        if isinstance(el, etree._Comment):
            pending_comment.append((el.text or "").strip())
            continue
        if not isinstance(el.tag, str):
            continue
        tag = etree.QName(el.tag).localname
        line_end = max((x.sourceline or 0 for x in el.iter() if isinstance(x.tag, str)), default=el.sourceline)
        line_end = max(line_end, el.sourceline + (etree.tostring(el, encoding="unicode", with_tail=False).count("\n")))
        comment = "\n".join(pending_comment) or None
        pending_comment = []
        if tag in STATEMENT_TAGS or tag == "sql":
            e = Element(namespace=ns, id=el.get("id", ""), tag=tag, database_id=el.get("databaseId"), attrs=dict(el.attrib),
                        tree=build_tree(el), raw=etree.tostring(el, encoding="unicode", with_tail=False), path=rel,
                        line_start=el.sourceline, line_end=line_end, comment=comment, is_test=is_test)
            if tag == "sql":
                reg.fragments.setdefault((ns, e.id), []).append(e)
            else:
                reg.statements.append(e)
        elif tag == "resultMap":
            _result_map(el, ns, rel, reg, prefix_prop="", prefix_col="", rm_id=f"{ns}.{el.get('id')}", java_type=el.get("type"))
        else:
            reg.diagnostics.append(Diagnostic(path=rel, line=el.sourceline, extractor=EXTRACTOR, severity="info",
                                              code="unsupported_element", message=f"<{tag}> not modelled"))


def _result_map(el: etree._Element, ns: str, rel: str, reg: Registry, prefix_prop: str, prefix_col: str, rm_id: str, java_type: str | None) -> None:
    for c in el:
        if not isinstance(c.tag, str):
            continue
        tag = etree.QName(c.tag).localname
        if tag in {"association", "collection"}:
            _result_map(c, ns, rel, reg, f"{prefix_prop}{c.get('property', '')}.", prefix_col + (c.get("columnPrefix") or ""), rm_id, java_type)
        elif tag == "constructor":
            _result_map(c, ns, rel, reg, prefix_prop, prefix_col, rm_id, java_type)
        elif tag in {"id", "result", "idArg", "arg"} and c.get("column"):
            prop = c.get("property") or c.get("name") or f"<constructor arg {c.get('javaType', '')}>"
            reg.result_maps.append(ResultMapping(
                result_map=rm_id, java_type=java_type, column=prefix_col + c.get("column"), property=prefix_prop + prop,
                provenance=Provenance(path=rel, line_start=c.sourceline, line_end=c.sourceline, symbol=rm_id,
                                      extractor=EXTRACTOR, evidence_kind="mybatis_result_map"),
            ))


# ------------------------------------------------------------------ rendering


class _Unresolved(Exception):
    pass


@dataclass
class RenderCtx:
    dialect: str
    config_vars: dict[str, str]
    config_sources: dict[str, str]
    decisions: dict[str, Any]  # "choose@<path>:<line>" / "ifgroup@..." -> branch index
    if_default: bool | str  # True | False | "safe" (true unless the branch needs a runtime ${} substitution)
    foreach_iterations: int
    max_include_depth: int
    namespace: str
    path: str
    conditions: list[str] = field(default_factory=list)
    params: list[QueryParam] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    choose_sites: list[tuple[str, int]] = field(default_factory=list)  # (site key, n branches)
    diagnostics: list[str] = field(default_factory=list)


class MyBatisRenderer:
    def __init__(self, reg: Registry, limits) -> None:
        self.reg = reg
        self.limits = limits

    def _fragment(self, refid: str, namespace: str, dialect: str) -> Element | None:
        key = (refid.rsplit(".", 1)[0], refid.rsplit(".", 1)[1]) if "." in refid else (namespace, refid)
        cands = self.reg.fragments.get(key, [])
        specific = [c for c in cands if c.database_id == dialect]
        generic = [c for c in cands if c.database_id is None]
        return (specific or generic or [None])[0]

    def render_nodes(self, nodes: list[dict[str, Any]], ctx: RenderCtx, inc_props: dict[str, str], depth: int) -> str:
        out: list[str] = []
        for n in group_exclusive_ifs(nodes):
            t = n["type"]
            if t == "_exclusive_ifs":
                site = f"ifgroup@{ctx.path}:{n['line']}"
                if site not in [s for s, _ in ctx.choose_sites]:
                    ctx.choose_sites.append((site, len(n["members"])))
                idx = ctx.decisions.get(site, ctx.decisions.get("choose_default", 0))
                if idx == "otherwise_or_none" or idx >= len(n["members"]):
                    ctx.conditions.append(f"{site}: none of exclusive <if> on {n['expr']}")
                    continue
                member = n["members"][idx]
                ctx.conditions.append(f"{site}: if({member['test']})=true, other exclusive <if> on {n['expr']} false")
                out.append(self.render_nodes(member["children"], ctx, inc_props, depth))
            elif t == "text":
                out.append(self._subst_text(n["text"], ctx, inc_props))
            elif t == "placeholder":
                out.append(self._placeholder(n, ctx, inc_props))
            elif t == "if":
                take = ctx.if_default
                if take == "safe":
                    take = not self._needs_runtime_subst(n["children"], ctx, inc_props, depth)
                ctx.conditions.append(f"if({n['test']})={'true' if take else 'false'}")
                if take:
                    out.append(self.render_nodes(n["children"], ctx, inc_props, depth))
            elif t == "choose":
                site = f"choose@{ctx.path}:{n['line']}"
                if site not in [s for s, _ in ctx.choose_sites]:
                    ctx.choose_sites.append((site, len(n["branches"])))
                idx = ctx.decisions.get(site, ctx.decisions.get("choose_default", 0))
                branches = n["branches"]
                if idx == "otherwise_or_none":
                    idx = next((i for i, b in enumerate(branches) if b["type"] == "otherwise"), None)
                if idx is None or idx >= len(branches):
                    ctx.conditions.append(f"{site}: no branch")
                    continue
                b = branches[idx]
                ctx.conditions.append(f"{site}: {b['type']}" + (f"({b['test']})" if b.get("test") else ""))
                out.append(self.render_nodes(b["children"], ctx, inc_props, depth))
            elif t == "foreach":
                k = ctx.foreach_iterations
                ctx.conditions.append(f"foreach({n.get('collection')}) iterations={k}")
                body = [self.render_nodes(n["children"], ctx, inc_props, depth) for _ in range(k)]
                if k == 0:
                    continue
                out.append((n.get("open") or "") + (n.get("separator") or "").join(body) + (n.get("close") or ""))
            elif t in {"trim", "where", "set"}:
                out.append(self._trim(n, self.render_nodes(n["children"], ctx, inc_props, depth)))
            elif t == "include":
                if depth >= ctx.max_include_depth:
                    ctx.unresolved.append(f"include depth limit at {n['refid']}")
                    continue
                refid = self._subst_text(n["refid"], ctx, inc_props)
                frag = self._fragment(refid, ctx.namespace, ctx.dialect)
                if frag is None:
                    ctx.unresolved.append(f"include {refid} not found")
                    out.append(" /*unresolved include*/ ")
                    continue
                props = {k: self._subst_text(v or "", ctx, inc_props) for k, v in n["properties"].items()}
                prev_ns, prev_path = ctx.namespace, ctx.path
                ctx.namespace, ctx.path = frag.namespace, frag.path
                out.append(self.render_nodes(frag.tree, ctx, {**inc_props, **props}, depth + 1))
                ctx.namespace, ctx.path = prev_ns, prev_path
            elif t == "bind":
                ctx.params.append(QueryParam(name=n["name"], kind="bind_variable", resolved_value=None, java_type=None,
                                             type_handler=None, jdbc_type=None))
            elif t == "selectKey":
                ctx.diagnostics.append("selectKey body not rendered into statement")
            else:
                ctx.unresolved.append(f"unknown element <{n.get('tag')}>")
        return "".join(out)

    def _needs_runtime_subst(self, nodes: list[dict[str, Any]], ctx: RenderCtx, inc_props: dict[str, str], depth: int) -> bool:
        for n in nodes:
            if n["type"] == "placeholder" and n["syntax"] == "$" and n["expr"] not in inc_props and n["expr"] not in ctx.config_vars:
                return True
            if n["type"] == "include" and depth < ctx.max_include_depth:
                frag = self._fragment(n["refid"], ctx.namespace, ctx.dialect)
                if frag and self._needs_runtime_subst(frag.tree, ctx, {**inc_props, **n["properties"]}, depth + 1):
                    return True
            for key in ("children",):
                if key in n and self._needs_runtime_subst(n[key], ctx, inc_props, depth):
                    return True
            for b in n.get("branches", []) + n.get("members", []):
                if self._needs_runtime_subst(b["children"], ctx, inc_props, depth):
                    return True
        return False

    def _subst_text(self, text: str, ctx: RenderCtx, inc_props: dict[str, str]) -> str:
        """Resolves ${} in raw text (include refids / property values), leaving #{} untouched."""
        def repl(m: re.Match[str]) -> str:
            if m.group(1) == "#":
                return m.group(0)
            return self._dollar(m.group(2).strip(), ctx, inc_props, render_binds=False)
        return PLACEHOLDER_RE.sub(repl, text)

    def _dollar(self, expr: str, ctx: RenderCtx, inc_props: dict[str, str], render_binds: bool = True) -> str:
        if expr in inc_props:
            ctx.params.append(QueryParam(name=expr, kind="include_property", resolved_value=inc_props[expr]))
            return inc_props[expr]
        if expr in ctx.config_vars:
            value = ctx.config_vars[expr]
            ctx.params.append(QueryParam(name=expr, kind="config_property", resolved_value=value))
            if render_binds:
                value = PLACEHOLDER_RE.sub(lambda m: self._placeholder(_placeholder(m.group(1), m.group(2)), ctx, inc_props), value)
            return value
        ctx.params.append(QueryParam(name=expr, kind="substitution"))
        ctx.unresolved.append(f"${{{expr}}} runtime text substitution")
        return f"${{{expr}}}"

    def _placeholder(self, n: dict[str, Any], ctx: RenderCtx, inc_props: dict[str, str]) -> str:
        if n["syntax"] == "$":
            return self._dollar(n["expr"], ctx, inc_props)
        ctx.params.append(QueryParam(name=n["expr"], kind="bind", java_type=n.get("javaType"), jdbc_type=n.get("jdbcType"),
                                     type_handler=n.get("typeHandler")))
        return "?"

    @staticmethod
    def _trim(n: dict[str, Any], body: str) -> str:
        t = n["type"]
        prefix = n.get("prefix") or ("WHERE" if t == "where" else "SET" if t == "set" else "")
        suffix = n.get("suffix") or ""
        pre_over = n.get("prefixOverrides") or ("AND |OR " if t == "where" else "")
        suf_over = n.get("suffixOverrides") or ("," if t == "set" else "")
        s = body.strip()
        if not s:
            return ""
        for p in [x for x in pre_over.split("|") if x]:
            if s.upper().startswith(p.strip().upper()) and (len(s) == len(p.strip()) or not s[len(p.strip())].isalnum()):
                s = s[len(p.strip()):].lstrip()
                break
        for p in [x for x in suf_over.split("|") if x]:
            if s.upper().endswith(p.strip().upper()):
                s = s[: len(s) - len(p.strip())].rstrip()
                break
        return f" {prefix} {s} {suffix} "

    def variants(self, st: Element, dialect: str, config_vars: dict[str, str], config_sources: dict[str, str],
                 evidence: str) -> tuple[list[QueryVariant], list[str]]:
        max_v = self.limits.mybatis_max_variants
        # all-true, safe-true (skip branches needing runtime ${}), all-false; then choose alternatives on the safe base
        plans: list[tuple[bool | str, dict[str, Any]]] = [
            (True, {"choose_default": 0}), ("safe", {"choose_default": 0}), (False, {"choose_default": "otherwise_or_none"})]
        out: list[QueryVariant] = []
        seen: dict[str, QueryVariant] = {}
        diags: list[str] = []
        sites: list[tuple[str, int]] = []
        i = 0
        while i < len(plans) and len(out) < max_v:
            if_default, decisions = plans[i]
            ctx = RenderCtx(dialect=dialect, config_vars=config_vars, config_sources=config_sources, decisions=decisions,
                            if_default=if_default, foreach_iterations=self.limits.mybatis_foreach_iterations,
                            max_include_depth=self.limits.mybatis_max_include_depth, namespace=st.namespace, path=st.path)
            sql = re.sub(r"\s+", " ", self.render_nodes(st.tree, ctx, {}, 0)).strip()
            diags.extend(d for d in ctx.diagnostics if d not in diags)
            if i == 0:
                sites = ctx.choose_sites
                for site, n in sites:  # one alternative-branch variant per choose site and branch
                    for b in range(1, n):
                        plans.append(("safe", {"choose_default": 0, site: b}))
            if sql in seen:
                seen[sql].conditions.append("| " + "; ".join(_short_conditions(ctx.conditions)))
            else:
                v = QueryVariant(dialect=dialect, dialect_evidence=evidence, conditions=_short_conditions(ctx.conditions), sql=sql,
                                 partial=bool(ctx.unresolved), unresolved=ctx.unresolved, parameters=ctx.params)
                seen[sql] = v
                out.append(v)
            i += 1
        planned = 3 + sum(n - 1 for _, n in sites)
        if planned > len(plans[:i]):
            diags.append(f"variant limit {max_v} reached: {planned} planned for {dialect}")
        return out, diags


def _short_conditions(conds: list[str]) -> list[str]:
    """Collapse repeated if decisions; keep choose/foreach decisions."""
    ifs_true = sum(1 for c in conds if c.startswith("if(") and c.endswith("=true"))
    ifs_false = sum(1 for c in conds if c.startswith("if(") and c.endswith("=false"))
    others = list(dict.fromkeys(c for c in conds if not c.startswith("if(")))
    head = []
    if ifs_true:
        head.append(f"all {ifs_true} <if> tests true")
    if ifs_false:
        head.append(f"all {ifs_false} <if> tests false")
    return head + others


def statement_queries(reg: Registry, renderer: MyBatisRenderer, dialects: list[str],
                      vendor: dict[str, VendorProperties], table_prefix: str) -> list[QueryDef]:
    """Builds QueryDefs for every statement and fragment, with variants for every effective dialect."""
    by_key: dict[tuple[str, str], list[Element]] = {}
    for st in reg.statements:
        by_key.setdefault((st.namespace, st.id), []).append(st)
    out: list[QueryDef] = []
    for st in reg.statements:
        siblings = by_key[(st.namespace, st.id)]
        if st.database_id:
            eff = [st.database_id] if st.database_id in dialects else []
        else:
            overridden = {s.database_id for s in siblings if s.database_id}
            eff = [d for d in dialects if d not in overridden]
        q = _query_def(st)
        for d in eff:
            vp = vendor.get(d)
            config_vars = {"prefix": table_prefix, **(vp.values if vp else {})}
            sources = {"prefix": "profile parameter table_prefix", **({k: vp.path for k in vp.values} if vp else {})}
            evidence = f"databaseId={st.database_id}" if st.database_id else f"generic statement effective for {d} (no databaseId override)"
            vs, diags = renderer.variants(st, d, config_vars, sources, evidence)
            q.variants.extend(vs)
            q.diagnostics.extend(x for x in diags if x not in q.diagnostics)
        if not eff:
            q.diagnostics.append(f"databaseId {st.database_id} is not among configured dialects")
        out.append(q)
    for frags in reg.fragments.values():
        for f in frags:
            out.append(_query_def(f))
    return out


def _query_def(e: Element) -> QueryDef:
    key = f"mybatis:{e.namespace}.{e.id}" + (f"[{e.database_id}]" if e.database_id else "")
    return QueryDef(
        key=key, language="sql", origin="mybatis", statement_kind="sql_fragment" if e.tag == "sql" else e.tag,
        namespace=e.namespace, statement_id=e.id, database_id=e.database_id, result_map=e.attrs.get("resultMap"),
        parameter_type=e.attrs.get("parameterType"), result_type=e.attrs.get("resultType"),
        template={"type": e.tag, "id": e.id, "databaseId": e.database_id, "children": e.tree}, raw_text=e.raw,
        description=e.comment, is_test=e.is_test,
        provenance=Provenance(path=e.path, line_start=e.line_start, line_end=e.line_end, symbol=f"{e.namespace}.{e.id}",
                              extractor=EXTRACTOR, evidence_kind="mybatis_" + e.tag),
    )
