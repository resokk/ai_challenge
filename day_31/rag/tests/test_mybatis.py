from camunda_rag.config import Limits
from camunda_rag.extractors import mybatis as mb
from camunda_rag.extractors.vendor_properties import load_vendor_properties

MAPPERS = ["db/store/src/main/resources/mapper/Commons.xml", "db/store/src/main/resources/mapper/OrderMapper.xml"]
VENDOR = [f"db/schema/src/main/resources/db/vendor-properties/{d}.properties" for d in ("postgresql", "oracle", "h2")]


def _queries(repo, max_variants=8):
    reg = mb.Registry()
    for m in MAPPERS:
        mb.parse_mapper(repo, m, False, reg)
    vendor = load_vendor_properties(repo, VENDOR)
    qs = mb.statement_queries(reg, mb.MyBatisRenderer(reg, Limits(mybatis_max_variants=max_variants)),
                              ["postgresql", "oracle", "h2"], vendor, "")
    return {q.key: q for q in qs}, reg


def test_template_tree_keeps_dynamic_structure(mini_repo):
    qs, _ = _queries(mini_repo)
    tree = qs["mybatis:org.acme.store.OrderMapper.search"].template["children"]
    types = {n["type"] for n in tree}
    assert {"bind", "where", "if", "include", "placeholder", "text"} <= types
    sort_if = [n for n in tree if n["type"] == "if" and "sort" in n["test"]][0]
    ph = [n for n in sort_if["children"] if n["type"] == "placeholder"][0]
    assert (ph["syntax"], ph["expr"]) == ("$", "sort.column")


def test_dialect_specific_includes_and_config_variables(mini_repo):
    qs, _ = _queries(mini_repo)
    search = qs["mybatis:org.acme.store.OrderMapper.search"]
    by_dialect = {}
    for v in search.variants:
        by_dialect.setdefault(v.dialect, []).append(v)
    assert set(by_dialect) == {"postgresql", "oracle", "h2"}
    pg_complete = [v for v in by_dialect["postgresql"] if not v.partial]
    assert pg_complete and all(v.sql.endswith("LIMIT ? OFFSET ?") for v in pg_complete)
    assert any("FETCH NEXT ? ROWS ONLY" in v.sql for v in by_dialect["oracle"])
    kinds = {(p.name, p.kind) for v in pg_complete for p in v.parameters}
    assert ("paging.after", "config_property") in kinds and ("page.size", "bind") in kinds


def test_runtime_substitution_marks_variant_partial_without_inventing_names(mini_repo):
    qs, _ = _queries(mini_repo)
    search = qs["mybatis:org.acme.store.OrderMapper.search"]
    partial = [v for v in search.variants if v.partial]
    assert partial and all("${sort.column}" in v.sql for v in partial)
    assert all(any("sort.column" in u for u in v.unresolved) for v in partial)
    purge = qs["mybatis:org.acme.store.OrderMapper.purge"]
    assert all(v.partial and "${tableName}" in v.sql for v in purge.variants)


def test_where_set_trim_and_exclusive_ifs(mini_repo):
    qs, _ = _queries(mini_repo)
    search = qs["mybatis:org.acme.store.OrderMapper.search"]
    for v in search.variants:
        assert "WHERE AND" not in v.sql
        assert not ("CUSTOMER_ID = 'a'" in v.sql and "CUSTOMER_ID = 'b'" in v.sql)  # mutually exclusive <if>s
    assert any("CUSTOMER_ID = 'b'" in v.sql for v in search.variants)  # alternative branch is explored
    assert any(" IN (?)" in v.sql for v in search.variants)  # <choose> alternative with <foreach>
    upd = qs["mybatis:org.acme.store.OrderMapper.updateState"]
    full = [v for v in upd.variants if "STATE = ?" in v.sql][0]
    assert "SET STATE = ?, CUSTOMER_ID = ? WHERE" in full.sql  # trailing comma removed by <set>


def test_variant_limit_is_enforced_and_reported(mini_repo):
    qs, _ = _queries(mini_repo, max_variants=2)
    search = qs["mybatis:org.acme.store.OrderMapper.search"]
    assert len([v for v in search.variants if v.dialect == "postgresql"]) <= 2
    assert any("variant limit" in d for d in search.diagnostics)


def test_result_map_constructor_and_properties(mini_repo):
    _, reg = _queries(mini_repo)
    cols = {(m.column, m.property) for m in reg.result_maps}
    assert ("ORDER_KEY", "orderKey") in cols and ("STATE", "state") in cols
