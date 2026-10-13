from camunda_rag.discovery import file_role
from camunda_rag.extractors.liquibase import LiquibaseExtractor
from camunda_rag.models import FileRole
from camunda_rag.schema import eval_filter_expression, reconstruct

MASTER = "db/schema/src/main/resources/db/changelog/app/changelog-master.xml"


def _extract(repo):
    known = {str(p.relative_to(repo)) for p in repo.rglob("*") if p.is_file() and ".git" not in p.parts}
    lq = LiquibaseExtractor(repo, known, lambda p: file_role(p) == FileRole.test)
    files = [f for f in known if f.endswith((".xml", ".yaml")) and "changelog" in f]
    assert lq.roots(files) == [MASTER]
    return lq.extract_root(MASTER)


def _profile(res, dialect, contexts=()):
    return reconstruct(res.changes, res.properties, component=res.component, dialect=dialect, contexts=list(contexts),
                       labels=[], parameters={"prefix": "", "userCharColumnSize": "256"}, is_test=False)


def test_includes_are_followed_in_order_across_xml_and_yaml(mini_repo):
    res = _extract(mini_repo)
    assert [c.changeset_id for c in res.changes][:2] == ["create_orders", "create_order_items"]
    assert res.changes[-1].changeset_id == "sql_alter"
    assert res.files[-1].endswith("1.1.yaml")
    assert not res.diagnostics


def test_profile_replay_dbms_context_rename_sql_and_uncertainty(mini_repo):
    res = _extract(mini_repo)
    pg = _profile(res, "postgresql")
    tables = {e.name: e for e in pg.entities}
    assert set(tables) == {"ORDERS", "ORDER_ITEM", "TEST_ONLY"}  # no contexts requested -> all contexts run
    orders = {c.name: c for c in tables["ORDERS"].columns}
    assert "ORA_HINT" not in orders  # dbms=oracle only
    assert "NOTE" in orders  # applied from raw SQL via SQLGlot
    assert orders["PAYLOAD"].data_type == "TEXT"  # dbms-specific changelog property
    assert orders["CUSTOMER_ID"].data_type == "NVARCHAR(256)"
    assert orders["STATE"].nullable is False
    assert "PRODUCT_SKU" in {c.name for c in tables["ORDER_ITEM"].columns}
    fk = [k for k in tables["ORDER_ITEM"].constraints if k.kind == "foreign_key"][0]
    assert (fk.ref_table, fk.ref_columns) == ("ORDERS", ["ORDER_KEY"])
    # sqlCheck precondition cannot be evaluated statically -> profile is partial, change kept as uncertain
    assert pg.completeness["level"] == "partial"
    assert any(o.startswith("uncertain_precondition") for _, o in pg.applied_changes)

    ora = _profile(res, "oracle", contexts=["prod"])
    ora_tables = {e.name: e for e in ora.entities}
    assert "TEST_ONLY" not in ora_tables
    assert "ORA_HINT" in {c.name for c in ora_tables["ORDERS"].columns}
    assert {c.name: c for c in ora_tables["ORDERS"].columns}["PAYLOAD"].data_type == "CLOB"


def test_precondition_marks_ran_when_table_exists(mini_repo):
    res = _extract(mini_repo)
    twice = res.changes[:1] + [res.changes[0].model_copy(update={"order": 99})]
    p = reconstruct(twice, res.properties, component="c", dialect="postgresql", contexts=[], labels=[],
                    parameters={"prefix": ""}, is_test=False)
    assert [o for _, o in p.applied_changes] == ["applied", "skipped:precondition"]


def test_context_expressions():
    assert eval_filter_expression("test", [])
    assert not eval_filter_expression("test", ["prod"])
    assert eval_filter_expression("test, prod", ["prod"])
    assert eval_filter_expression("!test", ["prod"])
    assert not eval_filter_expression("prod and !test", ["prod", "test"])
