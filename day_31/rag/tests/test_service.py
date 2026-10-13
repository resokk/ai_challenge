import sqlite3

from camunda_rag.storage import current_generation, generations_dir


def test_inventory_records_status_for_every_tracked_file(indexed, mini_repo):
    db = sqlite3.connect(generations_dir(indexed.data_dir) / current_generation(indexed.data_dir) / "catalog.sqlite")
    rows = dict(db.execute("SELECT path, status || ':' || role FROM source_files").fetchall())
    tracked = {str(p.relative_to(mini_repo)) for p in mini_repo.rglob("*") if p.is_file() and ".git" not in p.parts}
    assert set(rows) == tracked
    assert rows["web/src/query.ts"].startswith("unsupported")  # SQL candidate, no extractor for TS
    assert rows["web/src/app.ts"].startswith("irrelevant")
    assert rows["db/store/src/test/resources/sql/init.h2.sql"] == "processed:test"
    assert rows["docs/data-model.md"] == "processed:doc"
    assert rows["db/store/src/main/java/org/acme/store/OrderDao.java"].startswith("unsupported")  # --skip-java


def test_describe_reads_full_catalog_with_provenance(service):
    d = service.describe("ORDERS")
    assert d["status"] == "ok"
    assert d["entity"]["storage_kind"] == "relational_table"
    assert d["purpose"] == {"text": "Customer orders", "status": "declared", "sources": d["purpose"]["sources"]}
    cols = {c["name"]: c for c in d["columns"]}
    assert cols["CUSTOMER_ID"]["purpose"] == "External customer id"
    assert cols["STATE"]["purpose"] == "назначение не установлено по исходникам"
    assert cols["ORDER_KEY"]["java_mappings"][0]["property"] == "orderKey"
    assert "ORA_HINT" not in cols
    assert d["profile"]["completeness"]["level"] == "partial"
    assert d["evidence"][0]["commit_sha"] and d["evidence"][0]["line_start"]
    assert d["usage_summary"]["status"] == "inferred"
    assert any(c["changeset_id"] == "unknown_check" and c["outcome"].startswith("uncertain") for c in d["schema_changes"])


def test_declared_fk_is_separated_from_join_inference(service):
    rel = service.describe("ORDER_ITEM")["relationships"]
    assert rel["declared_foreign_keys"][0]["to"] == "ORDERS(ORDER_KEY)"
    assert rel["inferred_from_joins"] and rel["inferred_from_joins"][0]["status"] == "recovered"


def test_describe_dialect_profile_and_non_relational_entities(service):
    ora = service.describe("ORDERS", dialect="oracle")
    assert "ORA_HINT" in {c["name"] for c in ora["columns"]}
    es = service.describe("acme-order")
    assert es["entity"]["storage_kind"] == "elasticsearch_index"
    assert {"customer.name", "customer.name.raw"} <= {c["name"] for c in es["columns"]}
    assert service.describe("ORDRS")["status"] == "not_found"


def test_queries_by_table_column_and_access(service):
    res = service.queries(table="ORDERS", access="write")
    keys = {q["key"] for q in res["queries"]}
    assert {"mybatis:org.acme.store.OrderMapper.insert", "mybatis:org.acme.store.OrderMapper.updateState"} <= keys
    s = service.queries(column="PRODUCT_SKU")["queries"][0]
    assert s["key"] == "mybatis:org.acme.store.OrderMapper.search"
    assert s["representative"]["dialect"] == "postgresql"
    assert "postgresql" in s["variant_statuses"] and "oracle" in s["variant_statuses"]
    purge = service.queries(mapper="OrderMapper", text="purge", include_variants=True)["queries"][0]
    assert {v["parse_status"] for v in purge["variants"]} == {"partial"}


def test_search_uses_exact_fts_and_dense_channels(indexed):
    from camunda_rag.service import CatalogService
    from conftest import FakeEmbedder

    svc = CatalogService(indexed, embedder=FakeEmbedder())
    res = svc.search("ORDER_ITEM table", k=5)
    assert res["channels_used"] == ["exact", "fts", "dense"]
    top = res["hits"][0]
    assert top["doc_type"] == "entity" and top["title"].endswith("ORDER_ITEM")
    assert "exact" in top["channels"]
    docs = svc.search("search index mirrors orders", k=5, doc_types=["documentation"])
    assert docs["hits"] and docs["hits"][0]["provenance"]["path"] == "docs/data-model.md"
    es = svc.search("customer", storage_kind="elasticsearch_index")
    assert all(h["storage_kind"] == "elasticsearch_index" for h in es["hits"])


def test_validation_against_catalog(service):
    ok = service.validate("SELECT o.STATE FROM ORDERS o JOIN ORDER_ITEM i ON i.ORDER_KEY = o.ORDER_KEY WHERE i.PRODUCT_SKU = ?")
    assert ok.status == "valid"
    bad = service.validate("SELECT SKU FROM ORDER_ITEM")  # renamed by the YAML changeset
    assert bad.status == "invalid" and bad.statements[0].issues[-1].code == "unknown_column"
    assert service.validate("SELECT * FROM ORDRS").statements[0].issues[-1].code == "unknown_table"
    assert service.validate("SELECT ORA_HINT FROM ORDERS", dialect="oracle").status == "valid"
    assert service.validate("SELECT ORA_HINT FROM ORDERS", dialect="postgresql").status == "invalid"
    es = service.validate('SELECT * FROM "acme-order"')
    assert es.statements[0].issues[-1].code == "non_relational_entity"
    assert service.validate("SELECT * FROM ORDERS WHERE STATE = #{state}").issues[0].code == "unrendered_template"
    assert service.validate("VACUUM").status == "unsupported"
    assert service.validate("SELECT FROM WHERE").status == "parse_error"


def test_generation_switch_is_atomic(indexed, service):
    from camunda_rag.storage import publish

    before = service.status()["generation"]
    publish(indexed.data_dir, before)  # republish same id: pointer replaced atomically, service keeps working
    assert service.describe("ORDERS")["generation"] == before
