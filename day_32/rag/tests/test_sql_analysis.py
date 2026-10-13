from camunda_rag.sql_analysis import analyze


def test_tables_columns_joins_and_lineage():
    a = analyze("SELECT o.ORDER_KEY, COUNT(*) AS n FROM ORDERS o JOIN ORDER_ITEM i ON i.ORDER_KEY = o.ORDER_KEY "
                "WHERE o.STATE = ? GROUP BY o.ORDER_KEY ORDER BY n", "postgresql")
    assert a.status == "parsed" and a.statement_type == "select"
    assert {(t.name, t.access) for t in a.tables} == {("ORDERS", "read"), ("ORDER_ITEM", "read")}
    j = a.joins[0]
    assert {(j.left_table, j.left_column), (j.right_table, j.right_column)} == {("ORDER_ITEM", "ORDER_KEY"), ("ORDERS", "ORDER_KEY")}
    assert ("ORDERS", "STATE", "where") in {(c.table, c.column, c.clause) for c in a.columns}
    assert a.aggregates == ["COUNT"] and a.placeholders == 1
    assert {"output": "ORDER_KEY", "sources": ["ORDERS.ORDER_KEY"]} in a.lineage


def test_write_dependencies_and_unqualified_columns_with_schema():
    a = analyze("UPDATE ORDERS SET STATE = ? WHERE ORDER_KEY IN (SELECT ORDER_KEY FROM ORDER_ITEM WHERE SKU = ?)",
                "postgresql", {"ORDERS": {"STATE", "ORDER_KEY"}, "ORDER_ITEM": {"ORDER_KEY", "SKU"}})
    assert {(t.name, t.access) for t in a.tables} == {("ORDERS", "write"), ("ORDER_ITEM", "read")}
    assert ("ORDERS", "STATE", "set") in {(c.table, c.column, c.clause) for c in a.columns}
    assert ("ORDER_ITEM", "SKU") in {(c.table, c.column) for c in a.columns}


def test_fingerprint_ignores_literals_and_opaque_command_is_unsupported():
    a = analyze("SELECT * FROM ORDERS WHERE STATE = 'A'", "postgresql")
    b = analyze("SELECT * FROM ORDERS WHERE STATE = 'B'", "postgresql")
    assert a.fingerprint == b.fingerprint
    assert analyze("VACUUM ORDERS", "postgresql").status == "unsupported"
    assert analyze("SELECT FROM WHERE", "postgresql").status == "error"


def test_dialects_are_respected():
    assert analyze("SELECT TOP 5 * FROM ORDERS", "mssql").status == "parsed"
    assert analyze("SELECT * FROM ORDERS FETCH FIRST 5 ROWS ONLY", "oracle").status == "parsed"
    h2 = analyze("SELECT 1", "h2")
    assert any("generic" in d for d in h2.diagnostics)
