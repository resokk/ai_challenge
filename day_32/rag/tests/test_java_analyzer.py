import shutil

import pytest

from camunda_rag.config import Settings
from camunda_rag.extractors import java as java_x

pytestmark = pytest.mark.skipif(not Settings().java_analyzer_jar.exists() or shutil.which("java") is None,
                                reason="java-analyzer jar not built (camunda-rag build-java)")

FILES = ["db/store/src/main/java/org/acme/store/OrderDao.java", "db/store/src/main/java/org/acme/store/OrderMapper.java",
         "db/store/src/main/java/org/acme/store/OrderEntity.java", "db/store/src/main/java/org/acme/store/TableNames.java"]


def test_analyzer_evaluates_strings_constants_builders_and_mappers(mini_repo, tmp_path):
    s = Settings(repo_path=mini_repo)
    out = java_x.run_analyzer(s, mini_repo, FILES, FILES, ["org.acme.store.OrderMapper"], tmp_path)
    queries, symbols, files, _, header = java_x.convert(java_x.read_jsonl(out), lambda p: False, "")
    assert header["schema_version"] == 1
    sqls = {v.sql for q in queries for v in q.variants}
    # StringBuilder with an if: both branches, cross-file constant, `prefix` parameter -> profile prefix (assumed)
    assert "SELECT COUNT(*) FROM ORDERS" in sqls and "SELECT COUNT(*) FROM ORDERS WHERE STATE = 'OPEN'" in sqls
    assert "SELECT ORDER_KEY, STATE FROM ORDERS WHERE ORDER_KEY = ?" in sqls  # static field + local + concat
    assert any(v.startswith("DELETE FROM ORDER_ITEM") for v in sqls)  # text block
    assert not any("hello" in v for v in sqls)  # non-SQL strings are not query candidates
    builder = [q for q in queries if q.statement_kind == "string_builder"][0]
    assert any("onlyOpen" in c for v in builder.variants for c in v.conditions)
    mapper = [s for s in symbols if s.kind == "method" and s.name == "purge"][0]
    assert mapper.attributes["params"][0]["param_name"] == "tableName"
    assert files["db/store/src/main/java/org/acme/store/TableNames.java"]["records"] == 0
    assert any(s.fqn == "org.acme.store.OrderEntity" and s.javadoc for s in symbols)
