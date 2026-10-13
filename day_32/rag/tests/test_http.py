import json

from fastapi.testclient import TestClient

from camunda_rag.http_app import create_app

MCP_HEADERS = {"accept": "application/json, text/event-stream", "content-type": "application/json"}


def _rpc(client, method, params, id_=1):
    r = client.post("/mcp", headers=MCP_HEADERS, json={"jsonrpc": "2.0", "id": id_, "method": method, "params": params})
    assert r.status_code == 200, r.text
    return r.json()


def test_rest_validate_and_health(indexed, service):
    with TestClient(create_app(indexed, service), base_url="http://127.0.0.1:8765") as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/validate", json={"sql": "SELECT STATE FROM ORDERS", "dialect": "postgresql"})
        assert r.status_code == 200 and r.json()["status"] == "valid"
        assert c.post("/validate", json={"sql": "SELECT 1", "dialect": "db2"}).status_code == 422


def test_mcp_tools_over_streamable_http(indexed, service):
    with TestClient(create_app(indexed, service), base_url="http://127.0.0.1:8765") as c:
        init = _rpc(c, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                      "clientInfo": {"name": "test", "version": "0"}})
        assert init["result"]["serverInfo"]["name"] == "camunda-data-model"
        tools = _rpc(c, "tools/list", {}, 2)["result"]["tools"]
        assert {t["name"] for t in tools} == {"search", "describe", "queries", "validate_sql"}
        res = _rpc(c, "tools/call", {"name": "describe", "arguments": {"name": "ORDER_ITEM"}}, 3)["result"]
        payload = res.get("structuredContent") or json.loads(res["content"][0]["text"])
        payload = payload.get("result", payload)
        assert payload["entity"]["name"] == "ORDER_ITEM"
        res = _rpc(c, "tools/call", {"name": "queries", "arguments": {"table": "ORDERS", "access": "write"}}, 4)["result"]
        assert not res.get("isError")
        res = _rpc(c, "tools/call", {"name": "validate_sql",
                                     "arguments": {"sql": "SELECT ID FROM NO_SUCH_TABLE", "dialect": "postgresql"}}, 5)["result"]
        payload = res.get("structuredContent") or json.loads(res["content"][0]["text"])
        assert payload["status"] == "invalid" and "unknown_table" in {i["code"] for i in payload["statements"][0]["issues"]}
        res = _rpc(c, "tools/call", {"name": "validate_sql", "arguments": {"sql": "SELECT 1", "dialect": "db2"}}, 6)["result"]
        assert res.get("isError") and "db2" in res["content"][0]["text"]
