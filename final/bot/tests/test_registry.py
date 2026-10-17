"""The dispatcher and the MCP adapter."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from types import SimpleNamespace

import config
from tools import mcp_client
from tools.registry import Registry, Tool, ToolError

SCHEMA = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"], "additionalProperties": False}


def registry_with(func, timeout=None) -> Registry:
    registry = Registry()
    registry.register(Tool("echo", "Echo", SCHEMA, func, timeout))
    return registry


async def test_unknown_tool_lists_available(make_ctx):
    reply = await registry_with(lambda ctx, args: None).call_tool(make_ctx(), "nope", {})
    assert "nope" in reply.text and "echo" in reply.text


async def test_invalid_arguments_are_text(make_ctx):
    async def echo(ctx, args):
        return str(args["n"])

    registry = registry_with(echo)
    assert "n" in (await registry.call_tool(make_ctx(), "echo", {})).text
    assert "integer" in (await registry.call_tool(make_ctx(), "echo", {"n": "x"})).text
    assert (await registry.call_tool(make_ctx(), "echo", {"n": 3})).text == "3"


async def test_timeout_is_text(make_ctx, monkeypatch):
    monkeypatch.setattr(config, "TOOL_TIMEOUT_SEC", 0.1)

    async def slow(ctx, args):
        await asyncio.sleep(5)

    reply = await registry_with(slow).call_tool(make_ctx(), "echo", {"n": 1})
    assert "не ответил" in reply.text


async def test_exceptions_are_text(make_ctx):
    async def broken(ctx, args):
        raise RuntimeError("boom")

    async def refused(ctx, args):
        raise ToolError("понятная причина")

    assert "boom" in (await registry_with(broken).call_tool(make_ctx(), "echo", {"n": 1})).text
    assert "понятная причина" in (await registry_with(refused).call_tool(make_ctx(), "echo", {"n": 1})).text


async def test_result_is_truncated(make_ctx, monkeypatch):
    monkeypatch.setattr(config, "TOOL_RESULT_MAX_CHARS", 50)

    async def big(ctx, args):
        return "x" * 500

    text = (await registry_with(big).call_tool(make_ctx(), "echo", {"n": 1})).text
    assert text.startswith("x" * 50) and "обрезано" in text and len(text) < 200


class FakeClient:
    def __init__(self):
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def list_tools(self, cursor=None):
        tool = SimpleNamespace(name="search", title=None, description="Search things",
                               input_schema={"type": "object", "properties": {"q": {"type": "string"}}})
        return SimpleNamespace(tools=[tool], next_cursor=None)

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=f"found {arguments['q']}")],
                               structured_content=None, is_error=False)


async def test_mcp_tools_are_registered_with_prefix(tmp_path, monkeypatch, make_ctx):
    config_path = tmp_path / "mcp.json"
    config_path.write_text('{"servers": {"docs": {"command": "x"}, "down": {"url": "http://127.0.0.1:1/mcp"}}}')
    client = FakeClient()

    def fake_client(server):
        if isinstance(server, str):
            raise ConnectionError("down")
        return client

    monkeypatch.setattr(mcp_client, "Client", fake_client)
    registry = Registry()
    async with AsyncExitStack() as stack:
        await mcp_client.connect_all(config_path, registry, stack)
        assert list(registry.tools) == ["docs__search"]
        reply = await registry.call_tool(make_ctx(), "docs__search", {"q": "pathlib"})
    assert reply.text == "found pathlib"
    assert client.calls == [("search", {"q": "pathlib"})]
    assert "docs__search" in registry.catalog()
