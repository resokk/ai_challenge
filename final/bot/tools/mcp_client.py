"""MCP servers from `mcp.json`, their tools adapted into the registry."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import Client, StdioServerParameters

from run_as import UserContext
from tools.registry import Registry, Tool, ToolError

logger = logging.getLogger(__name__)


def load_config(path: Path) -> dict[str, dict]:
    """`{"servers": {name: {"command", "args", "env"} | {"url"}}}`."""
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("servers", {})


def _server(spec: dict):
    if "url" in spec:
        return spec["url"]
    return StdioServerParameters(command=spec["command"], args=spec.get("args", []), env=spec.get("env"))


def _result_text(result) -> str:
    parts = []
    for item in result.content:
        text = getattr(item, "text", None)
        parts.append(text if text is not None else f"[{item.type}]")
    if not parts and result.structured_content is not None:
        parts.append(json.dumps(result.structured_content, ensure_ascii=False))
    text = "\n".join(parts)
    if result.is_error:
        raise ToolError(text or "MCP-инструмент вернул ошибку")
    return text


def adapt(server: str, client, tool) -> Tool:
    """A registry tool calling `tool` on `client`; the user context is not the
    MCP server's business."""

    async def call(ctx: UserContext, arguments: dict) -> str:
        return _result_text(await client.call_tool(tool.name, arguments))

    return Tool(
        name=f"{server}__{tool.name}",
        description=tool.description or tool.title or tool.name,
        parameters=tool.input_schema or {"type": "object", "properties": {}},
        func=call,
    )


async def _list_tools(client) -> list:
    tools, cursor = [], None
    while True:
        page = await client.list_tools(cursor=cursor)
        tools += page.tools
        cursor = page.next_cursor
        if not cursor:
            return tools


async def connect_all(path: Path, registry: Registry, stack: AsyncExitStack) -> None:
    """Connect every configured server and register its tools. A server that
    does not come up is logged and skipped; the sessions live on `stack`."""
    for name, spec in load_config(path).items():
        server_stack = AsyncExitStack()
        try:
            client = await server_stack.enter_async_context(Client(_server(spec)))
            tools = [adapt(name, client, tool) for tool in await _list_tools(client)]
        except Exception:
            logger.exception("MCP server %s is unavailable, skipping it", name)
            try:
                await server_stack.aclose()
            except Exception:
                logger.debug("Closing failed MCP server %s", name, exc_info=True)
            continue
        stack.push_async_callback(server_stack.aclose)
        for tool in tools:
            try:
                registry.register(tool)
            except Exception as error:
                logger.warning("MCP tool %s skipped: %s", tool.name, error)
        logger.info("MCP server %s: %d tools", name, len(tools))


class McpServers:
    """Owns the MCP sessions in one long-lived task: anyio requires a session
    to be closed by the task that opened it."""

    def __init__(self) -> None:
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def start(self, path: Path, registry: Registry) -> None:
        ready = asyncio.Event()
        self._task = asyncio.create_task(self._run(path, registry, ready))
        await ready.wait()

    async def _run(self, path: Path, registry: Registry, ready: asyncio.Event) -> None:
        try:
            async with AsyncExitStack() as stack:
                await connect_all(path, registry, stack)
                ready.set()
                await self._stop.wait()
        finally:
            ready.set()

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            try:
                await self._task
            except Exception:
                logger.exception("Closing MCP sessions failed")
