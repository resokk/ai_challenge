from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import yandex

# Over HTTP like day_16's calculator: a service of its own that clients POST
# MCP to. Auth comes from the client's MCP connection settings - it sends
# "Authorization: Bearer <Yandex OAuth token>" and this server passes the token
# on to Yandex. Nothing is stored here, so each client reads its own home.
server = MCPServer("ya-sh")

# Loopback only: the token travels to this server in plain HTTP.
HOST = "127.0.0.1"
PORT = 8766


def _token(ctx: Context) -> str:
    auth = (ctx.headers or {}).get("authorization", "")
    scheme, _, token = auth.partition(" ")
    if scheme.lower() not in ("bearer", "oauth") or not token.strip():
        raise ToolError("No Yandex OAuth token: set the header 'Authorization: Bearer <token>' in the MCP client")
    return token.strip()


@server.tool()
async def list_weather_sensors(ctx: Context) -> list[dict[str, str | None]]:
    """List the weather (climate) sensors in the Yandex Smart Home: id, name and room.

    Pass a sensor's id to get_temperature to read it.
    """
    try:
        return await yandex.list_weather_sensors(_token(ctx))
    except yandex.YandexError as exc:
        # As in calc-mcp: only a ToolError's message reaches the caller.
        raise ToolError(str(exc)) from None


@server.tool()
async def get_temperature(sensor_id: str, ctx: Context) -> dict[str, Any]:
    """Read the current temperature from a sensor, by the id list_weather_sensors gave."""
    try:
        return await yandex.get_temperature(_token(ctx), sensor_id)
    except yandex.YandexError as exc:
        raise ToolError(str(exc)) from None


if __name__ == "__main__":
    # Stateless and plain JSON, for the same reasons as calc-mcp: every call
    # stands alone (the token rides on each request) and answers are immediate.
    server.run(
        transport="streamable-http",
        host=HOST,
        port=PORT,
        json_response=True,
        stateless_http=True,
    )
