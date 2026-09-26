from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import storage

# Over HTTP like calc-mcp and ya_sh_mcp: a service of its own that clients
# POST MCP to. The AI reads a sensor elsewhere (ya_sh_mcp, for instance) and
# hands the reading here to be kept.
server = MCPServer("sensor-log")

# Loopback only: nothing authenticates the callers, and anyone who can reach
# the port can write samples.
HOST = "127.0.0.1"
PORT = 8767


@server.tool()
def save_sample(sensor_name: str, sensor_id: str, calling_time: str, sensing_time: str, temperature: float) -> dict[str, int]:
    """Save one temperature reading.

    calling_time is when the sensor was asked; sensing_time is when the sensor
    measured the value. Both are ISO 8601 times, e.g. 2026-09-26T12:00:00+03:00.
    temperature is in degrees Celsius. The sensor is registered by its id on
    first save; a new name for a known id renames it.
    """
    try:
        return {"sample_id": storage.save_sample(sensor_name, sensor_id, calling_time, sensing_time, temperature)}
    except storage.StorageError as exc:
        # Only a ToolError's message reaches the caller; anything else arrives
        # as a bare "Error executing tool".
        raise ToolError(str(exc)) from None


@server.tool()
def list_sensors() -> list[dict[str, Any]]:
    """List every sensor that has samples saved: its unique id and its name."""
    return storage.list_sensors()


@server.tool()
def list_samples(sensor_id: str, n: int = 10) -> list[dict[str, Any]]:
    """List a sensor's last n saved samples, newest first: calling_time, sensing_time and temperature."""
    try:
        return storage.last_samples(sensor_id, n)
    except storage.StorageError as exc:
        raise ToolError(str(exc)) from None


if __name__ == "__main__":
    server.run(
        transport="streamable-http",
        host=HOST,
        port=PORT,
        json_response=True,
        stateless_http=True,
    )
