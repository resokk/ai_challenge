from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx2

# The Yandex Smart Home user API. It is read with the user's own OAuth token
# (scope iot:view), which the caller supplies with every request - see server.py.
API = "https://api.iot.yandex.net/v1.0"

# What Yandex calls a climate sensor: the device that measures temperature,
# humidity and pressure. That is the "weather sensor" this server is about.
CLIMATE_SENSOR = "devices.types.sensor.climate"
TEMPERATURE = "temperature"


class YandexError(Exception):
    """Something the caller can act on: a bad token, an unknown sensor."""


async def _get(token: str, path: str) -> dict[str, Any]:
    async with httpx2.AsyncClient(timeout=10) as client:
        try:
            response = await client.get(f"{API}{path}", headers={"Authorization": f"Bearer {token}"})
        except httpx2.HTTPError as exc:
            raise YandexError(f"Yandex Smart Home is unreachable ({type(exc).__name__})") from None
    if response.status_code == 401:
        raise YandexError("Yandex rejected the OAuth token: it is invalid, expired or lacks iot:view")
    try:
        body = response.json()
    except ValueError:
        body = {}
    # Errors come back as {"status": "error", "message": ...}, usually with a
    # 4xx code, and the message is the useful part of them.
    if response.status_code != 200 or body.get("status") != "ok":
        message = body.get("message") or response.reason_phrase
        raise YandexError(f"Yandex Smart Home answered {response.status_code}: {message}")
    return body


async def list_weather_sensors(token: str) -> list[dict[str, str | None]]:
    info = await _get(token, "/user/info")
    rooms = {room["id"]: room["name"] for room in info.get("rooms", [])}
    return [
        {"id": device["id"], "name": device["name"], "room": rooms.get(device.get("room"))}
        for device in info.get("devices", [])
        if device.get("type") == CLIMATE_SENSOR
    ]


async def get_temperature(token: str, sensor_id: str) -> dict[str, Any]:
    # Quoted whole: the id is caller input, and "../user/info" must stay an id
    # Yandex does not know rather than become a different request.
    device = await _get(token, f"/devices/{quote(sensor_id, safe='')}")
    for prop in device.get("properties", []):
        if prop.get("parameters", {}).get("instance") != TEMPERATURE:
            continue
        state = prop.get("state") or {}
        if state.get("value") is None:
            raise YandexError(f"Sensor {device['name']!r} has not reported a temperature yet")
        return {
            "sensor": device["name"],
            "temperature": state["value"],
            # "unit.temperature.celsius" -> "celsius"
            "unit": prop["parameters"].get("unit", "").rsplit(".", 1)[-1] or None,
            "updated": prop.get("last_updated"),
        }
    raise YandexError(f"Device {device['name']!r} does not measure temperature")
