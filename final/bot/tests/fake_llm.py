"""Scripted stand-ins for chat completion responses."""

from __future__ import annotations

import json
from types import SimpleNamespace


class FakeCall:
    def __init__(self, call_id: str, name: str, arguments: dict):
        self.id = call_id
        self.type = "function"
        self.function = SimpleNamespace(name="call_tool", arguments=json.dumps({"name": name, "arguments": arguments}))

    def model_dump(self, include=None):
        return {"id": self.id, "type": self.type,
                "function": {"name": self.function.name, "arguments": self.function.arguments}}


def response(content: str | None = None, calls: list[FakeCall] | None = None, prompt=100, completion=10):
    message = SimpleNamespace(content=content, tool_calls=calls or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                           usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion))


class ScriptedLLM:
    """Replaces llm.complete: returns the scripted responses in order and
    records every request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[dict] = []

    async def __call__(self, messages, tools=None, temperature=None):
        self.requests.append({"messages": messages, "tools": tools})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
