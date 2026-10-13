"""Console chat with a DeepSeek model: python chat.py

Keeps the dialogue history for the session and streams replies as they arrive.
Every tool of every MCP server in SERVERS becomes a command /<tool> at startup;
/mcp lists them. /help <question> answers from the project docs found by the RAG
server's search tool, /reset clears the history, /exit (or Ctrl+D) quits.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp import Client, StdioServerParameters
from mcp.types import CallToolResult, Tool
from openai import OpenAI, OpenAIError

load_dotenv()

DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
RAG_MCP_URL = os.environ.get("RAG_MCP_URL", "http://127.0.0.1:8765/mcp")

# MCP servers by name: a URL (streamable HTTP) or a command the chat launches (stdio).
SERVERS: dict[str, str | StdioServerParameters] = {
    "rag": RAG_MCP_URL,
    "git": StdioServerParameters(
        command=sys.executable, args=[str(Path(__file__).with_name("git_server.py")), os.getcwd()]
    ),
}
BUILTIN_COMMANDS = {"help", "mcp", "reset", "exit"}

RAG_TOP_K = 8
SNIPPET_CHARS = 1500  # per hit, so a few long entities don't crowd out the rest
SYSTEM_PROMPT = "You are a helpful assistant for a software developer. Answer concisely."
HELP_PROMPT = """Answer the question about the project using only the context below, found by searching
the project's docs and data model. Cite the sources you use as [n]. If the context does not
answer the question, say so instead of guessing.

Context:
{context}

Question: {question}"""


class McpCallError(Exception):
    pass


@dataclass
class Command:
    server: str
    tool: Tool


def root_cause(e: BaseException) -> BaseException:
    while isinstance(e, BaseExceptionGroup):  # the client's task group wraps the real error
        e = e.exceptions[0]
    return e


async def list_tools(server: str) -> list[Tool]:
    try:
        async with Client(SERVERS[server]) as client:
            return (await client.list_tools()).tools
    except Exception as e:
        raise McpCallError(f"MCP server {server!r} is unavailable: {root_cause(e)!r}") from e


async def call_tool(server: str, name: str, arguments: dict[str, Any]) -> CallToolResult:
    try:
        async with Client(SERVERS[server]) as client:
            result = await client.call_tool(name, arguments)
    except Exception as e:  # connection and protocol errors surface as assorted types
        raise McpCallError(f"MCP server {server!r} is unavailable: {root_cause(e)!r}") from e
    if result.is_error:
        raise McpCallError(f"{name} failed: {result_text(result)}")
    return result


def result_text(result: CallToolResult) -> str:
    text = "\n".join(c.text for c in result.content if getattr(c, "text", None))
    if not text and result.structured_content is not None:
        text = json.dumps(result.structured_content, ensure_ascii=False, indent=2)
    return text


def discover_commands() -> dict[str, Command]:
    """Register /<tool> for every tool of every reachable server. A name already taken
    by a built-in or another server's tool becomes /<server>_<tool>."""
    commands: dict[str, Command] = {}
    for server in SERVERS:
        try:
            tools = asyncio.run(list_tools(server))
        except McpCallError as e:
            print(f"Warning: {e}", file=sys.stderr)
            continue
        for tool in tools:
            name = tool.name
            if name in BUILTIN_COMMANDS or name in commands:
                name = f"{server}_{tool.name}"
            commands[name] = Command(server, tool)
    return commands


def param_types(schema: dict) -> set[str]:
    options = schema.get("anyOf", [schema])
    return {o["type"] for o in options if "type" in o}


def usage(name: str, tool: Tool) -> str:
    props = tool.input_schema.get("properties", {})
    required = set(tool.input_schema.get("required", []))
    parts = []
    for key, schema in props.items():
        types = "|".join(sorted(param_types(schema) - {"null"}))
        parts.append(f"{key}={types}" if key in required else f"[{key}={types}]")
    return " ".join([f"/{name}", *parts])


def parse_arguments(tool: Tool, line: str) -> dict[str, Any]:
    """Turn "free text key=value ..." into tool arguments. key=value sets a parameter;
    remaining words go to the first required string parameter (or the first string one).
    Non-string values are parsed as JSON: k=5, staged=true, doc_types='["entity"]'."""
    props = tool.input_schema.get("properties", {})
    required = tool.input_schema.get("required", [])
    try:
        tokens = shlex.split(line)
    except ValueError as e:
        raise ValueError(f"bad quoting: {e}") from e

    arguments: dict[str, Any] = {}
    words = []
    for token in tokens:
        key, sep, value = token.partition("=")
        if sep and key in props:
            arguments[key] = value
        else:
            words.append(token)
    if words:
        strings = [k for k in [*required, *props] if "string" in param_types(props[k])]
        target = next((k for k in strings if k not in arguments), None)
        if target is None:
            raise ValueError(f"unexpected arguments: {' '.join(words)}")
        arguments[target] = " ".join(words)

    for key, value in arguments.items():
        if isinstance(value, str) and "string" not in param_types(props[key]):
            try:
                arguments[key] = json.loads(value)
            except json.JSONDecodeError as e:
                raise ValueError(f"{key}: {value!r} is not valid JSON") from e
    missing = [k for k in required if k not in arguments]
    if missing:
        raise ValueError(f"missing {', '.join(missing)}")
    return arguments


def print_mcp(commands: dict[str, Command]) -> None:
    if not commands:
        print("No MCP commands available.")
        return
    for server in SERVERS:
        own = [(n, c) for n, c in commands.items() if c.server == server]
        print(f"\n{server}:" if own else f"\n{server}: unavailable")
        for name, command in own:
            description = (command.tool.description or "").strip().splitlines()
            print(f"  {usage(name, command.tool)}")
            if description:
                print(f"      {description[0].strip()}")


def source_of(hit: dict) -> str:
    prov = hit.get("provenance") or {}
    path = prov.get("path")
    if not path:
        return hit["title"]
    return f"{hit['title']} ({path}:{prov.get('line_start')}-{prov.get('line_end')})"


def help_message(question: str, hits: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i}] {source_of(h)}\n{h.get('snippet', '')[:SNIPPET_CHARS]}" for i, h in enumerate(hits, 1)
    )
    return HELP_PROMPT.format(context=context, question=question)


def stream_reply(client: OpenAI, messages: list[dict]) -> str:
    """Print the model's reply as it streams in and return it whole."""
    stream = client.chat.completions.create(model=DEEPSEEK_MODEL, messages=messages, stream=True)
    parts = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            print(delta, end="", flush=True)
            parts.append(delta)
    print()
    return "".join(parts)


def main() -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        sys.exit("DEEPSEEK_API_KEY is not set (put it in .env)")
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    commands = discover_commands()

    print(f"Chat with {DEEPSEEK_MODEL}. /help <question> asks about the project, /mcp lists MCP commands, "
          "/reset clears history, /exit quits.")
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        name, _, rest = text.partition(" ")
        if name == "/exit":
            break
        if name == "/reset":
            del messages[1:]
            print("History cleared.")
            continue
        if name == "/mcp":
            print_mcp(commands)
            continue
        if name.startswith("/") and name[1:] in commands:
            command = commands[name[1:]]
            try:
                arguments = parse_arguments(command.tool, rest)
                print(result_text(asyncio.run(call_tool(command.server, command.tool.name, arguments))))
            except ValueError as e:
                print(f"Error: {e}\nUsage: {usage(name[1:], command.tool)}", file=sys.stderr)
            except McpCallError as e:
                print(f"Error: {e}", file=sys.stderr)
            continue

        hits = []
        if name == "/help":
            question = rest.strip()
            if not question:
                print("Usage: /help <question about the project>")
                continue
            try:
                result = asyncio.run(call_tool("rag", "search", {"query": question, "k": RAG_TOP_K}))
            except McpCallError as e:
                print(f"Error: {e}", file=sys.stderr)
                continue
            hits = (result.structured_content or {}).get("hits", [])
            if not hits:
                print("Nothing found in the project docs.")
                continue
            text = help_message(question, hits)
        elif name.startswith("/"):
            print(f"Unknown command {name}. /mcp lists MCP commands.")
            continue

        messages.append({"role": "user", "content": text})
        try:
            reply = stream_reply(client, messages)
        except OpenAIError as e:
            messages.pop()  # keep history consistent: no reply, no question
            print(f"Error: {e}", file=sys.stderr)
            continue
        except KeyboardInterrupt:
            messages.pop()
            print("\n[interrupted]")
            continue
        messages.append({"role": "assistant", "content": reply})
        if hits:
            print("\nSources:")
            for i, h in enumerate(hits, 1):
                print(f"  [{i}] {source_of(h)}")


if __name__ == "__main__":
    main()
