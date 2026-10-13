"""Console chat with a DeepSeek model: python chat.py

Keeps the dialogue history for the session and streams replies as they arrive.
Every tool of every MCP server in SERVERS becomes a command /<tool> at startup;
/mcp lists them. The model can run the same commands: they are described after the
system prompt, and a reply line "/<command> {json arguments}" runs one. Both the call
and its result are shown, and the result goes back to whoever asked: a command typed
by the user prints for the user only, one requested by the model is sent back to it.
/help <question> answers from the project docs found by the RAG server's search tool,
/review [path] has the model review the uncommitted git diff, checking the tables it
touches against the RAG data model and its SQL with validate_sql,
/reset clears the history, /exit (or Ctrl+D) quits.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
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
BUILTIN_COMMANDS = {"help", "review", "mcp", "reset", "exit"}

RAG_TOP_K = 8
SNIPPET_CHARS = 1500  # per hit, so a few long entities don't crowd out the rest
SYSTEM_PROMPT = "You are a helpful assistant for a software developer. Answer concisely."
COMMANDS_PROMPT = """You can run the commands below to look into the project. To run one, write a line
containing only the command and its arguments as one JSON object on that same line, e.g.
/git_diff {{"path": "src"}}
then stop and wait: its result comes back in the next message, and you continue from there.
Several command lines in one reply run in order. Run a command only when you need its result.

Commands:
{commands}"""
COMMAND_LINE = re.compile(r"^/([\w-]+)(?:\s+(\{.*\}))?$")
MAX_COMMAND_ROUNDS = 5  # model -> command -> model rounds per user message
REVIEW_COMMAND_ROUNDS = 8  # a review checks every table and SQL statement, possibly in several rounds
REVIEW_PROMPT = """Review the uncommitted changes in the git diff below as a senior engineer.
The diff (staged and unstaged) is complete: do not fetch it again.

Before writing the review, do these checks with the commands. They are mandatory:
1. Tables. For every database table, index or column family the diff creates, reads, writes or names
   (in SQL, MyBatis or ORM mappings, migrations, entity annotations, config), run
   /describe {{"name": "<TABLE>"}} to confirm it exists in the project's data model; if the name is
   uncertain, find it with /search first. Check the columns the diff uses against the describe result.
   A table or column that does not exist is a problem unless the diff itself creates it.
2. SQL. For every SQL statement the diff adds or changes, run /validate_sql {{"sql": "<statement>"}}
   with the statement written out in full; replace template placeholders (#{{x}}, ${{x}}) with ? and
   pass "dialect" when the file implies one. Every issue it reports with severity error is a problem.
Put all the checks of a step into one reply. If the diff has no tables or no SQL, skip that step.
If a command is unavailable or fails, say in the review that the check could not be done: never
report a check you did not run.

Then write the review in Russian, in exactly this form:

## Рецензия
What the change does, the overall verdict (approve / approve with remarks / request changes), and
which tables and SQL statements were checked, with the outcome of each check.

## Проблемы
A numbered list, most severe first. Each item: severity (critical | major | minor), file:line from
the diff, what is wrong and why, and a suggested fix. Look for correctness bugs, failed table and
SQL checks, security issues, missing error handling and maintainability problems. Report only
problems grounded in the diff or the command results. If there are none, write "Проблем не найдено."

Diff:
```diff
{diff}
```"""
MAX_RESULT_CHARS = 20_000  # of a command result sent back to the model
SHOWN_RESULT_CHARS = 1_500  # of a model-requested result echoed to the terminal
HELP_PROMPT = """Answer the question about the project using only the context below, found by searching
the project's docs and data model. Cite the sources you use as [n]. If the context does not
answer the question, say so instead of guessing.

Context:
{context}

Question: {question}"""


# Terminal colors: user input green, tool calls and results gray, model replies yellow.
# Off when output is not a terminal or NO_COLOR is set (https://no-color.org).
USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ
USER, TOOL, LLM = "32", "90", "33"


def paint(text: str, color: str) -> str:
    return f"\033[{color}m{text}\033[0m" if USE_COLOR else text


def set_color(color: str | None) -> None:
    """Switch the color of what is printed (or typed) next; None resets it."""
    if USE_COLOR:
        print(f"\033[{color or 0}m", end="", flush=True)


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
    Non-string values are parsed as JSON: k=5, include_tests=true, doc_types='["entity"]'."""
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


def describe_commands(commands: dict[str, Command]) -> str:
    """Every command with its full description and parameters, for the system prompt."""
    blocks = []
    for name, command in commands.items():
        schema = command.tool.input_schema
        required = set(schema.get("required", []))
        lines = [f"/{name}", inspect.cleandoc(command.tool.description or "")]
        for key, prop in schema.get("properties", {}).items():
            types = "|".join(sorted(param_types(prop) - {"null"})) or "any"
            spec = "required" if key in required else f"optional, default {json.dumps(prop.get('default'))}"
            note = f" - {prop['description']}" if prop.get("description") else ""
            lines.append(f"  {key}: {types} ({spec}){note}")
        if not schema.get("properties"):
            lines.append("  no arguments")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def command_calls(reply: str, commands: dict[str, Command]) -> list[tuple[str, str]]:
    """The command lines of a model reply as (name, JSON arguments text)."""
    calls = []
    for line in reply.splitlines():
        match = COMMAND_LINE.match(line.strip().strip("`").strip())
        if match and match[1] in commands:
            calls.append((match[1], match[2] or "{}"))
    return calls


def run_model_command(commands: dict[str, Command], name: str, raw_arguments: str) -> str:
    """Run a command the model asked for; errors are returned as text so the model can react."""
    print(paint(f"\n[model -> /{name} {raw_arguments}]", TOOL))
    try:
        arguments = json.loads(raw_arguments)
        if not isinstance(arguments, dict):
            raise ValueError("arguments must be a JSON object")
        command = commands[name]
        result = result_text(asyncio.run(call_tool(command.server, command.tool.name, arguments)))
    except (ValueError, McpCallError) as e:  # JSONDecodeError is a ValueError
        result = f"Error: {e}"
    shown = result if len(result) <= SHOWN_RESULT_CHARS else result[:SHOWN_RESULT_CHARS] + f"\n... [{len(result)} chars]"
    print(paint(f"[/{name} -> model]\n{shown}", TOOL))
    if len(result) > MAX_RESULT_CHARS:
        result = result[:MAX_RESULT_CHARS] + f"\n... [truncated, {len(result)} chars total]"
    return f"Result of /{name} {raw_arguments}:\n{result}"


def answer(client: OpenAI, messages: list[dict], commands: dict[str, Command],
           max_rounds: int = MAX_COMMAND_ROUNDS) -> None:
    """Get the model's reply to the last message, running the commands it asks for and
    feeding their results back until it answers without one."""
    for _ in range(max_rounds):
        reply = stream_reply(client, messages)
        messages.append({"role": "assistant", "content": reply})
        calls = command_calls(reply, commands)
        if not calls:
            return
        results = [run_model_command(commands, name, arguments) for name, arguments in calls]
        header = "Results of the commands you ran (sent by the chat, not typed by the user):\n\n"
        messages.append({"role": "user", "content": header + "\n\n".join(results)})
        print()
    print(f"[stopped: more than {max_rounds} command rounds]")


def review_message(path: str | None) -> str | None:
    """The review prompt over the uncommitted diff, or None when there are no changes."""
    diff = result_text(asyncio.run(call_tool("git", "git_diff", {"path": path} if path else {})))
    return None if diff == "No changes." else REVIEW_PROMPT.format(diff=diff)


def print_mcp(commands: dict[str, Command]) -> None:
    if not commands:
        print("No MCP commands available.")
        return
    for server in SERVERS:
        own = [(n, c) for n, c in commands.items() if c.server == server]
        print(paint(f"\n{server}:" if own else f"\n{server}: unavailable", TOOL))
        for name, command in own:
            description = (command.tool.description or "").strip().splitlines()
            print(paint(f"  {usage(name, command.tool)}", TOOL))
            if description:
                print(paint(f"      {description[0].strip()}", TOOL))


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
    set_color(LLM)
    try:
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                print(delta, end="", flush=True)
                parts.append(delta)
    finally:
        set_color(None)  # an error or Ctrl+C mid-stream must not leave the terminal yellow
    print()
    return "".join(parts)


def main() -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        sys.exit("DEEPSEEK_API_KEY is not set (put it in .env)")
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    commands = discover_commands()
    system_prompt = SYSTEM_PROMPT
    if commands:
        system_prompt += "\n\n" + COMMANDS_PROMPT.format(commands=describe_commands(commands))
    messages = [{"role": "system", "content": system_prompt}]

    print(f"Chat with {DEEPSEEK_MODEL}. /help <question> asks about the project, /review [path] reviews "
          "uncommitted changes, /mcp lists MCP commands, /reset clears history, /exit quits.")
    while True:
        try:
            text = input("\n> " + (f"\033[{USER}m" if USE_COLOR else "")).strip()
        except (EOFError, KeyboardInterrupt):
            set_color(None)
            print()
            break
        set_color(None)
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
                print(paint(result_text(asyncio.run(call_tool(command.server, command.tool.name, arguments))), TOOL))
            except ValueError as e:
                print(f"Error: {e}\nUsage: {usage(name[1:], command.tool)}", file=sys.stderr)
            except McpCallError as e:
                print(f"Error: {e}", file=sys.stderr)
            continue

        hits = []
        rounds = MAX_COMMAND_ROUNDS
        if name == "/review":
            try:
                review = review_message(rest.strip() or None)
            except McpCallError as e:
                print(f"Error: {e}", file=sys.stderr)
                continue
            if review is None:
                print("No uncommitted changes to review.")
                continue
            text, rounds = review, REVIEW_COMMAND_ROUNDS
        elif name == "/help":
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

        turn_start = len(messages)
        messages.append({"role": "user", "content": text})
        try:
            answer(client, messages, commands, rounds)
        except OpenAIError as e:
            del messages[turn_start:]  # keep history consistent: no answer, no question
            print(f"Error: {e}", file=sys.stderr)
            continue
        except KeyboardInterrupt:
            del messages[turn_start:]
            print("\n[interrupted]")
            continue
        if hits:
            print(paint("\nSources:", LLM))
            for i, h in enumerate(hits, 1):
                print(paint(f"  [{i}] {source_of(h)}", LLM))


if __name__ == "__main__":
    main()
