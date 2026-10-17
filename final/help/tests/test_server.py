"""The MCP server: one `help` tool, its schema, calls and argument errors."""

from __future__ import annotations

import pytest
from mcp.server.mcpserver.exceptions import ToolError

import answer
import server


@pytest.fixture
def answers(monkeypatch):
    questions = []

    async def fake(question):
        questions.append(question)
        return answer.check_question(question) and f"ответ на: {question}\n\nСтатьи: Коммит (commit)"

    monkeypatch.setattr(answer, "answer", fake)
    return questions


async def test_exactly_one_tool_with_question_schema():
    [tool] = await server.mcp.list_tools()
    assert tool.name == "help"
    assert "как в этом боте сделать X" in tool.description
    schema = tool.input_schema
    assert schema["type"] == "object"
    assert set(schema["properties"]) == {"question"}
    assert schema["properties"]["question"]["type"] == "string"
    assert schema["required"] == ["question"]


async def test_tool_function_called_directly(answers):
    assert (await server.help_tool("как закоммитить")).startswith("ответ на: как закоммитить")
    assert answers == ["как закоммитить"]


async def test_call_through_server(answers):
    result = await server.mcp.call_tool("help", {"question": "как закоммитить"})
    assert not result.is_error
    assert result.content[0].text.endswith("Статьи: Коммит (commit)")


@pytest.mark.parametrize("arguments", [{}, {"question": 5}, {"query": "x"}])
async def test_bad_arguments_raise_tool_error(answers, arguments):
    with pytest.raises(ToolError):
        await server.mcp.call_tool("help", arguments)
    assert answers == []


@pytest.mark.parametrize("question, message", [("  ", "пустой"), ("x" * 1001, "слишком длинный")])
async def test_bad_question_is_a_tool_error(answers, question, message):
    with pytest.raises(ToolError, match=message):
        await server.help_tool(question)


def test_server_refuses_to_start_without_key(monkeypatch, capsys):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(SystemExit) as caught:
        server.main()
    assert caught.value.code == 1
    assert "DEEPSEEK_API_KEY" in capsys.readouterr().err
