from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from calc import CalculationError, evaluate

# One tool, over HTTP: the client POSTs MCP to this server rather than starting
# it as a subprocess, so the calculator is a service of its own - started,
# restarted and watched separately from whatever calls it. The SDK owns the
# protocol, so what is written here is the calculator and nothing else.
server = MCPServer("calc")

# Loopback only. Whatever calls this runs on the same machine, and a calculator
# open to the world is CPU the world can spend. Reaching it from elsewhere is a
# job for a reverse proxy that can authenticate, not for this bind address.
HOST = "127.0.0.1"
PORT = 8765


# Named for the protocol, not for Python: the tool is `eval` to whoever calls
# it, while the function keeps a name that does not shadow the builtin this
# calculator exists to avoid.
@server.tool(name="eval")
def evaluate_expression(expression: str) -> str:
    """Calculate a mathematical expression and return its value.

    Understands + - * / // % ** and parentheses, the constants pi, e and tau,
    and the functions sqrt, log, log2, log10, exp, sin, cos, tan, atan2, floor,
    ceil, hypot, abs, round, min, max and pow. Nothing else: this is a
    calculator, not a Python interpreter.
    """
    try:
        return str(evaluate(expression))
    except CalculationError as exc:
        # A CalculationError says what is wrong with the expression, which is
        # the one thing the caller can act on. The SDK forwards a ToolError's
        # message and treats every other exception as a crash, keeping its text
        # on the server, so the translation has to happen here - otherwise the
        # caller reads "Error executing tool eval" and learns nothing. It stays
        # a failed call either way: a refusal must not arrive looking like an
        # answer.
        raise ToolError(str(exc)) from None


if __name__ == "__main__":
    # Stateless, because a calculation depends on nothing but the expression it
    # was given: there is no session worth remembering between calls, so every
    # request stands on its own and no session table grows with callers.
    #
    # json_response, because an answer is ready the moment it is asked for.
    # The alternative streams it back as server-sent events, which buys nothing
    # for a calculator and costs every client - curl included - an event-stream
    # parser it would otherwise not need.
    server.run(
        transport="streamable-http",
        host=HOST,
        port=PORT,
        json_response=True,
        stateless_http=True,
    )
