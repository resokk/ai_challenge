"""Git MCP server over stdio: python git_server.py [repo]

Read-only access to the repository the chat runs in (or repo, if given):
the current branch and the working tree diff. The chat launches it itself.
"""

from __future__ import annotations

import subprocess
import sys

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

MAX_DIFF_CHARS = 50_000  # keeps a huge diff from flooding the terminal and the model's context

# WARNING: tool errors are logged at INFO to stderr, which the chat shares; it reports them itself.
server = MCPServer("git", instructions="Read-only view of the project's git repository: current branch and diff.",
                   log_level="WARNING")
repo = "."  # set by main()


def git(*args: str) -> str:
    proc = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise ToolError(proc.stderr.strip() or f"git {args[0]} failed")
    return proc.stdout


@server.tool()
def git_branch() -> str:
    """Name of the current git branch (or "HEAD" when detached)."""
    return git("rev-parse", "--abbrev-ref", "HEAD").strip()


@server.tool()
def git_diff(staged: bool = False, path: str | None = None) -> str:
    """Diff of uncommitted changes. staged=true shows what is staged for commit instead of
    unstaged changes; path limits the diff to one file or directory."""
    args = ["diff", "--no-color"]
    if staged:
        args.append("--staged")
    if path:
        args += ["--", path]  # after "--" so a path can't be read as an option
    diff = git(*args)
    if not diff:
        return "No changes."
    if len(diff) > MAX_DIFF_CHARS:
        return diff[:MAX_DIFF_CHARS] + f"\n... [truncated, {len(diff)} chars total]"
    return diff


def main() -> None:
    global repo
    if len(sys.argv) > 1:
        repo = sys.argv[1]
    server.run()  # stdio


if __name__ == "__main__":
    main()
