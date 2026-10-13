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


def run_git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)


def git(*args: str) -> str:
    proc = run_git(*args)
    if proc.returncode != 0:
        raise ToolError(proc.stderr.strip() or f"git {args[0]} failed")
    return proc.stdout


@server.tool()
def git_branch() -> str:
    """Name of the current git branch, also before the first commit; "detached at <sha>" when detached."""
    proc = run_git("symbolic-ref", "--short", "-q", "HEAD")  # unlike rev-parse, works without commits
    if proc.returncode == 0:
        return proc.stdout.strip()
    return f"detached at {git('rev-parse', '--short', 'HEAD').strip()}"


@server.tool()
def git_diff(path: str | None = None) -> str:
    """All uncommitted changes to tracked files: the staged diff (what the next commit would contain)
    and the unstaged one, each under its own heading. path limits both to one file or directory.
    Untracked files are not included until they are added with git add."""
    pathspec = ["--", path] if path else []  # after "--" so a path can't be read as an option
    sections = []
    for title, extra in (("Staged changes", ["--staged"]), ("Unstaged changes", [])):
        part = git("diff", "--no-color", *extra, *pathspec)
        if part:
            sections.append(f"## {title}\n{part}")
    if not sections:
        return "No changes."
    diff = "\n".join(sections)
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
