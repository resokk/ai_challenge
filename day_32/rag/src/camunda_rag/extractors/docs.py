"""Documentation sections split at heading boundaries (Markdown/AsciiDoc/RST) or paragraphs."""

from __future__ import annotations

import re
from pathlib import Path

from camunda_rag.models import DocSection, FileRole

HEADING = {
    "md": re.compile(r"^(#{1,6})\s+(.*)$"),
    "adoc": re.compile(r"^(={1,6})\s+(.*)$"),
}


def sections(repo: Path, rel: str, role: FileRole) -> list[DocSection]:
    lines = (repo / rel).read_text("utf-8", "replace").splitlines()
    ext = rel.rsplit(".", 1)[-1].lower()
    rx = HEADING.get(ext)
    out: list[DocSection] = []
    heading, start, buf = rel.rsplit("/", 1)[-1], 1, []
    in_code = False

    def flush(end: int) -> None:
        text = "\n".join(buf).strip()
        if text:
            out.append(DocSection(path=rel, heading=heading, text=text, line_start=start, line_end=end, role=role))

    for i, line in enumerate(lines, 1):
        if line.strip().startswith("```"):
            in_code = not in_code
        m = rx.match(line) if rx and not in_code else None
        if m:
            flush(i - 1)
            heading, start, buf = m.group(2).strip(), i, [line]
        else:
            buf.append(line)
    flush(len(lines))
    return out
