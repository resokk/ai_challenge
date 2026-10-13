"""Discovery: enumerate every tracked file of the snapshot and route it to an extractor.

Each file gets a role (main/test/doc/build) and an initial status. Extractors later confirm
`processed` or downgrade a candidate (e.g. a Java file without query/model records becomes
`irrelevant`). Regexes here only select candidates; parsing happens in the extractors.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from camunda_rag.models import FileRole, FileStatus, SourceFile

BINARY_EXT = {
    "png", "jpg", "jpeg", "gif", "ico", "svg", "webp", "woff", "woff2", "ttf", "eot", "otf", "jar", "zip",
    "gz", "tgz", "pdf", "class", "so", "dylib", "bin", "mp4", "webm", "bmp", "keystore", "jks", "p12", "pem",
}
DOC_EXT = {"md", "adoc", "rst", "txt"}
OTHER_CODE_EXT = {"go", "ts", "tsx", "js", "mjs", "cjs", "py", "sh", "kt", "groovy", "scala"}

SQL_IN_TEXT = re.compile(
    r"""["'`]\s*(SELECT\s[^"'`]*\sFROM|INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM|CREATE\s+TABLE|ALTER\s+TABLE)\b"""
)
JAVA_CANDIDATE = re.compile(
    r'("""|"\s*(?:\(\s*)?(?:SELECT|INSERT\s+INTO|UPDATE|DELETE\s+FROM|MERGE\s+INTO|CREATE\s+(?:UNIQUE\s+)?(?:TABLE|INDEX|VIEW)|ALTER\s+TABLE|DROP\s+(?:TABLE|INDEX)|TRUNCATE|WITH)\b'
    r"|@(?:Select|Insert|Update|Delete|Query|NamedQuery|NamedNativeQuery)\s*\(|createQuery\s*\(|createNativeQuery\s*\()"
)
JAVA_MODEL_NAME = re.compile(r"(DbModel|DbQuery|Entity|Dto|ColumnFamilies)\.java$")
DOC_TERMS = re.compile(
    r"\b(table|tables|column|columns|schema|index|indices|mapping|mappings|database|rdbms|liquibase|mybatis|"
    r"elasticsearch|opensearch|rocksdb|column famil|sql|exporter)\b",
    re.I,
)


@dataclass
class DiscoveredFile:
    path: str
    ext: str
    role: FileRole
    kind: str  # liquibase | mybatis | sql | vendor_properties | search_mapping | java | doc | other
    status: FileStatus
    reason: str = ""
    size: int = 0
    extractors: list[str] = field(default_factory=list)

    def to_model(self) -> SourceFile:
        return SourceFile(
            path=self.path, language=self.ext or "none", role=self.role, status=self.status,
            reason=self.reason, extractors=self.extractors, size=self.size,
        )


@dataclass
class Inventory:
    commit_sha: str
    files: list[DiscoveredFile]

    def by_kind(self, kind: str) -> list[DiscoveredFile]:
        return [f for f in self.files if f.kind == kind]

    def get(self, path: str) -> DiscoveredFile | None:
        return self._index.get(path)

    def __post_init__(self) -> None:
        self._index = {f.path: f for f in self.files}


def git_commit(repo: Path) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()


def tracked_files(repo: Path) -> list[str]:
    out = subprocess.run(["git", "-C", str(repo), "ls-files", "-z"], check=True, capture_output=True).stdout
    return [p for p in out.decode("utf-8", "surrogateescape").split("\0") if p]


def file_role(path: str) -> FileRole:
    p = "/" + path.lower()
    name = p.rsplit("/", 1)[-1]
    if (
        "/src/test/" in p or "/test/" in p or "/tests/" in p or "/qa/" in p or "/testing/" in p
        or "/e2e" in p or "/load-tests/" in p or "/microbenchmarks/" in p or "/it/" in p
        or re.search(r"(test|tests|it|spec)\.(java|ts|tsx|js|go|py)$", name)
        or name.endswith("_test.go") or "/fixtures/" in p or "/__snapshots__/" in p
    ):
        return FileRole.test
    ext = name.rsplit(".", 1)[-1] if "." in name else ""
    if ext in DOC_EXT or "/docs/" in p or "/monorepo-docs-site/" in p:
        return FileRole.doc
    if name in {"pom.xml", "makefile", "dockerfile"} or name.endswith((".dockerfile", ".gradle")) or "/.github/" in p \
            or "/.ci/" in p or "/build-tools/" in p:
        return FileRole.build
    return FileRole.main


def _read_head(path: Path, n: int = 8192) -> bytes:
    with path.open("rb") as fh:
        return fh.read(n)


def _classify(repo: Path, rel: str, max_bytes: int) -> DiscoveredFile:
    abs_path = repo / rel
    name = rel.rsplit("/", 1)[-1]
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    role = file_role(rel)
    df = DiscoveredFile(path=rel, ext=ext, role=role, kind="other", status=FileStatus.irrelevant)
    if not abs_path.is_file():
        df.status, df.reason = FileStatus.excluded, "not present in working tree (symlink or submodule)"
        return df
    df.size = abs_path.stat().st_size
    if ext in BINARY_EXT:
        df.status, df.reason = FileStatus.excluded, f"binary format .{ext}"
        return df
    head = _read_head(abs_path)
    if b"\0" in head:
        df.status, df.reason = FileStatus.excluded, "binary content"
        return df
    if df.size > max_bytes:
        df.status, df.reason = FileStatus.excluded, f"larger than {max_bytes} bytes"
        return df
    text_head = head.decode("utf-8", "replace")

    if ext in {"xml", "yaml", "yml", "json"} and "databaseChangeLog" in text_head:
        df.kind, df.status = "liquibase", FileStatus.processed
    elif ext == "xml" and ("mybatis.org//DTD Mapper" in text_head or re.search(r"<mapper\s+namespace=", text_head)):
        df.kind, df.status = "mybatis", FileStatus.processed
    elif ext == "sql":
        df.kind, df.status = "sql", FileStatus.processed
    elif ext == "properties" and "/vendor-properties/" in rel:
        df.kind, df.status = "vendor_properties", FileStatus.processed
    elif ext == "json" and re.search(r'"(mappings|index_patterns|template)"\s*:', text_head):
        df.kind, df.status = "search_mapping", FileStatus.processed
    elif ext == "java":
        text = abs_path.read_text("utf-8", "replace")
        if JAVA_CANDIDATE.search(text) or JAVA_MODEL_NAME.search(name) or re.search(r"^\s*public\s+interface\s+\w+Mapper\b", text, re.M):
            df.kind, df.status = "java", FileStatus.processed  # confirmed by the analyzer
        else:
            df.reason = "java: no query literal, query annotation or model type candidate"
    elif ext in DOC_EXT:
        text = abs_path.read_text("utf-8", "replace")
        if len(DOC_TERMS.findall(text)) >= 3:
            df.kind, df.status = "doc", FileStatus.processed
        else:
            df.reason = "documentation without data-model terms"
    elif ext in OTHER_CODE_EXT:
        text = abs_path.read_text("utf-8", "replace")
        if SQL_IN_TEXT.search(text):
            df.status = FileStatus.unsupported
            df.reason = f".{ext}: SQL-like string candidate found; no extractor for this language"
        else:
            df.reason = f".{ext}: no SQL/DSL candidate by prefilter"
    else:
        df.reason = f"no data-model extractor applies to .{ext or name}"
    return df


def discover(repo: Path, max_bytes: int) -> Inventory:
    sha = git_commit(repo)
    files = [_classify(repo, rel, max_bytes) for rel in tracked_files(repo)]
    return Inventory(commit_sha=sha, files=files)
