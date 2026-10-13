"""Application settings: defaults < config.toml < environment (CAMUNDA_RAG_*)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import os

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict, TomlConfigSettingsSource

PROJECT_DIR = Path(__file__).resolve().parents[2]

# Camunda databaseId / vendor-properties name -> SQLGlot dialect. H2 has no SQLGlot dialect;
# it is parsed with the generic dialect and every result carries a diagnostic saying so.
SQLGLOT_DIALECTS: dict[str, str] = {
    "postgresql": "postgres",
    "mysql": "mysql",
    "mariadb": "mysql",
    "oracle": "oracle",
    "mssql": "tsql",
    "h2": "",
}


class EmbeddingSettings(BaseModel):
    model_name: str = "BAAI/bge-m3"
    model_path: Path | None = None  # local snapshot dir; when set nothing is downloaded
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    batch_size: int = 8
    max_tokens: int = 1024  # chunk budget measured with the model tokenizer
    offline: bool = False  # forbid any hub access (HF_HUB_OFFLINE)


class Limits(BaseModel):
    mybatis_max_variants: int = 8  # per statement and dialect
    mybatis_foreach_iterations: int = 1
    mybatis_max_include_depth: int = 10
    java_max_depth: int = 8
    java_max_variants: int = 8
    java_max_loop_unroll: int = 1
    max_file_bytes: int = 2_000_000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CAMUNDA_RAG_", env_nested_delimiter="__", toml_file=PROJECT_DIR / "config.toml"
    )

    repo_path: Path = Path.home() / "src" / "camunda"
    repository_url: str = "https://github.com/camunda/camunda.git"
    data_dir: Path = PROJECT_DIR / "data"
    java_analyzer_jar: Path = PROJECT_DIR / "java-analyzer" / "target" / "java-analyzer.jar"
    maven_cmd: str = "mvn"
    java_cmd: str = "java"
    dialects: list[str] = Field(default_factory=lambda: list(SQLGLOT_DIALECTS))
    default_dialect: str = "postgresql"
    # Liquibase changelog parameters for the reconstructed profile (${prefix} etc.)
    table_prefix: str = ""
    liquibase_contexts: list[str] = Field(default_factory=list)  # empty = no filter (Liquibase semantics)
    liquibase_labels: list[str] = Field(default_factory=list)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    limits: Limits = Field(default_factory=Limits)
    host: str = "127.0.0.1"
    port: int = 8765

    @model_validator(mode="after")
    def _resolve_paths(self) -> "Settings":
        def resolve(p: Path) -> Path:
            p = p.expanduser()
            return p if p.is_absolute() else PROJECT_DIR / p

        self.repo_path = resolve(self.repo_path)
        self.data_dir = resolve(self.data_dir)
        self.java_analyzer_jar = resolve(self.java_analyzer_jar)
        self.maven_cmd = os.path.expanduser(self.maven_cmd)
        if self.embedding.model_path is not None:
            self.embedding.model_path = resolve(self.embedding.model_path)
        return self

    @classmethod
    def settings_customise_sources(
        cls, settings_cls: type[BaseSettings], init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource, dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return init_settings, env_settings, TomlConfigSettingsSource(settings_cls)


def sqlglot_dialect(dialect: str) -> str:
    try:
        return SQLGLOT_DIALECTS[dialect]
    except KeyError as e:
        raise ValueError(f"unknown dialect {dialect!r}; known: {sorted(SQLGLOT_DIALECTS)}") from e
