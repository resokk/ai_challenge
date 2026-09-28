# codeindex

Local semantic code search. Chunks a source tree, embeds the chunks with an Ollama
embedding model, stores the vectors in FAISS and the chunk metadata in SQLite. Each
chunking mode has its own index in `index/<mode>/` (`meta.db` + `vectors.faiss`).

## Setup

Python 3.11+ and [Ollama](https://ollama.com):

```sh
pip install -r requirements.txt
pip install tree-sitter tree-sitter-language-pack   # only for --chunking treesitter
ollama serve
ollama pull nomic-embed-text      # embeddings (default; override with --model or OLLAMA_EMBED_MODEL)
ollama pull qwen2.5-coder:1.5b    # only for --chunking summary
```

`OLLAMA_HOST` defaults to `http://localhost:11434`.

## Usage

```sh
python -m codeindex index ../day_18                   # build or update index/ast
python -m codeindex search "read sensor temperature"  # -k 10 by default
python -m codeindex stats
```

Every command takes `--chunking <mode>` (default `ast`). Re-running `index` only re-embeds
files whose content changed and drops chunks of deleted files. If the mode, embedding model
or repo root differ from the ones the index was built with, `index` stops; add `--rebuild`.

`--db <name>` picks the SQLite file inside the index folder (default `meta.db`); its vectors
go beside it as `<name>.faiss`, so several indexes of one mode can share a folder:

```sh
python -m codeindex index ormar/docs --chunking tokens --db ormar_docs.db
python -m codeindex search "foreign key" --chunking tokens --db ormar_docs.db
```

Search options: `--path-prefix`, `--kind {file,module,class,function,method,block}`,
`--class`, `--json`.

## Chunking modes

| mode | chunks |
|---|---|
| `ast` | one per function/method and class, plus module-level code (Python `ast`) |
| `lines` | 60-line windows, 10 lines of overlap |
| `treesitter` | like `ast`, for many languages |
| `recursive` | `ast` units, those over 6000 chars split into `part` 1..n |
| `tokens` | ~512-token windows, 64 tokens of overlap, snapped to lines |
| `contextual` | `recursive`, embedding the file's imports, enclosing class and called names too |
| `multi` | file, class and function/method levels together |
| `summary` | `ast` units, embedding an LLM summary with the code |

Modes that need a Python AST chunk other files as `lines`; `treesitter` does the same for
languages it doesn't know.

```sh
python -m codeindex index ../day_18 --chunking ast
python -m codeindex search "list climate sensors" --chunking ast --class YandexClient

python -m codeindex index ../day_18 --chunking lines
python -m codeindex search "oauth bearer header" --chunking lines

python -m codeindex index ../day_18 --chunking treesitter
python -m codeindex search "sqlite schema for samples" --chunking treesitter --kind function

python -m codeindex index ../day_18 --chunking recursive
python -m codeindex search "parse device properties" --chunking recursive   # parts merge into one result

python -m codeindex index ../day_18 --chunking tokens
python -m codeindex search "mcp tool definition" --chunking tokens -k 5

python -m codeindex index ../day_18 --chunking contextual
python -m codeindex search "http request to yandex api" --chunking contextual

python -m codeindex index ../day_18 --chunking multi
python -m codeindex search "store temperature samples" --chunking multi --kind method

python -m codeindex index ../day_18 --chunking summary --summary-model qwen2.5-coder:1.5b
python -m codeindex search "which sensors report temperature" --chunking summary --json
```

`summary` makes one LLM call per unit, so the first build is slow. Summaries are cached in
`meta.db` by content hash, so re-indexing (even with `--rebuild`) only summarizes changed code.

## Tests

The tests use a fake embedder and summarizer, so they don't need Ollama:

```sh
python -m unittest
```
