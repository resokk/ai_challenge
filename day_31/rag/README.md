# camunda-rag — модель данных Camunda 8 по исходникам

Индексирует репозиторий [camunda/camunda](https://github.com/camunda/camunda) и даёт:

* CLI-индексатор всего репозитория (`camunda-rag index`);
* HTTP MCP-сервер (streamable HTTP, `/mcp`) с инструментами `search`, `describe`, `queries`;
* HTTP API статической валидации SQL на SQLGlot (`POST /validate`);
* локальные эмбеддинги BAAI/bge-m3, плотные векторы в FAISS, структурированный каталог и FTS5 в SQLite.

Внешние LLM и облачные embedding API не используются. BGE-M3 нужна только для векторов поиска,
всё извлечение структурное (JavaParser, lxml/YAML, SQLGlot).

## Быстрый старт

```bash
# зависимости (Python 3.11+, Java 21+, Maven 3.9+)
uv sync
camunda-rag build-java            # собирает java-analyzer/target/java-analyzer.jar (Maven)

# snapshot репозитория и модель
git clone --depth 1 https://github.com/camunda/camunda.git ~/src/camunda
uv run python -c "from huggingface_hub import snapshot_download as d; d('BAAI/bge-m3', local_dir='models/bge-m3', \
  allow_patterns=['*.json','sentencepiece.bpe.model','pytorch_model.bin','1_Pooling/*'])"

uv run camunda-rag index          # discovery → extractors → schema → SQLite → BGE-M3/FAISS → publish
uv run camunda-rag serve          # http://127.0.0.1:8765  (MCP: /mcp, OpenAPI: /docs)
```

Другие команды: `camunda-rag describe INCIDENT [--dialect oracle]`, `camunda-rag queries --table INCIDENT --access write`,
`camunda-rag search "инциденты процесса"`, `camunda-rag validate "SELECT ..." --dialect mssql`,
`camunda-rag inventory --status unsupported`. Флаги `index --no-embed` (только каталог и FTS) и `--skip-java`.

Конфигурация: `config.toml`, переопределяется переменными `CAMUNDA_RAG_*` (вложенные: `CAMUNDA_RAG_EMBEDDING__DEVICE=cpu`).
Ключевые параметры: `repo_path`, `dialects`, `default_dialect`, `table_prefix`, `liquibase_contexts/labels`,
`[embedding] model_path, device (auto|cpu|mps|cuda), batch_size, max_tokens, offline`, `[limits]`.
Для `device=auto` ускоритель проверяется явно (CUDA, затем MPS, иначе CPU). Явно запрошенное недоступное устройство даёт ошибку.

## Snapshot и карта источников

Проиндексирован commit `3c9f96c185ff74a3343eef15d88f45ea7407ee00` (13.10.2026). Обследованы все 26 909
отслеживаемых файлов (`git ls-files`). Каждый получает роль (`main/test/doc/build`) и статус: `processed`,
`irrelevant`, `excluded` (с причиной) или `unsupported`. Статусы хранятся в `source_files`
и доступны через `camunda-rag inventory`.

| Источник | Где найден | Экстрактор |
|---|---|---|
| Liquibase changelog (XML) | `db/rdbms-schema/src/main/resources/db/changelog/rdbms-exporter/**` (master + 8.9.0 … 8.11.0, schema-version-seed); тестовые changelog в `db/rdbms*/src/test/resources` | `extractors/liquibase.py` + `schema.py` |
| Параметры диалектов | `db/rdbms-schema/src/main/resources/db/vendor-properties/*.properties` (h2, mariadb, mssql, mysql, oracle, postgresql) | `extractors/vendor_properties.py` |
| MyBatis XML | `db/rdbms/src/main/resources/mapper/*.xml` (39 файлов, около 390 statements, около 100 `<sql>`, `databaseId` для 6 СУБД) | `extractors/mybatis.py` |
| Mapper-интерфейсы, DbModel/DbQuery/Entity/DTO | `db/rdbms/**/sql`, `db/rdbms/**/read|write/domain`, `search/search-domain`, `webapps-schema/**/entities`, `optimize/**` | `java-analyzer` (JVM) |
| SQL в Java-строках | `db/rdbms` (`RdbmsSchemaVersionStore`, typehandlers, `LiquibaseScriptGenerator`), тесты `qa/acceptance-tests` | `java-analyzer` |
| SQL-файлы | `db/rdbms-schema/src/test/resources/*.sql`, `db/docker-compose-init/*.sql`, `zeebe/test-util/**/db-init-scripts` | `extractors/sql_files.py` |
| Elasticsearch/OpenSearch маппинги | `webapps-schema/src/main/resources/schema/{elasticsearch,opensearch}/create/{index,template}`, `zeebe/exporters/{elasticsearch,opensearch}-exporter/src/main/resources` | `extractors/search_schema.py` |
| RocksDB column families | `zeebe/protocol/.../ZbColumnFamilies.java` | `java-analyzer` → `rocksdb_column_family` |
| ES DSL в text blocks | `optimize`, тесты | `java-analyzer`, `language=es_dsl` (без SQLGlot) |
| Документация | `docs/**`, `*.md` с терминами модели данных (около 140 файлов) | `extractors/docs.py` |

JPA/JPQL/HQL в snapshot не найдены (`jakarta.persistence` отсутствует). Распознавание `@Query`/`createQuery`
всё равно встроено, и такие запросы получили бы `language=jpql` без SQLGlot.

## Архитектура

```
discovery ─► extractors ─► models (промежуточная модель) ─► schema (replay профиля) ─► storage (SQLite)
                │                                                                     │
                └─ java-analyzer (JVM, JSONL v1)                    embeddings (BGE-M3) ─► FAISS
                                                                                      │
                     service (единый слой, один опубликованный snapshot) ◄────────────┘
                        ├─ cli.py (Typer)
                        └─ http_app.py (FastAPI REST + MCP streamable HTTP)
```

| Модуль | Ответственность |
|---|---|
| `discovery.py` | `git ls-files`, роль файла, первичная маршрутизация. Регулярки только отбирают кандидатов |
| `extractors/*` | структурный разбор каждого вида источника, provenance для каждого факта |
| `models.py` | промежуточная модель (Pydantic), `FactStatus`: `declared / recovered / assumed / unknown` |
| `schema.py` | восстановление схемы для профиля `(компонент, диалект, contexts, labels, параметры)` |
| `sql_analysis.py` | SQLGlot: AST, нормализация, fingerprint, таблицы/колонки/алиасы, CTE, подзапросы, JOIN, фильтры, агрегаты, сортировки, параметры, read/write, lineage проекций |
| `storage.py` | DDL каталога, запись, атомарная публикация поколения (`data/CURRENT`) |
| `embeddings.py` / `retrieval.py` | BGE-M3, чанкинг токенизатором модели, гибридный поиск |
| `validation.py` | статическая валидация SQL против каталога профиля |
| `service.py` | `search / describe / queries / validate / status` над одним поколением |
| `java-analyzer/` | JavaParser 3.26 + Symbol Solver, отдельный процесс, версия формата JSONL: `schema_version=1` |

Поколение индекса собирается в `data/generations/<id>/` (`catalog.sqlite`, `vectors.faiss`) и публикуется
атомарной заменой `data/CURRENT`. Сервис привязан к одному поколению и переключается под блокировкой,
поэтому все операции одного запроса видят один snapshot.

## Структурное извлечение и границы поддержки

### Java (`java-analyzer`)
* Разбираются строковые литералы, text blocks, конкатенации, `String.valueOf`, `Enum.name()`, условный оператор.
* `final`-константы разрешаются через Symbol Solver, в том числе межфайловые (`RdbmsTableNames.SCHEMA_VERSION`).
* Effectively-final локальные переменные разрешаются; переприсвоенные остаются дырой.
* Разбираются аннотации `@Select/@Insert/@Update/@Delete/@Query/@Named(Native)Query`.
* Цепочки `StringBuilder/StringBuffer` разбираются вместе с инициализатором `new StringBuilder(x).append(y)`, ветками `if/else` и циклами.
* Пределы: глубина вычисления `java_max_depth`, число вариантов `java_max_variants`, развёртка циклов `java_max_loop_unroll`. Их превышение записывается в диагностику.
* Неразрешённое выражение (параметр, вызов метода, `String.format`, переприсвоенная переменная) остаётся дырой `⟪expr⟫` с причиной, а вариант помечается `partial`. Единственная подстановка — дыра, которая буквально равна параметру `prefix`: в неё ставится префикс профиля, и параметр помечается как `java_hole`.
* Выгружаются mapper-интерфейсы (методы, параметры, `@Param`, Javadoc), классы `*DbModel/*DbQuery/*Entity/*Dto` (поля, Javadoc) и enum `*ColumnFamilies`.
* Код не исполняется. Билдеры запросов (jOOQ/QueryDSL) в snapshot отсутствуют. ES-запросы, собираемые Java-клиентом (`optimize`), и маппинги Optimize, собираемые в Java (`XContentBuilder`), **не восстанавливаются** и остаются символами Java.
* Для других языков (TS/Go/Python/shell) действует prefilter: файлы с SQL-кандидатами получают статус `unsupported`. В этом snapshot таких нет, поэтому Tree-sitter не подключался.

### MyBatis
* Дерево шаблона сохраняется целиком (`queries.template_json`): `text`, `placeholder` (`#`/`$`, jdbcType, typeHandler), `if`, `choose/when/otherwise`, `foreach`, `trim/where/set`, `include` (с `<property>`), `bind`.
* `databaseId` учитывается по правилам MyBatis: для диалекта берётся statement или `<sql>` со своим `databaseId`, иначе общий. `include` разрешаются между файлами.
* Семантика `${}`:
  * `include_property` — свойство из `<include>`;
  * `config_property` — переменная конфигурации из vendor properties или `prefix` профиля;
  * всё остальное — `substitution`, то есть runtime-подстановка. Вариант с ней получает статус `partial`, в SQLGlot **не передаётся**, а имя никогда не подменяется фиктивным.
* `#{}` превращается в bind `?`.
* План вариантов без декартова произведения: «все `<if>` истинны», «истинны все, кроме веток, требующих runtime `${}`», «все ложны», затем по одному варианту на каждую альтернативную ветку каждого `<choose>`. Предел — `mybatis_max_variants` на statement и диалект, превышение записывается в диагностику. `foreach` разворачивается в `mybatis_foreach_iterations` итераций.
* Соседние `<if>`, проверяющие одно выражение на равенство разным константам (`x == 'A'`, `x.equals('B')`), считаются взаимоисключающими и рендерятся как неявный `<choose>`. Прочие OGNL-условия не вычисляются.
* Каждый вариант хранит условия своего формирования (`conditions_json`).

### SQL и DDL (SQLGlot 30)
* Диалект определяется по `databaseId`, имени и каталогу SQL-файла или по имени класса (Java). Если доказательств нет, берётся `default_dialect`, и это помечается как допущение.
* Соответствие диалектов: `mariadb` разбирается как `mysql`, `mssql` как `tsql`. Для H2 диалекта в SQLGlot нет: используется общий, это видно в диагностике.
* Непрозрачный узел `Command` получает статус `unsupported`, а не «разобрано».
* Хранятся исходный SQL, нормализованный SQL, сериализованный AST, fingerprint (литералы заменены на `?`), версия парсера и диагностика.

### Схема и миграции
* Liquibase XML/YAML/JSON сводятся к единой структуре.
* Поддерживается: `include/includeAll` (в порядке Liquibase), `property` (с `dbms`/`context`, первое определение побеждает), `dbms`, `context`/`contextFilter` и `labels` (выражения с `,`/`and`/`!`), `preConditions` с трёхзначной логикой. `tableExists`, `columnExists`, `indexExists`, `primaryKeyExists`, `foreignKeyConstraintExists` и `dbms` вычисляются по восстановленному состоянию; `sqlCheck` и прочие остаются неизвестными.
* Поддерживаемые изменения: `createTable`, `add/drop/renameColumn`, `renameTable`, `dropTable`, `modifyDataType`, `add/dropNotNullConstraint`, `add/dropDefaultValue`, PK/UK/FK, `create/dropIndex`. `<sql>` разбирается через SQLGlot: `CREATE TABLE/INDEX`, `ALTER ADD/DROP COLUMN`, `DROP`. DML не влияет на схему.
* Профиль — состояние **после** replay, исторические определения не сливаются. Неопределённость остаётся видимой:
  * `uncertain_precondition:*` и `unknown_effect:*` в `schema_change_outcomes`;
  * `completeness.level = complete | partial` с причинами;
  * неразрешённые параметры changelog.
* Параметры changelog (`prefix`, `userCharColumnSize`, `errorMessageSize`, `treePathSize`) берутся так же, как их задаёт `LiquibaseScriptGenerator`: из vendor properties диалекта.
* Тип колонки — абстрактный тип Liquibase (`type_source=liquibase_type`), его отображение в конкретный тип СУБД не вычисляется. Тип Java-поля никогда не используется как SQL-тип.

### Нереляционные хранилища
* ES/OS-маппинги дают сущности с `storage_kind` `elasticsearch_index`, `opensearch_index` или `*_index_template`. Поля — плоские пути с multi-fields.
* RocksDB даёт `rocksdb_column_family`: имена и scope из `ZbColumnFamilies`. Типы ключей и значений не извлекаются, поэтому `completeness=partial`.
* Нереляционные сущности не валидируются через SQLGlot: валидатор отвечает `non_relational_entity`.

## Каталог (SQLite)

Таблицы: `snapshots, index_generations, source_files, schema_profiles, entities, columns, constraints, indexes,
schema_changes, schema_change_outcomes, relationships, queries, query_variants, query_parameters,
query_table_refs, query_column_refs, symbols, mappings, evidence, diagnostics, retrieval_documents, chunks,
chunks_fts (FTS5), embedding_records`.

* **Provenance.** `evidence(subject_kind, subject_id, repository, commit_sha, path, line_start, line_end, symbol, extractor, evidence_kind, confidence, fact_status)` хранится для сущностей, колонок, ограничений, индексов, изменений схемы, связей, запросов, символов и mappings.
* **Идентификатор сущности:** `profile_key | storage_kind | component | schema | name`. Профиль включает компонент, диалект, contexts, labels и параметры.
* **Связи.** `declared_fk` (объявлены в changelog) отделены от `inferred_join` (восстановлены из JOIN-условий SQLGlot; хранится число запросов-доказательств).
* **Mappings:**
  * `mapper_method` — namespace плюс id ↔ метод интерфейса;
  * `result_map` — колонка ↔ свойство Java-типа. Колонка привязывается через lineage проекции, а если его нет — через единственную подходящую таблицу; неоднозначные случаи не записываются.
* **Назначение таблицы или колонки.** Источники по порядку:
  1. `remarks` и комментарии changelog (`declared`);
  2. Javadoc связанного Java-типа или поля (`linked`);
  3. иначе «назначение не установлено по исходникам».

  Отдельно хранится детерминированное резюме по использованию (`usage_summary`, `status=inferred`).

## Поиск (BGE-M3 + FAISS + FTS5)

* **Документы:** таблица или индекс, группы колонок больших сущностей (по 25 колонок), связь, запрос, mapper-метод, Java-тип, секция документации.
* **Чанкинг.** Документ режется по границам смысловых единиц (строки, абзацы, SQL-клаузы), а длина считается **токенизатором BGE-M3**. Единица, которая не помещается целиком, явно делится по токенам, и это записывается в статистику (`oversized_units_split`). `embed()` отвергает вход длиннее `max_seq_length`, так что молча ничего не обрезается.
* **Каналы:**
  1. точные идентификаторы (имена таблиц и колонок, statement id, FQN);
  2. FTS5 BM25 по тексту и разложенным идентификаторам (`PROCESS_INSTANCE_KEY` → `process instance key`);
  3. dense-поиск: нормализованные векторы BGE-M3 в `IndexFlatIP`.

  Результаты объединяются через RRF (k=60).
* Фильтры: `storage_kind`, `component`, `doc_types`, `include_tests`.
* `describe` и валидация читают полный каталог и не зависят от top-k.

## MCP-инструменты (`/mcp`, streamable HTTP, JSON-ответы)

| Инструмент | Аргументы | Возвращает |
|---|---|---|
| `search` | `query, k=10, storage_kind?, component?, doc_types?, include_tests=false` | хиты с RRF-score, рангами по каналам, типом документа, сущностью-субъектом, provenance и фрагментом |
| `describe` | `name, dialect?, storage_kind?, component?, include_tests=false` | `status` (`ok`, `ambiguous` с кандидатами, `not_found` с подсказками); сущность, профиль и полнота, назначение с источниками, `usage_summary`, колонки (тип, nullability, default, PK, назначение, Java-mappings, история), ограничения, индексы, связи `declared_foreign_keys` и `inferred_from_joins`, изменения схемы с исходами, число запросов, evidence, другие профили |
| `queries` | `table?, column?, mapper?, access=read/write?, dialect?, language?, text?, include_tests, include_variants, limit` | запросы с mapper-методом, таблицами, статусами вариантов по диалектам, репрезентативным вариантом (SQL, условия, параметры, fingerprint, неразрешённые части) и provenance |

Те же операции есть в REST: `GET /search`, `GET /describe/{name}`, `GET /queries`, `GET /status`, `GET /health`.
MCP принимает только локальный `Host` (защита SDK от DNS rebinding).

## API валидации SQL

`POST /validate {"sql": "...", "dialect": "postgresql|mysql|mariadb|oracle|mssql|h2", "component": null}`

* Только статический анализ, SQL не исполняется. Схема берётся из Liquibase-профилей диалекта. С `component` можно выбрать другой профиль, например SQL-скрипты.
* Статусы: `valid`, `invalid`, `unsupported` (непрозрачный `Command`), `parse_error` (с line/col).
* Коды проблем: `unknown_table`, `unknown_column` (с подсказками), `ambiguous_column`, `non_relational_entity`, `unrendered_template` (`#{}`/`${}`), `partial_schema`, `generic_dialect`.
* Ответ содержит для каждого statement тип, нормализованный SQL и таблицы read/write.

## Тесты

`uv run pytest`. Тесты используют мини-репозиторий `tests/fixtures/minirepo` (Liquibase XML+YAML, vendor properties,
MyBatis, Java, ES-маппинг, документация, SQL-файл, TS) и детерминированный fake-embedder, поэтому модель не нужна.
Покрыто:
* порядок include, dbms/context/precondition, неопределённость, rename, `<sql>`-DDL;
* дерево MyBatis, databaseId, config-переменные, `partial` для runtime `${}`, `where`/`set`, взаимоисключающие `<if>`, лимит вариантов, resultMap;
* SQLGlot-анализ;
* статусы всех файлов, `describe`, FK против JOIN, профили диалектов, ES-сущности, `queries`, три канала поиска, валидация, атомарная публикация;
* REST и MCP через JSON-RPC;
* Java-анализатор (запускается, если собран jar);
* чанкинг без потерь.

## Известные ограничения

* OGNL-условия `<if>` не вычисляются, кроме распознавания взаимоисключающих проверок на равенство. Варианты — ограниченная выборка, а не полное покрытие.
* Абстрактные типы Liquibase не переводятся в типы конкретной СУБД. Для H2 нет SQLGlot-диалекта.
* `sqlCheck`-preconditions всегда неопределённы. В основном changelog Camunda их нет, поэтому профиль `rdbms-exporter` полный.
* Индексы и запросы Optimize, собираемые Java-кодом, и типы ключей и значений RocksDB не восстанавливаются.
* Исходное задание пришло обрезанным (на разделе 5). Контракты MCP и API валидации, описанные выше, определены в этом проекте.
