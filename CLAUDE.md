# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Install: `uv sync`
- Run: `uv run --env-file .env uvicorn doc_ai.main:app --reload` — requires `OPENAI_API_KEY`;
  `core/dependencies.py` builds the OpenAI embedding and chat services at import time, so
  the app (and any `import doc_ai.main` or `doc_ai.core.dependencies`) fails without it.
  Copy `.env.example` to `.env`.
- Tests: `uv run pytest` (no API key, network or OpenAI credits needed). The real OpenAI
  flow is only checked manually; see "Manual end-to-end check" in README.md.
- Single test: `uv run pytest tests/unit/test_chunking_service.py -k test_name`
  (async tests get an `[asyncio]` id suffix, so `-k` is easier than `::name`)
- Lint/format: `uvx ruff check src tests` and `uvx ruff format --check src tests`
  (ruff is not a project dependency)
- The `doc-ai` console script is a placeholder that only prints a greeting.

## Architecture

Upload flow (`DocumentService.create_document`): validate → extract → chunk → embed → store.
- Extraction normalizes everything to `list[PageText]`: PDFs give one per page (numbered
  from 1), `.txt` gives a single one with `page_number=None`. Chunking consumes only this.
- The document, its chunks and its embeddings are stored together at the very end, so a
  failure at any step leaves nothing behind. Keep it that way. `vector_store.add` is the
  first step of that final commit because it validates and can raise; it inserts all or
  nothing, and nothing after it awaits.
- Storage is in memory on `DocumentService`: `documents` list plus `chunks` and `embeddings`
  dicts keyed by document id. Ids come from `itertools.count`. Don't go back to
  `len(documents) + 1`: it races across the embedding await (covered by a concurrency test).

Layering:
- Services take their collaborators through the constructor, typed as `Protocol`s from
  `interfaces/`. All composition happens in `core/dependencies.py`; routers only use
  `DocumentServiceDependency` / `QuestionServiceDependency` and must not construct services
  or import LangChain.
- LangChain is confined: `langchain_text_splitters` only in `services/chunking_service.py`,
  `langchain_openai` only in the factories `create_openai_embedding_service`
  (`services/embedding_service.py`) and `create_openai_llm_service`
  (`services/llm_service.py`). Other layers use the project's own models.
  `tests/unit/test_architecture.py` enforces this with an AST import check.
- `models/` holds internal Pydantic models; `schemas/` holds API response models.
  Embedding vectors are deliberately not exposed through the API.
- Errors are domain exceptions in `exceptions/`. A new one needs a handler in
  `exceptions/handlers.py` and registration in `main.py`. The `EmbeddingError` and
  `LLMError` handlers log the provider's cause and return only a generic message (502).
- `Settings` (`core/config.py`) is a plain class that reads `os.getenv` when instantiated.
  It must not require `OPENAI_API_KEY`, because the tests import it; the key is checked in
  the provider factories instead. Invalid chunk/retrieval settings raise at startup.

Vector store (`services/vector_store.py`, `InMemoryVectorStore`): exact cosine search
with NumPy (a runtime dependency) over the precomputed `ChunkEmbedding`s; it never embeds
text and doesn't use LangChain. Vectors are normalized to float32 on add; equal scores keep
insertion order. Invalid input (id/model/dimension mismatch, duplicate ids, empty, zero or
non-finite vectors, `k <= 0`) raises `ValueError`.

Retrieval (`services/retrieval_service.py`, `RetrievalService.retrieve(question,
document_id=None)`): strips and validates the question (`InvalidQuestionError`, 400, raised
before the provider is called), then `EmbeddingService.embed_query` → `vector_store.search`
with `RETRIEVAL_TOP_K`. `embed_query` returns a `QueryEmbedding` carrying the model name and
the store rejects a query from a model other than the stored one, so retrieval and upload
must share one `EmbeddingService`. Returns `SearchResult` as-is; vectors never leave it.
It depends only on the two Protocols, so a DB-backed store must not require changes here.
An unknown `document_id` returns `[]`; `QuestionService` does the 404 check.

Q&A flow (`POST /questions` → `QuestionService.answer(question, document_id=None)` →
`Answer(text, sources)`; the API returns `{"answer": text, "sources": [...]}`):
- If `document_id` is set, `DocumentService.get_document` runs first (404 before any paid
  provider call). Then retrieval. If retrieval returns `[]` (nothing to search), it returns
  `NO_CONTENT_ANSWER` without calling the LLM. Otherwise `build_prompt` → `LLMService`.
  It catches no exceptions; the existing handlers map them (400/404/502).
- `LLMServiceInterface.generate(Prompt) -> str` is provider-agnostic: `LLMService` wraps
  any LangChain `BaseChatModel`, and `create_openai_llm_service` (`LLM_MODEL`, fixed 60 s
  timeout) is the only OpenAI part. It knows nothing about retrieval or documents.
- `build_prompt` (`services/question_prompt.py`) is a pure function, not a service: system
  = grounding rules, user = numbered `<source id file [page]>` blocks + the question. Never
  send scores, ids or vectors; `<source` tags inside chunk text are escaped. The source
  numbers match `Answer.sources` order, for future citations.
- The router maps each `SearchResult` in `Answer.sources` to a `SourceResponse`
  (`filename`, `page_number` only), keeping retrieval order. Never expose `SearchResult`,
  scores, ids, chunk indexes, start indexes or vectors from this endpoint. The mapping is
  an HTTP concern and stays out of `QuestionService`.
- `QuestionRequest` has no validation constraints on purpose: `RetrievalService` is the
  single question validator, so every invalid question gets 400 (not a mix of 400 and 422).

Not built yet: inline citations in the answer text, streaming, conversation history. `DocumentDetailResponse` is an
unused placeholder.

## Testing conventions

- Async tests use `pytestmark = pytest.mark.anyio` (the anyio plugin ships with FastAPI;
  there is no pytest-asyncio).
- Tests use real services rather than mocks. Embeddings use LangChain's
  `DeterministicFakeEmbedding` or small `Embeddings` subclasses defined in the test file;
  chat models use LangChain's `FakeListChatModel` / `ParrotFakeChatModel` (echoes the
  prompt) or small `BaseChatModel` subclasses. A fake that raises when called proves a
  provider was not reached. PDFs are generated inside tests with pymupdf.
- HTTP tests (`test_questions_router.py`) use `TestClient` (`httpx` dev dependency). They
  import `doc_ai.main` / `doc_ai.core.dependencies` inside a fixture after patching
  `settings.openai_api_key` (patching the env var is too late: `settings` is created on
  first import of `core.config`), then replace both service dependencies via
  `app.dependency_overrides`. Exception handlers are also tested directly.

## Development Instructions

### General

* Explain what you are going to do and why before making significant changes.
* Prefer simple, maintainable solutions over unnecessary abstractions.
* Do not rewrite working code without a reason.
* Do not introduce dependencies unless they are necessary.
* Keep changes focused on the current task.
* Do not modify unrelated files.
* Before making architectural changes, explain the proposed approach first.
* If something is ambiguous, inspect the existing code and follow its conventions rather than inventing a new pattern.

### Workflow

1. Inspect the relevant files first.
2. Explain the approach briefly.
3. Make the smallest reasonable change.
4. Run the relevant tests or checks.
5. Report what changed and whether the checks passed.

### Code

* Follow the existing project structure and naming conventions.
* Prefer explicit, readable code.
* Avoid unnecessary cleverness.
* Avoid premature abstraction.
* Keep functions and modules focused.
* Handle errors explicitly.
* Do not add comments that merely restate the code.

### Git

* Do not create commits unless explicitly asked.
* Do not modify or delete unrelated changes.
* Preserve existing user changes.
* Before large changes, inspect `git diff` when useful.

### Communication

* Be concise and direct.
* Explain concepts when they are relevant to the implementation.
* Do not over-engineer solutions.
* If there are multiple reasonable approaches, briefly explain the trade-offs before choosing one.
