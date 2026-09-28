# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Install: `uv sync`
- Run: `uv run --env-file .env uvicorn doc_ai.main:app --reload` — requires `OPENAI_API_KEY`;
  `core/dependencies.py` builds the OpenAI embedding service at import time, so the app
  (and any `import doc_ai.main`) fails without it. Copy `.env.example` to `.env`.
- Tests: `uv run pytest` (no API key or network needed)
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
  failure at any step leaves nothing behind. Keep it that way.
- Storage is in memory on `DocumentService`: `documents` list plus `chunks` and `embeddings`
  dicts keyed by document id. Ids come from `itertools.count`. Don't go back to
  `len(documents) + 1`: it races across the embedding await (covered by a concurrency test).

Layering:
- Services take their collaborators through the constructor, typed as `Protocol`s from
  `interfaces/`. All composition happens in `core/dependencies.py`; routers only use
  `DocumentServiceDependency` and must not construct services or import LangChain.
- LangChain is confined: `langchain_text_splitters` only in `services/chunking_service.py`,
  `langchain_openai` only in `create_openai_embedding_service` in
  `services/embedding_service.py`. Other layers use the project's own models.
- `models/` holds internal Pydantic models; `schemas/` holds API response models.
  Embedding vectors are deliberately not exposed through the API.
- Errors are domain exceptions in `exceptions/`. A new one needs a handler in
  `exceptions/handlers.py` and registration in `main.py`. The `EmbeddingError` handler logs
  the provider's cause and returns only a generic message to the client.
- `Settings` (`core/config.py`) is a plain class that reads `os.getenv` when instantiated.
  It must not require `OPENAI_API_KEY`, because the tests import it; the key is checked in
  the embedding factory instead. Invalid chunk settings raise at startup.

Not built yet: retrieval, vector store, LLM Q&A. `schemas/question.py` and
`DocumentDetailResponse` are unused placeholders for that.

## Testing conventions

- Async tests use `pytestmark = pytest.mark.anyio` (the anyio plugin ships with FastAPI;
  there is no pytest-asyncio).
- Tests use real services rather than mocks. Embeddings use LangChain's
  `DeterministicFakeEmbedding` (needs the `numpy` dev dependency) or small `Embeddings`
  subclasses defined in the test file. PDFs are generated inside tests with pymupdf.
- There are no HTTP-level tests (`httpx` isn't installed). Exception handlers are tested by
  calling the handler functions directly.

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
