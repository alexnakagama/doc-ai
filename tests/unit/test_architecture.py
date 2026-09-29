import ast
from pathlib import Path

import doc_ai

SOURCE_ROOT = Path(doc_ai.__file__).parent

PROVIDER_MODULES = {"services/embedding_service.py", "services/llm_service.py"}
LANGCHAIN_MODULES = PROVIDER_MODULES | {"services/chunking_service.py"}


def imported_packages(path: Path) -> set[str]:
    packages: set[str] = set()

    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            packages.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            packages.add(node.module.split(".")[0])

    return packages


def modules_importing(prefixes: tuple[str, ...]) -> set[str]:
    return {
        path.relative_to(SOURCE_ROOT).as_posix()
        for path in SOURCE_ROOT.rglob("*.py")
        if any(package.startswith(prefixes) for package in imported_packages(path))
    }


def test_openai_is_imported_only_by_the_provider_factories():
    assert modules_importing(("openai", "langchain_openai")) <= PROVIDER_MODULES


def test_langchain_is_confined_to_the_provider_and_chunking_services():
    assert modules_importing(("langchain",)) <= LANGCHAIN_MODULES
