"""
requirements.txt must list every third-party package the repository imports
(static check: parses imports with ast, no network, nothing installed needed).
Run:  python -m pytest tests/test_requirements.py -q
"""
import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# import name -> distribution name, where they differ
DIST = {
    "dotenv": "python-dotenv", "sklearn": "scikit-learn", "yaml": "pyyaml", "jwt": "pyjwt",
    "langchain_core": "langchain-core", "langchain_groq": "langchain-groq",
    "langchain_mistralai": "langchain-mistralai", "sentence_transformers": "sentence-transformers",
}
# Imported only inside a guarded fallback branch (third-choice LLM provider in
# agent/graph.py::get_medxai_llm); not required for the default Groq / Mistral setup.
OPTIONAL = {"langchain_openai"}
SKIP_DIRS = {".git", "venv", ".venv", "node_modules", "__pycache__", "data"}


def _local_modules() -> set[str]:
    names = set()
    for entry in os.listdir(ROOT):
        if entry.endswith(".py"):
            names.add(entry[:-3])
        elif os.path.isdir(os.path.join(ROOT, entry)) and entry not in SKIP_DIRS:
            names.add(entry)
    return names


def _imported_modules() -> set[str]:
    mods = set()
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            if not f.endswith(".py"):
                continue
            with open(os.path.join(dirpath, f), encoding="utf-8") as fh:
                tree = ast.parse(fh.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    mods |= {a.name.split(".")[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    mods.add(node.module.split(".")[0])
    return mods - set(sys.stdlib_module_names) - _local_modules()


def _required() -> dict[str, str]:
    out = {}
    with open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                name = re.split(r"[<>=!~;\[ ]", line, maxsplit=1)[0]
                out[name.lower().replace("_", "-")] = line
    return out


def test_every_imported_third_party_package_is_in_requirements():
    req = _required()
    missing = sorted(m for m in _imported_modules() - OPTIONAL
                     if DIST.get(m, m).lower().replace("_", "-") not in req)
    assert not missing, f"imported but not in requirements.txt: {missing}"


def test_review2_packages_are_listed_with_a_bounded_version():
    req = _required()
    for pkg in ("langchain-groq", "matplotlib", "pytest", "wfdb", "openpyxl", "pandas"):
        assert pkg in req, pkg
        assert ">=" in req[pkg] and "<" in req[pkg].replace("<=", ""), req[pkg]
