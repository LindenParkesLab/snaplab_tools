"""Checks that the documentation stays in sync with the code.

The API reference pages under ``docs/api/`` are generated from each module's ``__all__`` by
``docs/_gen_api.py`` and committed. These tests fail when the two drift apart, so adding a public
function without regenerating the pages is caught here rather than showing up as a silently missing
page on the published site.
"""
import importlib
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
API = DOCS / "api"

sys.path.insert(0, str(DOCS))
_gen_api = importlib.import_module("_gen_api")


def _documented_names(page_text, module):
    """Names listed in the autosummary block for `module` on a generated page."""
    # Each block is: .. currentmodule:: <module> ... .. autosummary:: ... <indented names>
    blocks = re.split(r"\.\. currentmodule:: ", page_text)
    for block in blocks[1:]:
        name, _, rest = block.partition("\n")
        if name.strip() != module:
            continue
        body = rest.split("```")[0]
        return {
            line.strip()
            for line in body.splitlines()
            if line.startswith("   ") and line.strip() and not line.strip().startswith(":")
        }
    return set()


ALL_MODULES = [
    (slug, module)
    for slug, _, _, sections in _gen_api.PAGES
    for module, _ in sections
]


@pytest.mark.parametrize("slug,module", ALL_MODULES)
def test_api_page_covers_module_all(slug, module):
    """Every name in a module's __all__ has an entry on its API page."""
    page = API / f"{slug}.md"
    assert page.exists(), f"missing API page {page.relative_to(REPO)}"

    expected = set(importlib.import_module(module).__all__) - _gen_api.CONSTANTS
    documented = _documented_names(page.read_text(), module)

    missing = expected - documented
    assert not missing, (
        f"{module}: {sorted(missing)} in __all__ but not on docs/api/{slug}.md. "
        f"Run `python docs/_gen_api.py` and commit the result."
    )

    extra = documented - expected
    assert not extra, (
        f"{module}: {sorted(extra)} on docs/api/{slug}.md but not in __all__. "
        f"Run `python docs/_gen_api.py` and commit the result."
    )


@pytest.mark.parametrize("slug,module", ALL_MODULES)
def test_all_names_exist_and_are_documented(slug, module):
    """Every name in __all__ resolves to a real object that has a docstring."""
    mod = importlib.import_module(module)
    for name in mod.__all__:
        assert hasattr(mod, name), f"{module}.__all__ names {name!r}, which does not exist"
        obj = getattr(mod, name)
        if name in _gen_api.CONSTANTS:
            continue
        assert obj.__doc__, f"{module}.{name} has no docstring"


def test_generated_pages_are_current():
    """The committed API pages match what _gen_api.py would write right now."""
    stale = []
    for slug, title, primary, sections in _gen_api.PAGES:
        expected = _gen_api.build_page(slug, title, primary, sections)
        actual = (API / f"{slug}.md").read_text()
        if expected != actual:
            stale.append(slug)
    assert not stale, (
        f"docs/api/{{{','.join(stale)}}}.md are out of date. "
        f"Run `python docs/_gen_api.py` and commit the result."
    )


def _tutorial_notebooks():
    """Tutorial notebooks currently in the repo (there may be none while they are being rewritten)."""
    tutorials = DOCS / "tutorials"
    return sorted(tutorials.glob("*.ipynb")) if tutorials.is_dir() else []


def test_every_tutorial_is_in_the_toctree():
    """Any tutorial notebook present is reachable from docs/tutorials/index.md.

    Passes trivially while no tutorials exist -- their absence is a deliberate state, not a
    failure. The checks below still apply the moment one is added back.
    """
    notebooks = [p.stem for p in _tutorial_notebooks()]
    if not notebooks:
        pytest.skip("no tutorial notebooks in the repo")

    index = (DOCS / "tutorials" / "index.md").read_text()
    orphans = [nb for nb in notebooks if nb not in index]
    assert not orphans, f"tutorials not listed in tutorials/index.md: {orphans}"


def test_tutorials_have_committed_outputs():
    """Notebooks must ship with their outputs.

    The docs build renders rather than executes them (nb_execution_mode = "off"), so a notebook
    committed with cleared outputs publishes as a page of code and no results. Executing them is
    CI's job -- see .github/workflows/tutorials.yml.

    A cell with no output at all is fine; plenty of cells legitimately produce none. What this
    catches is a notebook where *nothing* was executed.
    """
    import json

    offenders = []
    for path in _tutorial_notebooks():
        notebook = json.loads(path.read_text())
        code_cells = [c for c in notebook["cells"] if c.get("cell_type") == "code"]
        if code_cells and not any(c.get("outputs") for c in code_cells):
            offenders.append(path.name)

    assert not offenders, (
        f"notebooks have no committed outputs: {offenders}. Run them and save, or use "
        f"`jupyter nbconvert --execute --inplace docs/tutorials/<name>.ipynb`."
    )


def test_tutorials_do_not_reference_local_paths():
    """No tutorial may depend on a path that only exists on one machine.

    Tutorials must run on synthetic data so they execute during the docs build. This is what stops
    them going stale.
    """
    banned = ["/Users/", "/home/", "Google-Drive", "sys.path.extend", "sys.path.append"]
    offenders = {}
    for path in _tutorial_notebooks():
        text = path.read_text()
        hits = [token for token in banned if token in text]
        if hits:
            offenders[path.name] = hits
    assert not offenders, f"tutorials reference machine-specific paths: {offenders}"
