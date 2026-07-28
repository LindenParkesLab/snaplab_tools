"""Regenerate the API reference pages under ``docs/api/`` from each module's ``__all__``.

Run this after adding, removing, or renaming a public function::

    python docs/_gen_api.py

The generated pages are committed, not built on the fly, so the docs tree is inspectable and
diffable. ``tests/test_docs.py`` checks the committed pages still match ``__all__``, so forgetting
to re-run this is a test failure rather than a silently missing API page.
"""
import importlib
from pathlib import Path

DOCS = Path(__file__).resolve().parent
API = DOCS / "api"

# Each page groups one or more modules under a heading. Order here is the order on the page; the
# order of PAGES is the order in the sidebar.
PAGES = [
    ("stats", "Statistics", "snaplab_tools.stats", [
        ("snaplab_tools.stats", None),
    ]),
    ("nulls", "Null models", "snaplab_tools.nulls", [
        ("snaplab_tools.nulls", None),
    ]),
    ("gams", "GAMs and change points", "snaplab_tools.gams", [
        ("snaplab_tools.gams", None),
    ]),
    ("topology", "Network topology", "snaplab_tools.topology", [
        ("snaplab_tools.topology", None),
    ]),
    ("prediction", "Prediction", "snaplab_tools.prediction.regression", [
        ("snaplab_tools.prediction.regression", None),
    ]),
    ("plotting", "Plotting", "snaplab_tools.plotting", [
        ("snaplab_tools.plotting.plotting", "Figures"),
        ("snaplab_tools.plotting.utils", "Style, colormaps, and annotation"),
    ]),
    ("timeseries", "Time series", "snaplab_tools.signal", [
        ("snaplab_tools.signal", "Filtering"),
        ("snaplab_tools.timescales", "Autocorrelation and intrinsic timescales"),
        ("snaplab_tools.derivs", "Derived measures"),
    ]),
    ("datasets", "Example datasets", "snaplab_tools.datasets", [
        ("snaplab_tools.datasets", None),
    ]),
    ("brainmaps", "Brain maps", "snaplab_tools.brainmaps", [
        ("snaplab_tools.brainmaps", None),
    ]),
    ("utils", "Utilities", "snaplab_tools.utils", [
        ("snaplab_tools.utils", None),
    ]),
]

# Names in __all__ that are module constants rather than functions/classes; autosummary cannot
# build a stub page for these, so they are listed in prose instead.
CONSTANTS = {"SIGNALS", "TIMESCALE_METHODS", "WB_COMMAND", "YEO7_COLORS"}


def _autosummary_block(module, names):
    lines = [
        "```{eval-rst}",
        f".. currentmodule:: {module}",
        "",
        ".. autosummary::",
        "   :toctree: generated/",
        "   :nosignatures:",
        "",
    ]
    lines += [f"   {name}" for name in names]
    lines += ["```", ""]
    return lines


def build_page(slug, title, primary_module, sections):
    out = [
        f"# {title}",
        "",
    ]
    doc = importlib.import_module(primary_module).__doc__ or ""
    summary = doc.strip().split("\n\n")[0].replace("\n", " ").strip()
    if summary:
        out += [summary, ""]

    for module, heading in sections:
        mod = importlib.import_module(module)
        names = [n for n in mod.__all__ if n not in CONSTANTS]
        constants = [n for n in mod.__all__ if n in CONSTANTS]

        if heading:
            out += [f"## {heading}", ""]
        out += _autosummary_block(module, names)
        for const in constants:
            out += [
                f"```{{eval-rst}}",
                f".. autodata:: {module}.{const}",
                "   :no-value:",
                "```",
                "",
            ]
    return "\n".join(out).rstrip() + "\n"


def build_index():
    out = [
        "# API reference",
        "",
        "Every public function and class in `snaplab_tools`, grouped by what it is for. Each module",
        "page opens with a short orientation; follow a name through for the full signature,",
        "parameters, and notes.",
        "",
        "For worked examples rather than signatures, see the [tutorials](../tutorials/index.md).",
        "",
        "```{toctree}",
        ":maxdepth: 2",
        "",
    ]
    out += [slug for slug, _, _, _ in PAGES]
    out += ["```", ""]

    out += ["## Modules at a glance", "", "| Module | What it covers |", "| --- | --- |"]
    for slug, title, primary, _ in PAGES:
        doc = importlib.import_module(primary).__doc__ or ""
        first = doc.strip().split("\n\n")[0].replace("\n", " ").strip()
        # Keep the table scannable.
        if len(first) > 130:
            first = first[:127].rsplit(" ", 1)[0] + "..."
        out.append(f"| [{title}]({slug}.md) | {first} |")
    out.append("")
    return "\n".join(out)


def main():
    API.mkdir(parents=True, exist_ok=True)
    for slug, title, primary, sections in PAGES:
        (API / f"{slug}.md").write_text(build_page(slug, title, primary, sections))
        print(f"wrote api/{slug}.md")
    (API / "index.md").write_text(build_index())
    print("wrote api/index.md")


if __name__ == "__main__":
    main()
