# Contributing

## Building the docs locally

```bash
pip install -e ".[docs,surface,changepoint]"
cd docs
make html
```

Then open `docs/_build/html/index.html`.

The build only *renders* notebooks; it does not execute them, so it is fast. `make clean` clears
the built HTML **and** the generated API stubs in `api/generated/` — use it rather than
`rm -rf _build`, or stubs for a removed function linger and trip the strict build.

To reproduce what CI checks:

```bash
make strict     # sphinx-build -W --keep-going: warnings become errors
```

## Docstrings

NumPy style, rendered by `sphinx.ext.napoleon`. Every name in a module's `__all__` needs one —
{mod}`snaplab_tools.gams` and {mod}`snaplab_tools.nulls` are the best examples in the codebase to
copy from.

Cross-reference with `` {func}`~snaplab_tools.module.name` `` so the docs link up.

## Adding a public function

1. Write it, with a docstring.
2. Add its name to the module's `__all__`. This is what defines the public API — anything not
   listed is treated as internal and stays out of the documentation.
3. Regenerate the API pages:

   ```bash
   python docs/_gen_api.py
   ```

4. Commit the regenerated `docs/api/*.md` along with your change.

`tests/test_docs.py` fails if the committed API pages disagree with `__all__`, so a forgotten step
3 shows up as a test failure rather than a quietly missing page.

## Adding a tutorial

Tutorials live in `docs/tutorials/` as `.ipynb` files, listed in the toctree in
`docs/tutorials/index.md`. These are the rules that keep them working:

**Use synthetic data from {mod}`snaplab_tools.datasets`.** Every tutorial must run on a machine
with no access to lab data — that is what lets them execute during the docs build, which in turn is
what stops them going stale. If you need a data shape the module does not generate, add a generator
to `snaplab_tools/datasets.py` rather than reaching for a file path or hand-rolling it in the
notebook.

Its maps are built over the real Schaefer geometry bundled in `snaplab_tools/nulls/resources/`, so
they carry genuine spatial autocorrelation. That matters: a spatial null model has nothing to
preserve if you hand it `np.random.randn(400)`, so a tutorial built on plain noise would
demonstrate the machinery while misrepresenting what it does.

**Commit notebooks *with* their outputs.** The published site renders what you commit — a notebook
saved with cleared outputs publishes as a page of code and no results. Re-run it after any change:

```bash
jupyter nbconvert --execute --inplace docs/tutorials/your_tutorial.ipynb
```

Do **not** set `MPLBACKEND=Agg` when you do. It makes `plt.show()` discard the figure instead of
emitting it as cell output, so the notebook executes cleanly and produces no images at all.
ipykernel's default backend is already headless-safe.

`tests/test_docs.py` fails on a notebook whose code cells produced no output at all, which catches
the commonest version of this mistake.

**Keep it under about two minutes.** Turn permutation and bootstrap counts down (`n_perms=500`
rather than 5,000) and say in the text that you have done so, and why. This is now a courtesy to
CI rather than a hard build limit, but the suite still has to finish in reasonable time.

## Where notebooks actually get executed

Not during the docs build. `nb_execution_mode` is `"off"` in `docs/conf.py`, so Read the Docs only
renders the committed outputs.

Executing them is CI's job — `.github/workflows/tutorials.yml` runs every notebook on push and
fails if any of them break. That is what stops a tutorial silently rotting when the API changes.

The split exists because executing during the docs build meant every build installed the full
scientific stack (VTK and brainspace alone are ~200 MB) and depended on several third-party
downloads succeeding. A transient network failure took the documentation offline for reasons that
had nothing to do with the documentation.

One honest limitation: CI verifies that notebooks *run*, not that the committed outputs match what
they would produce now. If you change code that affects a tutorial's numbers, re-run the notebook
and commit the result — CI will not catch stale-but-valid output for you.

## Running the tests

```bash
pytest tests/
```
