# Contributing

## Building the docs locally

```bash
pip install -e ".[docs,surface,changepoint]"
cd docs
make html
```

Then open `docs/_build/html/index.html`.

There are currently no executable pages, so the build is fast. Once tutorials return they are run
during the build, with results cached in `_build/.jupyter_cache` so subsequent builds only re-run
pages whose content changed. `make clean` clears that cache, the built HTML, **and** the generated
API stubs in `api/generated/` — use it rather than `rm -rf _build`, or stubs for a removed function
linger and trip the strict build.

To reproduce what CI checks:

```bash
make strict     # sphinx-build -W --keep-going: warnings become errors
```

## Docstrings

NumPy style, rendered by `sphinx.ext.napoleon`. Every name in a module's `__all__` needs one —
{mod}`snaplab_tools.gams` and {mod}`snaplab_tools.nulls` are the best examples in the codebase to
copy from.

Two things worth more effort than the parameter list:

- **Say when the function is the wrong choice.** A reader deciding between
  {func}`~snaplab_tools.stats.steiger_test` and
  {func}`~snaplab_tools.stats.bootstrap_correlation_test` is better served by a sentence on the
  tradeoff than by two immaculate parameter tables.
- **Document surprising behaviour, not just intended behaviour.** NaN handling, whether an
  intercept is added, whether results come back in shuffled order — these are what cost people
  hours.

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

Tutorials are being rewritten and are not currently in the repository; `docs/tutorials/index.md` is
a placeholder. When they return, these are the rules that keep them working:

**Use synthetic data.** Every tutorial must run on a machine with no access to lab data — that is
what lets them execute during the docs build, which in turn is what stops them going stale. The
geodesic distance matrices and parcellations bundled in `snaplab_tools/nulls/resources/` are enough
to generate realistic brain maps offline: smoothing white noise with an exponential kernel over the
geodesic distances gives a map with the spatial autocorrelation real cortical data has, which
matters because a spatial null model has nothing to preserve without it.

**Commit notebooks with outputs stripped.** Outputs are regenerated at build time, and committed
ones only bloat diffs:

```bash
jupyter nbconvert --clear-output --inplace docs/tutorials/your_tutorial.ipynb
```

**Keep it under about two minutes.** The whole tutorial suite has to fit inside Read the Docs' build
limit. Turn permutation and bootstrap counts down (`n_perms=500` rather than 5,000) and say in the
text that you have done so, and why.

## Notebooks that cannot execute

`nb_execution_excludepatterns` in `docs/conf.py` lists notebooks skipped during the build. It should
stay empty. A notebook on that list ships whatever outputs happen to be committed, so it can
silently drift out of agreement with the code — the exact failure mode executing them is meant to
prevent. If something genuinely cannot run in the build environment, prefer restructuring it (or
adding the missing dependency to the `docs` extra) over adding it to the list.

## Running the tests

```bash
pytest tests/
```
