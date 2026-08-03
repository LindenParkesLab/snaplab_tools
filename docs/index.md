---
sd_hide_title: true
---

# snaplab_tools

::::{grid} 1
:::{grid-item}
:class: sd-text-center sd-pt-4 sd-pb-2

```{rubric} snaplab_tools
```

Analysis tools used by the [SNaP Lab](https://github.com/LindenParkesLab).

These tools are a constant work in progress, developed around the lab's own needs. Pull requests
are welcome if you want to contribute. However, we are not taking feature requests (e.g., through GitHub Issues) from outside the lab at this time.
:::
::::

```{code-block} bash
pip install -e ".[surface]"
```

---

## Where to start

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`code` API reference
:link: api/index
:link-type: doc

Every public function and class, with signatures, parameters, and notes on when each is the right
tool.
:::

:::{grid-item-card} {octicon}`download` Installation
:link: installation
:link-type: doc

Requirements, optional extras, and the extra setup needed for cortical surface rendering.
:::

:::{grid-item-card} {octicon}`book` Tutorials
:link: tutorials/index
:link-type: doc

Runnable worked examples. Every one executes on synthetic data, so you can follow along without
access to any particular dataset.
:::
::::

---

## What is in here

Statistics
: Partial correlations, and tests for comparing two *dependent* correlations that share a variable
  — the situation you are actually in whenever you ask "does X relate more strongly to Y than to
  Z?". Analytic ([Steiger](api/stats.md)), bootstrap, and permutation versions.
  Brain-map coupling and decoupling tests for subject-level data.

Spatial null models
: Parcels are not independent observations, so a parametric p-value on a brain-map correlation is
  not meaningful. {mod}`snaplab_tools.nulls` generates BrainSMASH surrogates that preserve spatial
  autocorrelation, using geodesic distance matrices **bundled with the package** — no download, no
  Connectome Workbench required, at every Schaefer resolution from 100 to 1000 parcels and for
  Glasser (HCP-MMP1.0).

GAMs and change points
: Penalized-spline fitting over any predictor, with derivative signals, multivariate change-point
  detection, and a parallel bootstrap engine. Domain-neutral: nothing in it knows about brains.

Network topology
: Consistency-based and quantile thresholding, volume normalization, and normalized rich-club
  coefficients.

Prediction
: Repeated k-fold cross-validated regression with nuisance control, PCA, and permutation testing,
  with every preprocessing step fit inside the training fold.

Plotting
: Annotated correlation plots (with optional embedded null distributions), cortical surface
  rendering, and a colormap system with a colour-vision-deficiency check.

---

## Citing

If you use these tools, please cite the method papers for whatever you actually used — BrainSMASH
for spatial nulls, the Brain Connectivity Toolbox for topology measures, and so on. The relevant
references are noted in each function's documentation.

```{toctree}
:hidden:
:maxdepth: 2

installation
api/index
tutorials/index
contributing
changelog
```
