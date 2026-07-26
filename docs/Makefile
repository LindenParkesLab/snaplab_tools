# Minimal makefile for Sphinx documentation.
#
# `make html` executes every tutorial notebook. The first run is slow; myst-nb caches results in
# _build/.jupyter_cache, so later runs only re-execute notebooks that changed.
#
# Useful targets beyond the Sphinx defaults:
#   make strict   build with warnings as errors (what CI should run)
#   make clean    remove _build, including the execution cache and generated API stubs

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help strict clean Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

strict:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" -W --keep-going $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(BUILDDIR)" api/generated

# Catch-all: route anything else to sphinx-build's -M mode.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
