"""Publication-figure helpers.

Split across two modules:

:mod:`~snaplab_tools.plotting.plotting`
    The plots themselves -- annotated correlation plots, cortical surface renderings, null
    distributions, KDE and paired-line plots.

:mod:`~snaplab_tools.plotting.utils`
    The pieces they are built from, most of which are useful on their own: global style settings,
    the lab colour palette, colormap factories with a colour-vision-deficiency check, p-value
    formatting, and axis styling.

Nothing is re-exported at package level, so import from the module you want::

    from snaplab_tools.plotting.plotting import plot_correlation
    from snaplab_tools.plotting.utils import set_plotting_params
"""
