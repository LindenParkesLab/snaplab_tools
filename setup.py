from setuptools import find_packages, setup

setup(
    name="snaplab_tools",
    version="0.0.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "statsmodels",
        "nibabel",
        "nilearn",
        "brainsmash",       # BrainSMASH variogram nulls (snaplab_tools.nulls)
        "bctpy",
        "pygam",            # penalized-spline GAM fitting (snaplab_tools.gams)
        "joblib",           # parallel bootstrap engine (snaplab_tools.gams)
        "tqdm",
        "matplotlib",
        "seaborn",
        "Pillow",
        "GitPython",
    ],
    extras_require={
        # Multi change-point / non-L2 cost models in snaplab_tools.gams. The single-boundary
        # L2 detector is exact and dependency-free; ruptures is only needed beyond that.
        "changepoint": ["ruptures"],
    },
    # Ship the bundled null-model resources (surfaces, parcellations, prebuilt distance matrices)
    # inside the wheel so `snaplab_tools.nulls` is self-contained on any machine, not just an
    # editable dev checkout.
    package_data={
        "snaplab_tools.nulls": [
            "resources/surfaces/*.surf.gii",
            "resources/parcellations/*.dlabel.nii",
            "resources/parcellations/*.csv",
            "resources/distances/*.npy",
        ],
    },
)
