import os

from setuptools import find_packages, setup

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
VERSION = "0.0.1"

# examples/tutorials
EXTRAS_REQUIRE = [
    "jupyter>=1.0.0",
    "graphviz>=0.8",
    "matplotlib>=1.3",
    # "torchvision>=0.12.0",
    # "visdom>=0.1.4,<0.2.2",  # FIXME visdom.utils is unavailable >=0.2.2
    "pandas",
    # "pillow==8.2.0",  # https://github.com/pytorch/pytorch/issues/61125
    # "scikit-learn",
    "seaborn>=0.11.0",
    # "wget",
    # "lap",
    # 'biopython>=1.54',
    # 'scanpy>=1.4',  # Requires HDF5
    # 'scvi>=0.6',  # Requires loopy and other fragile packages
]

setup(
    name="causal_pyro",
    version=VERSION,
    description="Causal inference with Pyro",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    packages=find_packages(include=["causal_pyro", "causal_pyro.*"]),
    author="Basis",
    # url="",
    # project_urls={
    #     "Documentation": "",
    #     "Source": "https://github.com/BasisResearch/causal_pyro",
    # },
    install_requires=[
        # if you add any additional libraries, please also
        # add them to `docs/requirements.txt`
        "pyro-ppl",
    ],
    extras_require={
        "extras": EXTRAS_REQUIRE,
        "test": EXTRAS_REQUIRE + [
            "pytest",
            "pytest-cov",
            "pytest-xdist",
            "mypy",
            "black",
            "flake8",
            "isort",
            "sphinx",
            "sphinx_rtd_theme",
            "myst_parser",
            "nbsphinx",
        ],
    },
    python_requires=">=3.8",
    keywords="machine learning statistics probabilistic programming bayesian modeling pytorch",
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10.7",
    ],
    # yapf
)
