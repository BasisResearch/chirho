import sys

from setuptools import find_packages, setup

VERSION = "0.3.0"

try:
    long_description = open("README.rst", encoding="utf-8").read()
except Exception as e:
    sys.stderr.write("Failed to read README: {}\n".format(e))
    sys.stderr.flush()
    long_description = ""

# examples/tutorials
EXTRAS_REQUIRE = [
    "jupyter",
    "graphviz",
    "matplotlib",
    "pandas",
    "seaborn",
    "pytorch-lightning",
    "scikit-image",
    "tensorboard",
    "typing_extensions",
]

DYNAMICAL_REQUIRE = ["torchdiffeq"]

setup(
    name="chirho",
    version=VERSION,
    description="Causal reasoning",
    long_description=long_description,
    packages=find_packages(include=["chirho", "chirho.*"]),
    author="Basis",
    url="https://www.basis.ai/",
    project_urls={
    #     "Documentation": "",
        "Source": "https://github.com/BasisResearch/chirho",
    },
    install_requires=[
        # if you add any additional libraries, please also
        # add them to `docs/source/requirements.txt`
        "pyro-ppl>=1.9.1",
    ],
    extras_require={
        "dynamical": DYNAMICAL_REQUIRE,
        "extras": EXTRAS_REQUIRE,
        "test": EXTRAS_REQUIRE + DYNAMICAL_REQUIRE
        + [
            "pytest",
            "pytest-cov",
            "pytest-xdist",
            "mypy==1.10",
            "black",
            "flake8",
            "isort",
            "sphinx==7.1.2",
            "sphinxcontrib-bibtex",
            "sphinxcontrib-jquery",
            "sphinx_rtd_theme==1.3.0",
            "myst_parser",
            "nbsphinx",
            "nbval",
            "nbqa",
        ],
    },
    python_requires=">=3.9",
    keywords="machine learning statistics probabilistic programming bayesian modeling pytorch",
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",

        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    # yapf
)
