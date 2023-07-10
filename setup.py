import os

from setuptools import find_packages, setup

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
VERSION = "0.0.1"

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
]

setup(
    name="chirho",
    version=VERSION,
    description="Causal reasoning",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    packages=find_packages(include=["chirho", "chirho.*"]),
    author="Basis",
    # url="",
    # project_urls={
    #     "Documentation": "",
    #     "Source": "https://github.com/BasisResearch/chirho",
    # },
    install_requires=[
        # if you add any additional libraries, please also
        # add them to `docs/requirements.txt`
        "pyro-ppl>=1.8.5",
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
            "sphinxcontrib-bibtex",
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
