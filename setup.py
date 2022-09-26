import os

from setuptools import find_packages, setup

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
VERSION = "0.0.1"

# examples/tutorials
EXTRAS_REQUIRE = []

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
    },
    python_requires=">=3.7",
    keywords="machine learning statistics probabilistic programming bayesian modeling pytorch",
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    # yapf
)
