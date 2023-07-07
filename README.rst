|Build Status|

Causal Probabilistic Programming with Causal Pyro
=================================================

Causal Pyro is a causal extension to the Pyro probabilistic programming
language. It was built to bridge the gap between the capabilities of
modern probablistic programming systems, such as Pyro, and the needs of
policymakers, scientists, and AI researchers, who often want to use
models to answer their questions about cause-and-effect relationships.
As a non-exhaustive set of examples, Causal Pyro makes it easier to
answer the following kinds of causal questions that appear frequently in
practice.

Installation
------------

**Install using pip:**

.. code:: sh

   pip install causal_pyro

**Install from source:**

.. code:: sh

   git clone git@github.com:BasisResearch/causal_pyro.git
   cd causal_pyro
   git checkout master
   pip install .

**Install with extra packages:**

To install the dependencies required to run the tutorials in
``examples``/``tutorials`` directories, please use the following
command:

.. code:: sh

   pip install causal_pyro[extras] 

Make sure that the models come from the same release version of the
`Causal Pyro source
code <https://github.com/BasisResearch/causal_pyro/releases>`__ as you
have installed.

Getting Started
---------------

Below is a simple example of how to use Causal Pyro to answer an
interventional question. For more in-depth examples, go to `Learn
more <#learn-more>`__.

.. code:: python

   import torch
   import pyro
   import pyro.distributions as dist
   from causal_pyro.interventional.handlers import do

   pyro.set_rng_seed(101)

   # Define a causal model with single confounder h
   def model():
       h = pyro.sample("h", dist.Normal(0, 1))
       x = pyro.sample("x", dist.Normal(h, 1))
       y = pyro.sample("y", dist.Normal(x + h, 1))
       return y

   # Define a causal query (here intervening on x)
   def query_model():
       return do(model, {"x": 1})

   # Generate 10,000 samples from the observational distribution P(y) ~ N(0, 2)
   obs_samples = pyro.infer.Predictive(model, num_samples=1000)()["y"]

   # Generate 10,000 samples from the interventional distribution P(y | do(X=1)) ~ N(1, 1)
   int_samples = pyro.infer.Predictive(query_model(), num_samples=1000)()["y"]

Learn more
----------

We have written a number of tutorials and examples for Causal Pyro. We
have tried to choose simple examples that would be of interest to both
the causal inference and probabilistic programming communities: they
collectively span Pearl’s causal hierarchy Pearl (Pearl 2001), and
most are broadly applicable, empirically validated, have an
unconventional or limited identification result, and make use of modern
probabilistic machine learning tools, like neural networks or stochastic
variational inference.

Our examples demonstrate how real-world causal assumptions can be expressed as probabilistic programs 
and real-world causal estimands can be expressed as program transformations.
These example illustrate how Causal Pyro is compatible with any inference method 
implemented in Pyro, including the kinds of scalable gradient-based
approximations that power much of the modern probabilistic machine learning landscape.

+-----------------------------------+-----------------------------------+
| Section                           | Description                       |
+===================================+===================================+
| `Documentation <https://basisr    | Full API documentation and        |
| esearch.github.io/causal_pyro>`__ | tutorials                         |
+-----------------------------------+-----------------------------------+
| `Tutorial                         | Key observations inspiring Causal |
| l <https://basisresearch.github.i | Pyro’s design and outlines a      |
| o/causal_pyro/tutorial_i.html>`__ | causal Bayesian workflow for      |
|                                   | using Causal Pyro to answer       |
|                                   | causal questions                  |
+-----------------------------------+-----------------------------------+
| `Example:                         | Adjusting for observed            |
| Backd                             | confounding with Pearl’s backdoor |
| oor <https://basisresearch.github | criteria                          |
| .io/causal_pyro/backdoor.html>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `Example:                         | Implementation of Causal Effect   |
| CEVAE <https://basisresearch.git  | Variational Autoencoder           |
| hub.io/causal_pyro/cevae.html>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `Example:                         | Mediation analysis to target      |
| Mediation                         | various effect estimands          |
| <https://basisresearch.github.    |                                   |
| io/causal_pyro/mediation.html>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `Example: Deep                    | Implementation of Deep Structural |
| SCM <https://basisresearch.githu  | Causal Model                      |
| b.io/causal_pyro/deepscm.html>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `Example: Structured Latent       | Causal effect estimation in the   |
| Con                               | presence of structured latent     |
| founders <https://basisresearch.g | confounders                       |
| ithub.io/causal_pyro/slc.html>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `Design notes                     | Technical implementation details  |
| n                                 | of Causal Pyro using effect       |
| otes <https://basisresearch.githu | handlers                          |
| b.io/causal_pyro/design_notes>`__ |                                   |
+-----------------------------------+-----------------------------------+

*Note*: The tutorials assume some familiarity with Pyro and
probabilistic programming. For introductory Pyro tutorials, please see
`Additional background reading
material <#additional-background-reading-material>`__ below.

Caveats
-------

Causal Pyro does not answer causal questions by magic. In fact, there is
no escaping the fact that

   *behind any causal conclusion there must lie some causal assumption,*

a phrase made famous by Judea Pearl (Pearl 2009). Instead,
Causal Pyro provides a substrate for writing causal assumptions as
probabilistic programs, and for writing causal questions in terms of
program transformations.

Additional background reading material
--------------------------------------

-  Causal Probabilistic Programming Without Tears
   https://drive.google.com/file/d/1Uzjg-vX77BdSnAcfpUcb-aIXxhnAPI24/view?usp=sharing
-  Introduction to Pyro: \ http://pyro.ai/examples/intro_long.html
-  Tensor shapes in Pyro: \ http://pyro.ai/examples/tensor_shapes.html
-  A guide to programming with effect handlers in
   Pyro \ http://pyro.ai/examples/effect_handlers.html
-  Minipyro: \ http://pyro.ai/examples/minipyro.html
-  Reparameterization of Pyro
   programs: \ https://docs.pyro.ai/en/stable/infer.reparam.html
-  Optional: getting started with
   NumPyro \ https://num.pyro.ai/en/stable/getting_started.html


References
----------
Pearl, Judea. *Bayesianism and Causality, or, Why I Am Only a Half-Bayesian*. Volume 24. Springer, Dordrecht, 2001.

Pearl, Judea. *Causality: Models, Reasoning and Inference*. 2nd ed. USA: Cambridge University Press, 2009.


.. |Build Status| image:: https://github.com/BasisResearch/causal_pyro/actions/workflows/test.yml/badge.svg
   :target: https://github.com/BasisResearch/causal_pyro/actions/workflows/test.yml