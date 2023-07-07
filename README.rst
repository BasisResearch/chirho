|Build Status|

.. index-inclusion-marker

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

-  **Interventional**: *How many COVID-19 hospitalizations will occur if
   the the USA imposes a national mask mandate?*

-  **Counterfactual**: *Given that 100,000 people were infected with
   COVID-19 in the past month, how many would have been infected if a
   mask mandate had been in place?*

-  **Explanation**: *Why were 100,000 people infected with COVID-19 in
   the past month?*

-  **Causal structure discovery**: *What individual attributes influence
   risk of COVID-19 hospitalization?*

In our `Tutorial <https://basisresearch.github.io/causal_pyro/tutorial_i.html>`_ we answer some of 
these questions. See also `Learn more <#learn-more>`__.

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
   def queried_model():
       return do(model, {"x": 1})

   # Generate 10,000 samples from the observational distribution P(y) ~ N(0, 2)
   obs_samples = pyro.infer.Predictive(model, num_samples=1000)()["y"]

   # Generate 10,000 samples from the interventional distribution P(y | do(X=1)) ~ N(1, 1)
   int_samples = pyro.infer.Predictive(queried_model(), num_samples=1000)()["y"]

Learn more
----------

We have written a number of tutorials and examples for Causal Pyro. We
have tried to choose simple examples that would be of interest to both
the causal inference and probabilistic programming communities: they
collectively span Pearl’s causal hierarchy (Pearl 2009), and
most are broadly applicable, empirically validated, have an
unconventional or limited identification result, and make use of modern
probabilistic machine learning tools, like neural networks or stochastic
variational inference.

Our examples demonstrate how real-world causal assumptions can be expressed as probabilistic programs 
and real-world causal estimands can be expressed as program transformations.
These example illustrate how Causal Pyro is compatible with any inference method 
implemented in Pyro, including the kinds of scalable gradient-based
approximations that power much of the modern probabilistic machine learning landscape.

- `Tutorial <https://basisresearch.github.io/causal_pyro/tutorial_i.html>`_
  - Key observations inspiring Causal Pyro's design and outlines a causal Bayesian workflow for using Causal Pyro to answer causal questions
- `Example: Backdoor Adjustment Criteria <https://basisresearch.github.io/causal_pyro/backdoor.html>`_
  - Adjusting for observed confounders
- `Example: Causal Effect Variational Autoencoder <https://basisresearch.github.io/causal_pyro/cevae.html>`_
  - Causal inference with deep models and proxy variables
- `Example: Mediation analysis and (in)direct effects <https://basisresearch.github.io/causal_pyro/mediation.html>`_
  - Mediation analysis for path specific effects
- `Example: Deep structural causal model counterfactuals <https://basisresearch.github.io/causal_pyro/deepscm.html>`_
  - Counterfactuals with normalizing flows
- `Example: Structured Latent Confounders <https://basisresearch.github.io/causal_pyro/slc.html>`_
  - Causal effect estimation when latent confounders are shared across groups
- `Design notes <https://basisresearch.github.io/causal_pyro/design_notes>`_
  - Technical implementation details of Causal Pyro using effect handlers

*Note*: These tutorials and examples assume some familiarity with Pyro and
probabilistic programming. For introductory Pyro tutorials, please see
`Additional background reading
material <#additional-background-reading-material>`__ below.

Documentation
-------------
- `Counterfactual <https://basisresearch.github.io/causal_pyro/counterfactual.html>`_
  - Effect handlers for counterfactual world splitting
- `Interventional <https://basisresearch.github.io/causal_pyro/interventional.html>`_
  - Effect handlers for performing interventions
- `Observational <https://basisresearch.github.io/causal_pyro/observational.html>`_
  - Effect handler utilities for computing probabilistic quantities for 
  partially deterministic models which is useful for counterfactual reasoning
- `Indexed <https://basisresearch.github.io/causal_pyro/indexed.html>`_
  - Effect handler utilities for named indices in Causal Pyro which is useful for manipluating
  and tracking counterfactual worlds
  
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
Pearl, Judea. *Causality: Models, Reasoning and Inference*. 2nd ed. USA: Cambridge University Press, 2009.


.. |Build Status| image:: https://github.com/BasisResearch/causal_pyro/actions/workflows/test.yml/badge.svg
   :target: https://github.com/BasisResearch/causal_pyro/actions/workflows/test.yml