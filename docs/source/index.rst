.. causal_pyro documentation master file, created by
   sphinx-quickstart on Mon Sep 26 14:49:58 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Causal Probabilistic Programming with Pyro
==========================================

Despite the tremendous progress over the last two decades in reducing
causal inference to statistical practice, the "causal revolution"
proclaimed by Pearl and others remains incomplete, with a sprawling and
fragmented literature inaccessible to non-experts and still somewhat
isolated from cutting-edge machine learning research and software tools.

Functional probabilistic programming languages are promising substrates
for bridging this gap thanks to the close correspondence between their
operational semantics and the field’s standard mathematical formalism of
structural causal models.

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   introduction_i
   introduction_ii

Example applications
--------------------

To illustrate the utility of this approach, we have included several
examples from the causal inference literature expressed as probabilistic
programs using variations on a twin-world semantics of causal inference,
in which a model defines a joint distribution on observed and
counterfactual outcomes conditional on a set of structural functions.

We have tried to choose simple examples that would be of interest to
both the causal inference and probabilistic programming communities:
they collectively span Pearl’s causal hierarchy [@pearl2001bayesian],
and most are broadly applicable, empirically validated, have an
unconventional or limited identification result, and make use of modern
probabilistic machine learning tools, like neural networks or stochastic
variational inference.

Our descriptions demonstrate how diverse real-world causal estimands and
causal assumptions can be expressed in declarative code free of
unnecessary jargon and compatible with any inference method implemented
in the underlying PPL, especially scalable gradient-based
approximations.

.. toctree::
   :maxdepth: 2
   :caption: Examples

   backdoor
   cevae
   deepscm
   slc
   mediation


Design notes
------------

The probabilistic programming language `Omega.jl <http://www.zenna.org/Omega.jl/latest/>`_
exemplifies this connection with its counterfactual semantics, but making standard probabilistic
inference methods compatible with Omega’s highly expressive
measure-theoretic semantics for conditioning remains an open problem.

In contrast, Pyro is an older, more established PPL built on top of
Python and PyTorch that already has a program transformation
``pyro.poutine.do`` for intervening on ``sample`` statements.

.. code:: python

   def model():
     x = sample("x", Normal(0, 1))
     y = sample("y", Normal(x, 1))
     return x, y

   x, y = model()
   assert x != 10  # with probability 1

   with pyro.poutine.do({"x": 10}):
     x, y = model()
   assert x == 10

However, this transformation is too limited to be ergonomic for most
causal inference problems of interest to practitioners. This Pyro
library explores a new programming model for causal inference
intermediate in expressivity between Pyro’s limited built-in approach
and Omega’s highly expressive one, in which interventions are defined as
operations on values within a Pyro model:

.. code:: python

   def intervene(obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
     return act

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   primitives
   counterfactual
   reparam
   query


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

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
