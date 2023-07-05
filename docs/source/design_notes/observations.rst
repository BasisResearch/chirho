Classical counterfactual formulations treat randomness as exogenous and shared across factual and counterfactual
worlds. Pyro, however, does not expose the underlying probability space to users, and the cardinality of
randomness is determined by the number of batched random variables at each ``sample`` site. This means that the twin-world
semantics above may not, in an arbitrary model, correspond directly to the classical, counterfactual formulation.
An arbitrary model may assign independent noise to the factual and counterfactual worlds.

.. code:: python

   def model():
     x = pyro.sample("x", Normal(0, 1))  # upstream of a
     ...
     a = intervene(f(x), a_cf)
     ...
     # Higher cardinality of a here will, by default induce independent normal draws,
     #  resulting in different exogenous noise variables in the factual and counterfactual worlds.
     y = pyro.sample("y", Normal(a, b))  # downstream of a
     ...
     z = pyro.sample("z", Normal(1, 1))  # not downstream of a
     # Here, because the noise is not "combined" with a except in this determinstic function g,
     #  the noise is shared across the factual and counterfactual worlds.
     z_a = g(a, z)  # downstream of a

Interestingly, nearly all `PyTorch and Pyro
distributions <https://pytorch.org/docs/stable/distributions.html>`__
have samplers that are implemented as
`deterministic functions of exogenous
noise <https://pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution.rsample>`__,
because `as discussed in Pyro’s tutorials on variational
inference <http://pyro.ai/examples/svi_part_iii.html#Easy-Case:-Reparameterizable-Random-Variables>`__
this leads to Monte Carlo estimates of gradients with much lower
variance. However, these noise variables are not
exposed via to users or to Pyro’s inference APIs.

Reusing and replicating exogenous noise
---------------------------------------

Pyro implements a number of generic measure-preserving
`reparameterizations of probabilistic
programs <https://docs.pyro.ai/en/stable/infer.reparam.html>`__ that
work by transforming individual ``sample`` sites. These are often used
to improve performance and reliability of gradient-based Monte Carlo
inference algorithms that depend on the geometry of the posterior
distribution. Some may introduce auxiliary random variables or only be
approximately measure-preserving.

For example, the standard ``LocScaleReparam`` can transform a
location-scale distribution like ``Normal(a, b)`` into an affine
function of standard white noise ``Normal(0, 1)``. If this distribution
is at a sample site downstream of an ``intervene`` statement whose
semantics are given by the ``TwinWorldCounterfactual`` effect handler
above, the noise value ``x_noise`` will be shared across the factual and
counterfactual worlds because it is no longer downstream.
``TransformReparam`` does something similar for arbitrary invertible
functions of exogenous noise.

.. code:: python

   @TwinWorldCounterfactual()
   @reparam(config={"x": LocScaleReparam()})
   def model():
     ...
     a = intervene(a, a_cf)
     ...
     x = sample("x", Normal(a, b))
     ...

   # the above is equivalent to the following model:
   @TwinWorldCounterfactual()
   def reparam_model():
     ...
     a = intervene(a, a_cf)
     ...
     x_noise = sample("x_noise", Normal(0, 1))
     x = sample("x", Delta(a + b * x_noise))  # degenerate sample() statement, usually abbreviated to deterministic()
     ...

This may still not seem very useful for us, since there is no reason to
expect that the causal mechanisms in a reparameterized model should
correspond a priori to those in the true causal model, even if the joint
observational distributions match perfectly. However, it turns out that
many of the causal quantities we’d like to estimate from data (and for
which doing so is possible at all) can be reduced to counterfactual
computations in surrogate structural causal models whose mechanisms are
determined by global latent variables or parameters.

Soft conditioning for likelihood-based inference
------------------------------------------------

Answering counterfactual queries requires conditioning on the value of
deterministic functions of random variables, an intractable problem in
general.

Approximate solutions to this problem can be implemented using the same
``Reparam`` API, making such models compatible with the full range of
Pyro’s existing likelihood-based inference machinery.

..
    TODO need to also cite the predicate exchange thing here if we want to use this example?

For example, we could implement a new ``Reparam``
class that rewrites observed deterministic functions to approximate soft
conditioning statements using a distance metric or positive semidefinite
kernel and the ``factor`` primitive. This is useful when the observed value is, for example, a predicate
of a random variable :cite:`tavaresPredicateExchangeInference2019`, or e.g. distributed according to a point mass.

.. code:: python

   class KernelABCReparam(Reparam):
     def __init__(self, kernel: pyro.contrib.gp.Kernel):
       self.kernel = kernel
       super().__init__()

     def apply(self, msg):
       if msg["is_observed"]:
         ...  # TODO
         factor(msg["name"] + "_factor", -self.kernel(msg["value"], obs))
         ...

   @reparam(config={"x": KernelABCReparam(...)})
   def model(x_obs):
     ...
     x_obs = sample("x", Delta(x), obs=x_obs)
     ...

This is not the only such approximation possible, and it may not be
appropriate for all random variables. For example, when a random
variable can be written as `an invertible transformation <https://pytorch.org/docs/master/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution>`_
of exogenous noise, conditioning can be handled exactly using something
similar to the existing
`Pyro TransformReparam <https://docs.pyro.ai/en/stable/infer.reparam.html#module-pyro.infer.reparam.transform>`_.
