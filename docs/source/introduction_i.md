---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Causal probabilistic programming without tears

+++

Despite the tremendous progress over the last two decades in reducing
causal inference to statistical practice, the \"causal revolution\"
proclaimed by Pearl and others remains incomplete, with a sprawling and
fragmented literature inaccessible to non-experts and still somewhat
isolated from cutting-edge machine learning research and software tools.

Functional probabilistic programming languages are promising substrates for bridging this gap thanks to the close correspondence between their operational semantics and the field’s standard mathematical formalism of structural causal models.

+++

## Observation 1: causal models are probabilistic programs

+++

Probabilistic programmers typically think of their code as defining a
probability distribution over a set of variables, but programs contain
more information than just the joint distributions they induce. In this
way, programs are similar to Bayesian networks, which encode not just a
joint distribution but also a generative process that we can imagine
unfolding in time. The language we use to describe Bayesian networks --
*parents* and *children*, *ancestors* and *descendants* -- reflects this
understanding: some variables are generated *before* other variables,
and intuitively, have a causal effect on their immediate children.

Formally, a causal model specifies a *family* of probability
distributions, indexed by a set of *interventions*. An intervention
represents a hypothetical experimental condition, under which we'd
expect the joint distribution over the variables of interest to change.
For example, in a model over the variables *smokes* and *cancer*, the
joint distribution would change under the experimental condition that
randomly assigns each participant to either smoke or not smoke. (For one
thing, the marginal probability of *smokes* would be changed to 50%.)

In probabilistic programs, we can understand interventions as **program
transformations**. For example, in the smoking/cancer model, the
experiment we considered above might be encoded as a program
transformation that replaces assignments to the *smokes* variable in the
true causal program with the line $smokes \sim bernoulli(0.5)$.

A probabilistic program specifies a causal model in that it
  1. specifies a "default" or "observational" joint distribution over the
variables of interest according to the usual semantics of probabilistic
programming languages
  2. encodes the necessary information to
determine the new joint distribution under an arbitrary intervention
(program transformation)---apply the transformation and derive the new
joint distribution.

+++

## Observation 2: causal computations are probabilistic computations

Once we have a causal model, what can we use it for? We briefly describe
several problem types that practitioners of causal inference may be
interested in solving (but do not claim that this is an exhaustive
list):

-   **Causal discovery.** Given data (either observational, or collected
    under experimental conditions, or both), infer the underlying causal
    model, from a class of possible models.

-   **Parameter estimation.** Given data (either observational, or
    collected under experimental conditions, or both), and a causal
    model with unknown parameters $\theta$, infer plausible values of
    $\theta$.

-   **Causal effect estimation.** Given data (either observational, or
    collected under experimental conditions, or both), and a causal
    model (possibly with unknown structure or parameters), estimate a
    *causal effect*, e.g. the Average Treatment Effect or the Individual
    Average Treatment Effect. Such queries are designed to answer
    questions like, "On average, how much better would a patient fare if
    they were given one medication vs. another?"

-   **Counterfactual prediction.** Given observed data, and a causal
    model (possibly with unknown structure or parameters), estimate a
    *counterfactual query*, designed to answer questions like, "Given
    what we know about this patient (including their observed health
    outcome), how would their outcome have differed had we treated them
    differently?"

All of these questions can be posed in a Bayesian framework. In particular, 
causal discovery, parameter estimation, causal effect estimation, and counterfactual prediction can be framed as Bayesian inference in appropriately specified generative models.


The quantities over which we have uncertainty are:

-   the structure of the true causal model,

-   the parameters of the true causal model, and

-   the values of any latent variables posited by the true causal model,
    for each subject in our dataset. (In the presence of experimental
    data, we are also uncertain about the latent variables posited by
    the intervened version of the true causal model, for each subject in
    the experimental dataset.)

We can express priors over these quantities, and likelihoods that relate
them to the observations. For example, suppose we are uncertain about
the true model structure $m$, and its unknown parameters $\theta$, as
well as the values of latent variables $x$, but we have observed $y$ for
a number of subjects, indexed $j = 1, \dots, N$. Then the likelihood for
$y_j$ is $p(y_j \mid m, \theta, x_j) = m_\theta(y_j \mid x_j)$. If we
also have observations $y'$ from an experimental setting modeled by
intervention $i$, then the likelihood is
$p(y'_j \mid m, \theta, x_j) = \text{intervene}(m_\theta, i)(y'_j \mid x_j)$.

Having expressed a prior and a likelihood, posterior inference can
recover causal structures $m$ and parameters $\theta$. Causal effects
and counterfactuals can be estimated by introducing additional variables
representing hypothetical *potential outcomes*. Such constructions might
usefully be automated by probabilistic programming languages, at which
point existing PPL inference machinery could be applied to estimating
the posterior.

+++

## Observation 3: causal uncertainty is probabilistic uncertainty

Bayesian causal inference places identifiability on a principled continuum of irreducible causal uncertainty.

On the surface, to claim that causal reasoning can be encapsulated by
probabilistic computation appears to be in direct conflict with Pearl's
insistence that causal and statistical concepts be kept
separate {cite:p}`pearl2001bayesian`. As Pearl describes them, statistical
concepts are those that summarize the distribution over observed
variables.

The probabilistic computations that we discuss in this
documentation are different in-kind from these assumption-free
summaries of data, in that we aim compute to probabilities of *latent*
causal structure, effects, and counterfactuals. In our proposed
approach, causal probabilistic programs play the role of causal
assumptions, relating observations to the latent causal quantities of
interest.

Casting causal inference as a particular instantiation of probabilistic
inference does not change the reality that many causal conclusions
cannot be unambiguously identified from data, regardless of sample size.
How much of the mutual information between treatment and outcome is
attributable to latent confounding? Does A cause B or does B cause A? If
C were c, what would have happened to D? Answers to all of these
questions are often ambiguous. 

Surprisingly, most existing formulations
of causal inference avoid quantifying these uncertainties, instead
abandoning problems in which latent causal quantities cannot be uniquely
inferred from data.[^2] Instead, the probabilistic programming approach
we espouse here enables users to express their assumptions, compute the
resulting uncertainty, be it irreducible or not, and then make decisions
accordingly.

+++

## References 

```{bibliography}
:filter: docname in docnames
```
