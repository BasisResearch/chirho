# Causal Probabilistic Programming with Causal Pyro

Welcome! On this page you'll find a collection of tutorials and examples
on Causal Pyro, a causal extension to the Pyro probabilistic programming
language. Our intention is to make these tutorials accesible for a broad
technical audience, spanning from computer science researchers to
(computationally literate) domain experts. If you find anything unclear,
please ask questions and join the conversation on our forum (TODO).

## Motivation

Causal Pyro was built to bridge the gap between the capabilities of
modern probablistic programming systems, such as Pyro, and the needs of
policymakers, scientists, and AI researchers, who often want to use
models to answer their questions about cause and effect relationships.
As a non-exhaustive set of examples, Causal Pyro makes it easier to
answer the following kinds of causal questions that appear frequently in
practice.

-   **Interventional** : "How many covid-19 hospitalizations will occur
    if the the USA imposes a national mask mandate?"
-   **Counterfactual** : "Given that 100,000 people were infected with
    covid-19 in the past month, how many would have been infected if a
    mask mandate had been in place?"
-   **Explanation** : "Why were 100,000 people infected with covid-19 in
    the past month?"
-   **Causal Structure Discovery** : "What individual attributes
    influence risk of covid-19 hospitalization?"

Importantly, Causal Pyro does not answer these kinds of questions by
magic. In fact, there is no escaping the fact that "behind any causal
conclusion there must lie some causal assumption", a phrase made famous
by Judea Pearl (TODO: CITE). Instead, Causal Pyro provides a substrate
for writing causal assumptions as probabilistic programs, and for
writing causal questions in terms of program transformations. To
understand this in a bit more detail, see the following in-depth
tutorials describes Causal Pyro's underlying machinery.

Note: These tutorials assume some familiarity with Pyro and
probabilistic programming. For introductory Pyro tutorials please see
"Additional background reading material" below.

<div class="toctree" maxdepth="2" caption="Tutorials">

tutorial_i

</div>

## Example applications

To illustrate the utility of this approach, we have included several
examples from the causal inference literature.

We have tried to choose simple examples that would be of interest to
both the causal inference and probabilistic programming communities:
they collectively span Pearl’s causal hierarchy \[@pearl2001bayesian\],
and most are broadly applicable, empirically validated, have an
unconventional or limited identification result, and make use of modern
probabilistic machine learning tools, like neural networks or stochastic
variational inference.

Our examples demonstrate how real-world causal assumptions can be
expressed as probabilistic programs and real-world causal estimands can
be expressed as program transformations. These example illustrate how
Causal Pyro is compatible with any inference method implemented in Pyro,
including the kinds of scalable gradient-based approximations that power
much of the modern probabilistic machine learning landscape.

<div class="toctree" maxdepth="2" caption="Examples">

backdoor cevae deepscm slc mediation

</div>

<div class="toctree" maxdepth="2" caption="Documentation">

counterfactual interventional observational indexed

</div>

<div class="toctree" maxdepth="2" caption="Design Notes">

design_notes/index design_notes/counterfactuals
design_notes/interventions design_notes/queries
design_notes/observations

</div>

## Additional background reading material

-   Causal Probabilistic Programming Without Tears
    <https://drive.google.com/file/d/1Uzjg-vX77BdSnAcfpUcb-aIXxhnAPI24/view?usp=sharing>
-   Introduction to Pyro: <http://pyro.ai/examples/intro_long.html>
-   Tensor shapes in Pyro: <http://pyro.ai/examples/tensor_shapes.html>
-   A guide to programming with effect handlers in
    Pyro <http://pyro.ai/examples/effect_handlers.html>
-   Minipyro: <http://pyro.ai/examples/minipyro.html>
-   Reparameterization of Pyro
    programs: <https://docs.pyro.ai/en/stable/infer.reparam.html>
-   Optional: getting started with
    NumPyro <https://num.pyro.ai/en/stable/getting_started.html>

# Indices and tables

-   `genindex`
-   `modindex`
-   `search`
