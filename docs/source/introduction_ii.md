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

# Classical causal inference: a brief overview

+++

## Background: Structural causal models

Underlying Pearl's causal hierarchy is a mathematical object known as a
structural causal model (SCM).

We refer the reader to Chapter 7 of *Causality* {cite:p}`pearl` for complete
mathematical details, including the notation we introduce in this
section. We show how SCM's can be represented in deterministic and
probabilistic programming languages.

$M = <\mathbf{U}, \mathbf{V}, \mathcal{F}>$ denotes a fully-specified
deterministic structural causal model (SCM).
$\mathbf{U}=\{U_1,U_2,\ldots,U_m\}$ represents a set of exogenous
(unobserved) variables that are determined by factors outside the model.
$\mathbf{V}=\{V_1,V_2,\ldots,V_n\}$ denotes a set of endogenous
variables that are determined by other variables
$\mathbf{U}\cup\mathbf{V}$ in the model. Associated with each endogenous
variable $V_i$ is a function
$f_i: \mathbf{U}_i\cup \mathbf{Pa}_i\to V_i$ that assigns a value
$v_i \leftarrow f_i(pa_i,u_i)$ to $V_i$ that depends                                     on the values of
the deterministic parents
$\mathbf{Pa}_i \subset \mathbf{V}\backslash V_i$ and a set of exogenous
variables $\mathbf{U}_i\subset \mathbf{U}$. Deterministic programming
languages are capable of representing deterministic SCMs.

The entire set of functions $\mathcal{F} = \{f_1,f_2,\ldots, f_n\}$
forms a mapping from $\mathbf{U}$ to $\mathbf{V}$. That is, the values
of the exogenous variables uniquely determine the values of the
endogenous variables.

Every SCM $M$ can be associated with a directed graph, $G(M)$, in which
each node corresponds to a variable and the directed edges point from
members of the parents $\mathbf{Pa}_i$ and $\mathbf{U}_i$ toward $V_i$.
Static analysis can be used to derive the dependency graph of
deterministic programs{cite:p}`koppel_2020`.

Let $X$ be a variable in $\mathbf{V}$, and $x$ a particular value of
$X$. We define the effect of an **intervention** $do(X=x)$ on an SCM $M$
as a submodel
$M_{do(X=x)} = \left<\mathbf{U}, \mathbf{V}, \mathcal{F}_{do(X)}\right>$
where $\mathcal{F}_{do(X)}$ is formed by retracting from $\mathcal{F}$
the function $f_X$ corresponding to $X$ and assigning $X$ a constant
value $X=x$. Intervention in programming languages can be represented as
a program transformation, as implemented in Pyro {cite:p}`bingham2018pyro`.

Let $X$ and $Y$ be two variables in $\mathbf{V}$. The **potential
outcome** of $Y$ to action $do(X=x)$, denoted $Y_{do(X=x)}$ is the
solution for $Y$ from the set of equations $\mathcal{F}_{do(X=x)}$. That
is, $Y_{do(X=x)}=Y_{M_{do(X=x)}}$.

Higher order interventions, such as $do(x=g(z))$ can be represented by
the replacement of equations by functions instead of constants, as
implemented in Omega  {cite:p}`tavares_2020`.

One can compute a counterfactual using a graphical approach known as the
**twin network method** {cite:p}`balke_1994`. It uses two graphs, one to
represent the factual world, and one to represent the counterfactual
world. Bayesian implementations of twin world networks are described
in {cite:p}`lattimore_2019`.

A fully-specified probabilistic structural causal model is a pair
$\left<M, P(\mathbf{U})\right>$, where $M$ is a fully-specified
deterministic structural causal model and $P(\mathbf{U})$ is a
probability function defined over the domain of $\mathbf{U}$.
Probabilistic structural causal models can be represented using
probabilistic programming languages,

Given a probabilistic SCM $\left<M, P(\mathbf{U})\right>$ the
conditional probability of a counterfactual sentence can be evaluated
using the following three steps:

Abduction:

:   Update $P(\mathbf{U})$ by the evidence $E=e$ to obtain
    $P(\mathbf{U}|E=e)$.

Action:

:   Modify $M$ by the intervention $do(X=x)$ to obtain the submodel
    $M_{do(X=x)}$.

Prediction:

:   Use the modified model $\left<M_X, P(\mathbf{U}|E=e)\right>$ to
    compute the probability of $Y_{do(X=x)}$, the potential outcome of
    the counterfactual.

+++

## Background: Classical causal inference with Pearl's do-calculus in Pyro

The do-calculus consists of 3 rules, and the second one applies in this
situation. It says that if one can stratify the observational
distribution by all these confounding factors, then what remains is the
true causal effect. That is,

$$\begin{aligned} \\
\sum_{X=x} P(Y=y|{\color{red}T=t},X=x)P(X=x) &\overset{i}{=}& \sum_{X=x} P(Y=y|{\color{red}do(T=t)},X=x)P(X=x) \\ &=& P(Y=y|do(T=t)) \end{aligned}$$

The replacement of $T=t$ in the
first expression with $do(T=t)$ in the second expression is licensed by
Rule 2 of the do-calculus.

+++


### Specifying and estimating the do calculus problem

For simplicity, let's just consider only one confounder, say political
affiliation. Let `theta` be an array of six learnable parameters. A
program that represents this situation could be as follows:

    def causal_model(theta):
       X ~ bernoulli(theta[0])
       T ~ bernoulli(theta[X+1])
       Y ~ bernoulli(theta[T+2*X+3])
       return Y

The classical causal inference approach would be to extract the causal
diagram from the model, identify the causal effect, and then estimate it
from data:

    >>> causal_graph = pyro.infer.inspect.get_dependencies(causal_model)
    >>> estimand = identify(P(Y|do(T)), causal_graph)
    >>> estimand

$$P(Y|do(T)) = \sum_X P(Y|T,X)P(X)$$

    >>> P_of_covid_given_do_vaccine = estimate(causal_model, estimand, data)
    >>> P_of_covid_given_do_vaccine
    {'covid-positive':0.00534, 'covid-negative': 0.99466}

The `get_dependencies` function takes a model as input. It then
performs a static analysis of the dependency structure to generate a
causal diagram.

The `identify()` function takes as input a causal
diagram and a causal query, represented as a symbolic probabilistic
expression. It then applies the do calculus to the diagram to identify
the query. If the causal query is identified, it will return an
estimand, represented as a symbolic probabilistic expression composed of
nested conditionals and marginals. If the query is not identified, it
will raise an exception.

The `estimate()` procedure takes a causal
model, an estimand and a dataframe containing measurements of the
observed variables as input. It then applies the estimand to the dataset
to generate an estimate of the original causal query.


+++

## Background: Causal discovery

Consider the task of *learning* a causal model from some class of models
$\mathcal{M}$, based on observational data $y_{1, \dots, n}$ and
experimental data from $E$ different experimental settings,
$y^{E_i}_{1, \dots, n_i}$. Here, $y$ may be multivariate and models
$m \in \mathcal{M}$ may or may not posit additional latent variables
$x_i^m$ for each subject. We write $E_i(m)$ for the causal model
obtained by applying an intervention modeling experiment $E_i$ to the
observational causal model $m$.

In the Bayesian setting, the practitioner needs to place a prior over
causal models, $p(m)$. The likelihood is then
$p(y_{1\dots n}, y^{E_i}_{1\dots n_i} \mid m) = \left[\prod_{i=1}^n \int m(x_i^m, y_i) dx_i^m \right] \cdot \left[\prod_{j=1}^{E} \prod_{i=1}^{n_j} \int E_j(m)(x_i^{E_j(m)}, y_i^{E_j}) dx_i^{E_j(m)}\right]$.
{cite:p}`witty2020` show that both the prior over causal models and the
likelihoods can be represented in a suitably expressive probabilistic
programming language.

### Embedded causal language for models $m \in \mathcal{M}$.

To represent the prior over causal models, {cite:p}`witty2020` introduce a
restricted *causal* probabilistic programming language *MiniStan*; the
prior over causal models is then an ordinary (Gen) probabilistic program
`prior` that generates MiniStan syntax trees. They further develop a Gen
probabilistic program `interpret` that interprets the syntax of a
MiniStan program, sampling the variables it defines, as well as a
function `intervene` that applies a program transformation to the syntax
of a MiniStan program $m$ to yield an experimental model program
$E_i(m)$.

### Causal discovery as Bayesian inference.

Having defined these helper functions, {cite:p}`witty2020` frame the entire
causal discovery problem as inference in the following program:

    def causal_discovery():
      # Generate a possible true model from the prior
      m ~ prior()
      
      # Generate observational data
      for i in range(n):
        y[i] ~ interpret(m)

      # For each experiment, generate experimental data
      for j in range(E):
        m_intervened = intervene(m, interventions[j])
        for i in range(n_experimental[j]):
          y_experimental[j][i] ~ interpret(m_intervened)

Using Gen's programmable inference, {cite:p}`witty2020` develop a sequential
Monte Carlo algorithm that incorporates one observation from each
experiment at each time step, inferring any latent variables posited by
the model $m$ or its intervened versions. Other inference algorithms
could also be applied. The result is a posterior over models.

+++

## References

```{bibliography}
:filter: docname in docnames
```
