A primary motivation for working in a PPL is separation of concerns
between models, queries, and inference. The design for causal inference
presented so far is highly modular, but up to now we have been
interleaving causal models and interventions, which is not ideal when we
might wish to compute multiple different counterfactual quantities using
the same model.

Fortunately, effect handlers make it easy to implement a basic query
interface for entire models rather than individual values that builds on
all of the machinery discussed so far. This interface should transform
models to models, so that query operators can be composed and queries
can be answered with any Pyro posterior inference algorithm.

.. code:: python

   Query = Callable[[Callable[A, B]], Callable[A, B]]

Here is a sketch for intervention queries on random variables
(``pyro.sample`` statements) essentially identical semantically to the
one built into Pyro. Note that it is entirely
separate from and compatible with any of the
counterfactual semantics in ``chirho.query.counterfactual``.

.. code:: python

   class do(Generic[T], Messenger):
     def __init__(self, actions: dict[str, Intervention[T]]):
       self.actions = actions
       super().__init__()

     def _pyro_sample(self, msg):
       msg["is_intervened"]: bool = msg.get("is_intervened", (msg["name"] in self.actions))

     def _pyro_post_sample(self, msg):
       msg["value"] = intervene(msg["value"], self.actions.get(msg["name"], None))

   def model():
     ...

   intervened_model = do(actions={...})(model)

We might also wish to expose the contents of the previous section on
reparameterization as part of a comprehensive
``pyro.infer.reparam.Strategy`` for counterfactual inference that
automatically applies local transformations specialized to specific
distribution and query types.

We can then define higher level causal query operators by composing
``do`` with other handlers like ``condition`` and ``reparam`` e.g.

.. code:: python

   def surrogate_counterfactual_scm(
     actions: dict[str, Intervention],
     data: dict[str, Tensor],
     strategy: Strategy
   ) -> Query[A, B]:

     def _query(model: Callable[A, B]) -> Callable[A, B]:
       return functools.wraps(model)(
         reparam(strategy)(
           condition(data)(
             do(actions)(
               model)))

     return _query