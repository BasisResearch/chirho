"""
Design notes: Counterfactual semantics from interventions and possible worlds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can give different semantics to ``intervene`` and ``sample`` with
Pyro’s effect handler API:

.. code:: python

   class BaseCounterfactual(Messenger):

     def _pyro_sample(self, msg) -> NoneType:
       pass

     def _pyro_intervene(self, msg) -> NoneType:
       pass

A trivial example of overloading ``intervene`` statements to ignore
intervened values:

.. code:: python

   class Observational(BaseCounterfactual):

     def _pyro_intervene(self, msg):
       if not msg["done"]:
         obs, act = msg["args"]
         msg["value"] = obs
         msg["done"] = True

Another trivial example of a semantics for ``intervene``, this time of
ignoring ``sample`` statements that have been intervened on through our
query interface defined above:

.. code:: python

   class Interventional(BaseCounterfactual):

     def _pyro_sample(self, msg):
       if msg.get("is_intervened", False):
         msg["stop"] = True

     def _pyro_intervene(self, msg):
       if not msg["done"]:
         obs, act = msg["args"]
         msg["value"] = act
         msg["done"] = True

The first conceptually nontrivial example: single-world intervention
graph semantics of ``intervene`` statements:

.. code:: python

   class SingleWorldCounterfactual(BaseCounterfactual):

     def _pyro_intervene(self, msg):
       if not msg["done"]:
         obs, act = msg["args"]
         msg["value"] = act
         msg["done"] = True

This prototyping effort will be focused on implementing a twin-world
semantics, in which there is one factual world where no interventions
happen and one counterfactual world where all interventions happen.

As it turns out, representing this efficiently is fairly straightforward
using the ``plate`` primitive included in Pyro.

.. code:: python

   class TwinWorldCounterfactual(BaseCounterfactual):

     def __init__(self, dim: int):
       self.dim = dim
       self._plate = pyro.plate("_worlds", size=2, dim=self.dim)
       super().__init__()

     def _is_downstream(self, value: Union[Tensor, Distribution]) -> bool: ...
     def _is_plate_active(self) -> bool: ...

     def _pyro_intervene(self, msg):
       if not msg["done"]:
         obs, act = msg["args"]

         if self._is_downstream(obs) or self._is_downstream(act):
           # in case of nested interventions:
           # intervention replaces the observed value in the counterfactual world
           #   with the intervened value in the counterfactual world
           obs = torch.index_select(obs, self.dim, torch.tensor([0]))
           act = torch.index_select(act, self.dim, torch.tensor([-1]))

         msg["value"] = torch.cat([obs, act], dim=self.dim)
         msg["done"] = True

     def _pyro_sample(self, msg):
       if self._is_downstream(msg["fn"]) or self._is_downstream(msg["value"]) and not self._is_plate_active():
         msg["stop"] = True
         with self._plate:
           obs_mask = [True, self._is_downstream(msg["value"])]
           msg["value"] = pyro.sample(
             msg["name"],
             msg["fn"],
             obs=msg["value"] if msg["is_observed"] else None,
             obs_mask=torch.tensor(obs_mask).expand((2,) + (1,) * (-self.dim - 1))
           )
           msg["done"] = True

"""
