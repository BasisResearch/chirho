from .index_set import IndexSet, gather, indices_of, scatter, merge
from .worlds import IndexPlatesMessenger, add_indices


class MultiWorldInterventions(IndexPlatesMessenger):
    def _pyro_post_intervene(self, msg):
        obs, act = msg["args"][0], msg["value"]
        event_dim = msg["kwargs"].setdefault("event_dim", 0)
        name = msg.setdefault("name", f"intervention_{self.first_available_dim}")

        obs_indices = IndexSet(**{name: {0}})
        act_indices = IndexSet.difference(
            indices_of(act, event_dim=event_dim), obs_indices
        )

        add_indices(IndexSet(**{name: set(range(max(act_indices[name])))}))

        msg["value"] = merge({obs_indices: obs, act_indices: act}, event_dim=event_dim)


class OnlyFactualConditioningReparam(pyro.infer.reparam.reparam.Reparam):

    def apply(self, msg):
        name = msg
        if not msg["is_observed"] or pyro.poutine.util.site_is_subsample(msg):
            return

        with OnlyFactual(prefix="factual") as fw:
            # TODO prevent unbounded recursion here
            fv = pyro.sample(msg["name"], msg["fn"], obs=msg["value"])

        with OnlyCounterfactual(prefix="counterfactual") as cw:
            cv = pyro.sample(msg["name"], msg["fn"])

        msg["value"] = merge({fw: fv, cw: cv}, event_dim=len(msg["fn"].event_shape))

        # emulate a deterministic statement
        msg["fn"] = pyro.distributions.Delta(
            msg["value"], event_dim=len(msg["fn"].event_shape)
        ).mask(False)
