from .index_set import IndexSet, indices_of, merge
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
