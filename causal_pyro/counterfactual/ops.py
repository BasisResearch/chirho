from .index_set import IndexSet, gather, indices_of, scatter
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

        obs = gather(obs, obs_indices, event_dim=event_dim)
        act = gather(act, act_indices, event_dim=event_dim)

        msg["value"] = scatter(obs, obs_indices, result=act, event_dim=event_dim)
