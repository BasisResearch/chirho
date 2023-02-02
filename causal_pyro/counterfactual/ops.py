from .index_set import IndexSet, indices_of, merge
from .worlds import IndexPlatesMessenger, add_indices


class MultiWorldInterventions(IndexPlatesMessenger):
    def _pyro_post_intervene(self, msg):
        obs, act = msg["args"][0], msg["value"]
        event_dim = msg["kwargs"].setdefault("event_dim", 0)
        if msg["name"] is None:
            msg["name"] = f"intervention_{self.first_available_dim}"
        name = msg["name"]

        obs_indices = IndexSet.join(IndexSet(**{name: {0}}), indices_of(obs, event_dim=event_dim))
        act_indices = IndexSet.join(IndexSet(**{name: {1}}), indices_of(act, event_dim=event_dim))

        add_indices(IndexSet.join(obs_indices, act_indices))

        msg["value"] = merge({obs_indices: obs, act_indices: act}, event_dim=event_dim)
