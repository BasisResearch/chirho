def test_edge_eq_neq():

    def model_independent():
        X = pyro.sample("X", dist.Bernoulli(0.5))
        Y = pyro.sample("Y", dist.Bernoulli(0.5))

    def model_connected():
        X = pyro.sample("X", dist.Bernoulli(0.5))
        Y = pyro.sample("Y", dist.Bernoulli(X))

    with ExtractSupports() as supports_independent:
        model_independent()

    with ExtractSupports() as supports_connected:
        model_connected()

    with MultiWorldCounterfactual() as mwc_independent:  
            with SearchForExplanation(
                supports=supports_independent.supports,
                antecedents={"X": torch.tensor(1.0)},
                consequents={"Y": torch.tensor(1.0)},
                witnesses={},
                alternatives={"X": torch.tensor(0.0)},
                antecedent_bias=-0.5,
                consequent_scale=0,
            ):
                with pyro.plate("sample", size=3):
                    with pyro.poutine.trace() as trace_independent:
                        model_independent()

    with MultiWorldCounterfactual() as mwc_connected:  
            with SearchForExplanation(
                supports=supports_connected.supports,
                antecedents={"X": torch.tensor(1.0)},
                consequents={"Y": torch.tensor(1.0)},
                witnesses={},
                alternatives={"X": torch.tensor(0.0)},
                antecedent_bias=-0.5,
                consequent_scale=0,
            ):
                with pyro.plate("sample", size=3):
                    with pyro.poutine.trace() as trace_connected:
                        model_connected()


    trace_connected.trace.compute_log_prob
    trace_independent.trace.compute_log_prob

    assert trace_independent.trace.nodes["__cause____consequent_Y"]["fn"].log_factor[1,0,0,0,:].sum() <0
    assert torch.all(trace_independent.trace.nodes["__cause____consequent_Y"]["fn"].log_factor[2,0,0,0,:] == 0)
    assert torch.all(trace_connected.trace.nodes["__cause____consequent_Y"]["fn"].log_factor[0,0,0,0,:] == 0)