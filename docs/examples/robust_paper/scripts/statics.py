import pyro.distributions as dist

MODELS = ["CausalGLM", "kernel_ridge", "neural_network"]
LINK_FUNCTIONS_DICT = {
    "normal": lambda mu: dist.Normal(mu, 1.0),
    "bernoulli": lambda mu: dist.Bernoulli(logits=mu),
}
EXPERIMENT_CATEGORIES = ["influence_approx", "estimator_approx", "capstone"]
DATASET_PATHS = [f"docs/examples/robust_paper/datasets/{i}" for i in range(1, 5)]
FUNCTIONALS = ["ATE", "ESD", "CATE"]
ESTIMATORS = ["plug_in", "tmle", "one_step", "double_ml"]
INFLUENCE_ESTIMATORS = ["monte_carlo", "analytical", "finite_difference"]
