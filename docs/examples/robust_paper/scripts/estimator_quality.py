import functools
import time
import torch
import pickle
from chirho.robust.handlers.predictive import PredictiveModel
from chirho.robust.handlers.estimators import one_step_corrected_estimator
from chirho.robust.ops import influence_fn
from docs.examples.robust_paper.scripts.statics import (
    LINK_FUNCTIONS_DICT,
    FUNCTIONALS_DICT,
    MODELS,
    ALL_EXP_CONFIGS,
)
from docs.examples.robust_paper.utils import get_mle_params_and_guide, MLEGuide
from docs.examples.robust_paper.analytic_eif import (
    analytic_eif_expected_density,
    analytic_eif_ate_causal_glm,
)


def run_experiment_one_step(exp_config):
    assert exp_config["estimator_str"] == "one_step"
    # Results dict
    results = dict()
    results["experiment_uuid"] = exp_config["experiment_uuid"]

    # Load in data
    data_config = exp_config["data_config"]
    true_treatment_weight = data_config["misc"]["treatment_weight"]
    results["treatment_weight"] = true_treatment_weight

    data = pickle.load(
        open(
            f"docs/examples/robust_paper/datasets/{exp_config['data_uuid']}/data.pkl",
            "rb",
        )
    )
    D_train = data["train"]
    D_test = data["test"]

    print(f"=== Running experiment {exp_config['experiment_uuid']} ===")

    # Load in model
    model_str = data_config["model_configs"]["model_str"]
    if model_str == "MultivariateNormalModel":
        model_kwargs = {
            "p": data_config["model_configs"]["p"],
        }
    elif model_str == "CausalGLM":
        model_kwargs = {
            "p": data_config["model_configs"]["p"],
            "link_fn": LINK_FUNCTIONS_DICT[
                data_config["model_configs"]["link_function_str"]
            ],
        }
    else:
        raise NotImplementedError
    model = MODELS[model_str]["model"](**model_kwargs)
    conditioned_model = MODELS[model_str]["conditioned_model"](D_train, **model_kwargs)

    # Load in functional
    functional_class = FUNCTIONALS_DICT[exp_config["functional_str"]]
    functional = functools.partial(functional_class, **exp_config["functional_kwargs"])

    # Fit MLE
    mle_start_time = time.time()
    theta_hat, mle_guide = get_mle_params_and_guide(conditioned_model)
    mle_end_time = time.time()
    mle_time_min = (mle_end_time - mle_start_time) / 60.0
    results["theta_hat"] = theta_hat
    results["mle_time_min"] = mle_time_min

    # Get plug-in estimate
    plug_in_start_time = time.time()
    plug_in_est = functional(PredictiveModel(model, mle_guide))()
    plug_in_end_time = time.time()
    plug_in_time_min = (plug_in_end_time - plug_in_start_time) / 60.0
    results["plug_in_est"] = plug_in_est
    results["plug_in_time_min"] = plug_in_time_min

    #### Monte Carlo EIF ####
    monte_eif_all_kwargs = exp_config["monte_carlo_influence_estimator_kwargs"]
    num_samples_inner = monte_eif_all_kwargs["num_samples_inner"]
    cg_iters = monte_eif_all_kwargs["cg_iters"]
    residual_tol = monte_eif_all_kwargs["residual_tol"]
    num_monte_carlo_outer = monte_eif_all_kwargs["num_samples_outer"]
    all_monte_carlo_eif_results = []

    monte_carlo_eif_results = dict()
    # Hack to avoid https://github.com/BasisResearch/chirho/issues/483
    theta_hat = {
        k: v.clone().detach().requires_grad_(True) for k, v in theta_hat.items()
    }
    mle_guide = MLEGuide(theta_hat)

    print(f"Running monte carlo eif with {num_monte_carlo_outer} samples")
    monte_carlo_eif_results["num_monte_carlo_outer"] = num_monte_carlo_outer
    monte_kwargs = {
        "num_samples_outer": num_monte_carlo_outer,
        "num_samples_inner": num_samples_inner,
        "cg_iters": cg_iters,
        "residual_tol": residual_tol,
    }

    # One step estimator
    monte_one_step_start = time.time()
    one_step_estimator = one_step_corrected_estimator(
        functional, D_test, **monte_kwargs
    )
    automated_monte_carlo_estimate = one_step_estimator(
        PredictiveModel(model, mle_guide)
    )()
    monte_one_step_end = time.time()
    monte_one_step_time_min = (monte_one_step_end - monte_one_step_start) / 60.0

    monte_carlo_eif_results["wall_time"] = monte_one_step_time_min
    monte_carlo_eif_results["correction"] = automated_monte_carlo_estimate - plug_in_est
    monte_carlo_eif_results["corrected_estimate"] = automated_monte_carlo_estimate

    results["all_monte_carlo_eif_results"] = all_monte_carlo_eif_results

    ### Analytic EIF ###
    if model_str == "CausalGLM":
        analytic_time_start = time.time()
        analytic_correction, analytic_eif_at_test_pts = analytic_eif_ate_causal_glm(
            D_test, theta_hat
        )
        analytic_time_end = time.time()
        analytic_time_min = (analytic_time_end - analytic_time_start) / 60.0
    elif model_str == "MultivariateNormalModel":
        analytic_time_start = time.time()
        analytic_correction, analytic_eif_at_test_pts = analytic_eif_expected_density(
            D_test, plug_in_est, PredictiveModel(model, mle_guide)
        )
        analytic_time_end = time.time()
        analytic_time_min = (analytic_time_end - analytic_time_start) / 60.0
    else:
        raise NotImplementedError

    # Can't pickle _to_functional_tensor so convert to vanilla torch tensor
    analytic_eif_results = dict()
    analytic_eif_results["wall_time"] = analytic_time_min
    analytic_eif_results["correction"] = torch.tensor(analytic_correction.item())
    analytic_eif_results["corrected_estimate"] = (
        plug_in_est + analytic_correction.item()
    )
    analytic_eif_results["pointwise_wall_time"] = analytic_time_min
    analytic_eif_results["pointwise_eif"] = torch.tensor(
        [e.item() for e in analytic_eif_at_test_pts]
    )
    results["analytic_eif_results"] = analytic_eif_results

    # Save results
    pickle.dump(
        results,
        open(
            f"docs/examples/robust_paper/experiments/{exp_config['experiment_uuid']}/results.pkl",
            "wb",
        ),
    )
    return results


def run_experiment(exp_config):
    if exp_config["estimator_str"] == "one_step":
        return run_experiment_one_step(exp_config)
