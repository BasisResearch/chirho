import functools
import pickle
from chirho.robust.handlers.predictive import PredictiveModel
from chirho.robust.handlers.estimators import one_step_corrected_estimator
from docs.examples.robust_paper.scripts.statics import (
    LINK_FUNCTIONS_DICT,
    FUNCTIONALS_DICT,
    MODELS,
    ALL_DATA_CONFIGS,
    ALL_EXP_CONFIGS,
)
from docs.examples.robust_paper.utils import get_mle_params_and_guide
from docs.examples.robust_paper.analytic_eif import (
    analytic_eif_expected_density,
    analytic_eif_ate_causal_glm,
)


def run_experiment(exp_config):
    data_config = ALL_DATA_CONFIGS[exp_config["data_uuid"]]
    data = pickle.load(
        open(
            f"docs/examples/robust_paper/datasets/{exp_config['data_uuid']}/data.pkl",
            "rb",
        )
    )
    D_train = data["train"]
    D_test = data["test"]

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

    functional_class = FUNCTIONALS_DICT[exp_config["functional_str"]]
    functional = functools.partial(functional_class, **exp_config["functional_kwargs"])

    one_step_estimator = one_step_corrected_estimator(
        functional, D_test, **exp_config["monte_carlo_influence_estimator_kwargs"]
    )

    # Run inference
    theta_hat, mle_guide = get_mle_params_and_guide(conditioned_model)
    plug_in_est = functional(PredictiveModel(model, mle_guide))()
    automated_monte_carlo_estimator = one_step_estimator(
        PredictiveModel(model, mle_guide)
    )()

    if model_str == "CausalGLM":
        # TODO
        pass
    elif model_str == "MultivariateNormalModel":
        analytic_correction, analytic_eif_at_test_pts = analytic_eif_expected_density(
            D_test, plug_in_est, PredictiveModel(model, mle_guide)
        )
    else:
        raise NotImplementedError

    # Save results
    results = {
        "theta_hat": theta_hat,
        "automated_monte_carlo_estimator": automated_monte_carlo_estimator,
        "analytic_correction": analytic_correction,
        "analytic_eif_at_test_pts": analytic_eif_at_test_pts,
        "analytic_one_step_estimator": analytic_correction + plug_in_est,
        "plug_in_est": plug_in_est,
        "experiment_uuid": exp_config["experiment_uuid"],
    }


if __name__ == "__main__":
    # expected density
    exp_config1 = ALL_EXP_CONFIGS["b175a477-1b1a-581b-68b2-d374e292a8e7"]
    print(exp_config1)
    run_experiment(exp_config1)

    # exp_config2 = ""
