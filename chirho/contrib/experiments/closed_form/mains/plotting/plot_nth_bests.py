import matplotlib.pyplot as plt
from chirho.contrib.experiments.closed_form.mains.utils import (
    load_metadata_from_dir
)
from chirho.contrib.experiments.closed_form.mains.plotting.utils import (
    get_matching_metadata,
    Result
)
import argparse
import chirho.contrib.experiments.closed_form as cfe
from copy import deepcopy
import os.path as osp
import numpy as np


def plot_nth_bests(dir_path: str, out_dir_path: str = None):

    if out_dir_path is None:
        out_dir_path = dir_path

    results = Result.load_results_from_dir(dir_path)

    # query = {
    #     "problem_setting_kwargs": {
    #         "n": 6,  # 3
    #         "rstar": 2.5,  # 4.5
    #         "seed": 21  # 22, 23, 24, 25
    #     },
    #     "hparam_consts": {
    #         "optimize_fn_name": "opt_with_mc_sgd",
    #         "unnorm_const": 1.0  # 0.1 10.0
    #     },
    # }

    all_zv_query = {
        "hparam_consts": {
            "optimize_fn_name": cfe.opt_with_zerovar_sgd.__name__
        },
    }
    all_zv = get_matching_metadata(metadatas_or_results=results, sub_metadata=all_zv_query)

    for zv in all_zv:
        # Now get all the results from different methods that sit in this exact same grid.
        query = deepcopy(zv.metadata)

        # First get the corresponding monte carlo result. This isn't indexed by unnorm_const.
        query["hparam_consts"]["optimize_fn_name"] = cfe.opt_with_mc_sgd.__name__
        mc = get_matching_metadata(metadatas_or_results=results, sub_metadata=query)[0]

        # Now get everything else.
        del query["hparam_consts"]["optimize_fn_name"]
        del query["hparam_consts"]["unnorm_const"]  # mc and zv aren't affected by normalization.

        matching_results = get_matching_metadata(metadatas_or_results=results, sub_metadata=query)

        # Now we have to group these by their unnorm_const, which isn't specified by mc or zv.
        grouped_results = dict()
        for mr in matching_results:
            # While we're at it, filter out the mc and zv.
            if mr is mc or mr is zv:
                continue
            unnorm_const = mr.metadata["hparam_consts"]["unnorm_const"]
            if unnorm_const not in grouped_results:
                grouped_results[unnorm_const] = []
            grouped_results[unnorm_const].append(mr)

        # Start extracting losses.
        zv_loss = zv.get_best_trial()["loss"]
        mc_loss = mc.get_best_trial()["loss"]

        # Get titling stuff.
        n = zv.metadata["problem_setting_kwargs"]["n"]
        rstar = zv.metadata["problem_setting_kwargs"]["rstar"]
        seed = zv.metadata["problem_setting_kwargs"]["seed"]

        # Now, for each unnorm_const, we want to plot the best result.
        for unnorm_const, results_to_plot in grouped_results.items():
            # Plot two panels, one for the whole loss plot, and one for the last 500 of each method.
            f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), dpi=300)

            lw = 0.8

            # Include all relevant stuff in the title.
            title = f"UnC: {unnorm_const}; N: {n}; R*: {rstar}; S: {seed}"
            ax1.set_title(title)

            # Plot the zv and mc losses.
            ax1.plot(zv_loss, label="ZeroVar", color="red", linewidth=lw)
            ax2.plot(np.array(zv_loss)[-500:], label="ZeroVar", color="red", linewidth=lw)
            ax1.plot(mc_loss, label="MonteCarlo", color="blue", linewidth=lw)
            ax2.plot(np.array(mc_loss)[-500:], label="MonteCarlo", color="blue", linewidth=lw)

            # Plot horizontal line for scipy optimal.
            ax1.axhline(zv.scipy_optimal['opt'], color='black', linestyle='--', label='Minimum')
            ax2.axhline(zv.scipy_optimal['opt'], color='black', linestyle='--', label='Minimum')

            # Order the results_to_plot alphabetically by their method name.
            results_to_plot = sorted(results_to_plot, key=lambda r: r.get_short_method_name())
            colors = plt.cm.viridis(np.linspace(0, 1, len(results_to_plot)))
            # Iterate through the other results and plot their losses.
            for color, result in zip(colors, results_to_plot):
                best_trial = result.get_best_trial()
                ax1.plot(best_trial["loss"], label=result.get_short_method_name(), linewidth=lw, color=color)
                # Don't plot PAIS because it's too biased.
                if result.metadata["hparam_consts"]["optimize_fn_name"] == cfe.opt_with_pais_sgd.__name__:
                    continue
                ax2.plot(np.array(best_trial["loss"])[-500:], label=result.get_short_method_name(),
                         linewidth=lw, color=color)

            ax1.legend()
            plt.tight_layout()

            # Save the plot.
            out_name = f"n{n}_rstar{rstar}_seed{seed}_unc{unnorm_const}.png"
            out_path = osp.join(out_dir_path, out_name)
            plt.savefig(out_path)

            plt.close(f)


def main(dir_path: str):
    plot_nth_bests(dir_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    args = parser.parse_args()
    main(args.path)
