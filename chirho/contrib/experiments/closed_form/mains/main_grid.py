import itertools
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run grid experiments")
    # High-level arguments
    parser.add_argument("--verbose", type=int, help="Verbosity level")
    # See default in main_one.py
    parser.add_argument("--storage_path", type=str, default=None, help="Path to store results")
    # See default in main_one.py
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples for hyperparameter tuning")
    parser.add_argument("--max_concurrent_grid_cell_trials", type=int, default=1,
                        help="Maximum number of concurrent grid cell trials")
    parser.add_argument("--max_concurrent_hyperparam_trials", type=int,
                        help="Maximum number of concurrent hyperparameter tuning trials within an experiment")

    # Grid definition arguments with nargs='+' to accept multiple values
    parser.add_argument("--q", nargs='+', type=float, help="Size of the risk curve.")
    parser.add_argument("--n", nargs='+', type=int, help="Dimensionality of the problem.")
    parser.add_argument("--rstar", nargs='+', type=float,
                        help="How far into the tails the optimum sits, in standard deviations.")
    parser.add_argument("--theta0_rstar_delta", nargs='+', type=float,
                        help="Delta, in standard deviations to initialize the parameters.")
    parser.add_argument("--seed", nargs='+', type=int, help="Values for seed")
    parser.add_argument("--optimize_fn_name", nargs='+', type=str, help="Values for optimize_fn_name")
    parser.add_argument("--unnorm_const", nargs='+', type=float,
                        default=None, help="Values for unnorm_const")

    args = parser.parse_args()

    # Handling default values for optional arguments
    if args.unnorm_const is None:
        args.unnorm_const = [1.0]  # Default value for unnorm_const

    return args


def run_experiment(command):
    # Wrapper to execute each command as a subprocess
    print(f"Executing: {' '.join(command)}")
    subprocess.run(command)


def main():
    args = parse_arguments()

    # Path to the experiment script (main_one.py)
    experiment_script_path = os.path.join(os.path.dirname(__file__), 'main_one.py')

    # Generate the grid of all combinations from command-line arguments
    grid = itertools.product(
        args.q,
        args.n,
        args.rstar,
        args.theta0_rstar_delta,
        args.seed,
        args.optimize_fn_name,
        args.unnorm_const
    )

    # Prepare high-level arguments
    high_level_args = []
    if args.verbose is not None:
        high_level_args += ['--verbose', str(args.verbose)]
    if args.storage_path is not None:
        high_level_args += ['--storage_path', args.storage_path]
    if args.num_samples is not None:
        high_level_args += ['--num_samples', str(args.num_samples)]
    if args.max_concurrent_hyperparam_trials is not None:
        high_level_args += ['--max_concurrent_trials', str(args.max_concurrent_hyperparam_trials)]

    # List to store the commands
    commands = []
    for combination in grid:
        command = [
                      'python', experiment_script_path,
                      '--q', str(combination[0]),
                      '--n', str(combination[1]),
                      '--rstar', str(combination[2]),
                      '--theta0_rstar_delta', str(combination[3]),
                      '--seed', str(combination[4]),
                      '--optimize_fn_name', combination[5],
                      '--unnorm_const', str(combination[6])
                  ] + high_level_args

        commands.append(command)

    # # Execute commands with a limit on the number of concurrent grid cell trials
    # with ProcessPoolExecutor(max_workers=args.max_concurrent_grid_cell_trials) as executor:
    #     futures = [executor.submit(run_experiment, command) for command in commands]
    #     for future in as_completed(futures):
    #         future.result()  # You can handle exceptions here if you want

    # Execute the commands serially.
    assert args.max_concurrent_grid_cell_trials == 1, "This is a serial execution now."
    for command in commands:
        run_experiment(command)


if __name__ == "__main__":
    main()
