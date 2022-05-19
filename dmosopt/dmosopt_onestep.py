import os, sys, logging, gc, pprint
from functools import partial
import click
import numpy as np
from dmosopt.dmosopt import init_from_h5
from dmosopt.MOASMO import onestep

script_name = os.path.basename(__file__)


def list_find(f, lst):
    """

    :param f:
    :param lst:
    :return:
    """
    i = 0
    for x in lst:
        if f(x):
            return i
        else:
            i = i + 1
    return None


@click.command()
@click.option("--file-path", "-p", required=True, type=click.Path())
@click.option("--opt-id", required=True, type=str)
@click.option("--resample-fraction", required=True, type=float)
@click.option("--population-size", required=True, type=int)
@click.option("--num-generations", required=True, type=int)
@click.option("--verbose", "-v", is_flag=True)
def main(
    file_path, opt_id, resample_fraction, population_size, num_generations, verbose
):

    (
        old_evals,
        param_names,
        is_int,
        lo_bounds,
        hi_bounds,
        objective_names,
        feature_names,
        constraint_names,
        problem_parameters,
        problem_ids,
    ) = init_from_h5(file_path, None, opt_id, None)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(opt_id)

    if problem_ids is None:
        problem_ids = [0]
    for problem_id in problem_ids:
        old_eval_xs = [e.parameters for e in old_evals[problem_id]]
        old_eval_ys = [e.objectives for e in old_evals[problem_id]]
        x = np.vstack(old_eval_xs)
        y = np.vstack(old_eval_ys)
        old_eval_fs = None
        f = None
        if feature_names is not None:
            old_eval_fs = [e.features for e in old_evals[problem_id]]
            f = np.concatenate(old_eval_fs, axis=None)
        old_eval_cs = None
        c = None
        if constraint_names is not None:
            old_eval_cs = [e.constraints for e in old_evals[problem_id]]
            c = np.vstack(old_eval_cs)
        x = np.vstack(old_eval_xs)
        y = np.vstack(old_eval_ys)

        print(f"Restored {x.shape[0]} solutions")
        sys.stdout.flush()

        n_dim = len(lo_bounds)
        n_objectives = len(objective_names)

        x_resample, _ = onestep(
            n_dim,
            n_objectives,
            np.asarray(lo_bounds),
            np.asarray(hi_bounds),
            resample_fraction,
            x,
            y,
            C=c,
            pop=population_size,
            optimizer_kwargs={
                "gen": num_generations,
                "crossover_rate": 0.9,
                "mutation_rate": None,
                "mu": 20,
                "mum": 20,
            },
            logger=logger,
        )

        for i, x in enumerate(x_resample):
            print(f"resampled coordinates {i}: {pprint.pformat(x)}")


if __name__ == "__main__":
    main(
        args=sys.argv[
            (list_find(lambda x: os.path.basename(x) == script_name, sys.argv) + 1) :
        ]
    )
