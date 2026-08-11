import os
import logging
import time
import itertools
from functools import partial
from collections.abc import Iterator, Sequence
from typing import Union, Dict, List, Optional
from types import GeneratorType
import numpy as np
from numpy.random import default_rng
import distwq
from dmosopt.config import import_object_by_path
from dmosopt import MOEA
import dmosopt.MOASMO as opt
from dmosopt.datatypes import (
    OptProblem,
    ParameterSpace,
    EvalEntry,
    EvalRequest,
    EpochResults,
    StrategyState,
    update_nested_dict,
)
from dmosopt.adaptive_termination import create_adaptive_termination

logger = logging.getLogger("dmosopt")

try:
    import h5py
except ImportError as e:
    logger.warning(f"unable to import h5py: {e}")

dopt_dict = {}


def anyclose(a, b, rtol=1e-4, atol=1e-4):
    for i in range(b.shape[0]):
        if np.allclose(a, b[i, :]):
            return True
    return False


class DistOptStrategy:
    def __init__(
        self,
        prob: OptProblem,
        n_initial: int = 10,
        initial=None,
        initial_maxiter: int = 5,
        initial_method: str = "slh",
        population_size: int = 100,
        resample_fraction: float = 0.25,
        num_generations: int = 100,
        surrogate_method_name: str = "gpr",
        surrogate_method_kwargs: Dict[str, Union[bool, str]] = {
            "anisotropic": False,
            "optimizer": "sceua",
        },
        surrogate_custom_training: Optional[str] = None,
        surrogate_custom_training_kwargs: Optional[Dict] = None,
        sensitivity_method_name: Optional[str] = None,
        sensitivity_method_kwargs={},
        distance_metric=None,
        optimizer_name: Union[str, Sequence[str]] = "nsga2",
        optimizer_kwargs: Union[Dict, Sequence[Dict]] = {
            "crossover_prob": 0.9,
            "mutation_prob": 0.1,
        },
        feasibility_method_name=None,
        feasibility_method_kwargs={},
        termination_conditions=None,
        optimize_mean_variance=False,
        local_random=None,
        logger=None,
        file_path=None,
    ):
        if local_random is None:
            local_random = default_rng()
        self.local_random = local_random
        self.logger = logger
        self.file_path = file_path
        self.feasibility_method_name = feasibility_method_name
        self.feasibility_method_kwargs = feasibility_method_kwargs
        self.surrogate_method_kwargs = surrogate_method_kwargs
        self.surrogate_method_name = surrogate_method_name
        self.surrogate_custom_training = surrogate_custom_training
        self.surrogate_custom_training_kwargs = surrogate_custom_training_kwargs
        self.sensitivity_method_kwargs = sensitivity_method_kwargs
        self.sensitivity_method_name = sensitivity_method_name
        self.optimizer_name = (
            optimizer_name
            if isinstance(optimizer_name, Sequence)
            and not isinstance(optimizer_name, str)
            else (optimizer_name,)
        )
        self.optimizer_kwargs = (
            optimizer_kwargs
            if isinstance(optimizer_kwargs, Sequence)
            else (optimizer_kwargs,)
        )
        self.optimize_mean_variance = optimize_mean_variance

        self.optimizer_iter = itertools.cycle(range(len(self.optimizer_name)))
        self.distance_metric = distance_metric
        self.prob = prob
        self.completed = []
        self.reqs = []
        self.t = None
        if initial is None:
            self.x = None
            self.y = None
            self.f = None
            self.c = None
        else:
            epochs, self.x, self.y, self.f, self.c = initial
        self.resample_fraction = resample_fraction
        self.num_generations = num_generations
        self.population_size = population_size
        self.termination = None
        if callable(termination_conditions):
            self.termination = termination_conditions(prob)
        elif termination_conditions:
            termination_kwargs = {
                "strategy": "comprehensive",
                "n_max_gen": num_generations,
            }
            if isinstance(termination_conditions, dict):
                termination_kwargs.update(termination_conditions)
            self.termination = create_adaptive_termination(prob, **termination_kwargs)
        nPrevious = None
        if self.x is not None:
            nPrevious = self.x.shape[0]
        xinit = opt.xinit(
            n_initial,
            prob.param_names,
            prob.lb,
            prob.ub,
            nPrevious=nPrevious,
            maxiter=initial_maxiter,
            method=initial_method,
            local_random=self.local_random,
            logger=self.logger,
        )
        self.reqs = []
        if xinit is not None:
            assert xinit.shape[1] == prob.dim
            if initial is None:
                self.reqs = [
                    EvalRequest(xinit[i, :], None, 0) for i in range(xinit.shape[0])
                ]
            else:
                self.reqs = filter(
                    lambda req: not anyclose(req.parameters, self.x),
                    [EvalRequest(xinit[i, :], None, 0) for i in range(xinit.shape[0])],
                )
        self.opt_gen = None
        self.epoch_index = -1

        self.stats = {}

    def append_request(self, req):
        if isinstance(self.reqs, Iterator):
            self.reqs = list(self.reqs)
        self.reqs.append(req)

    def has_requests(self):
        res = False
        if isinstance(self.reqs, Iterator):
            try:
                peek = next(self.reqs)
                self.reqs = itertools.chain([peek], self.reqs)
            except StopIteration:
                pass
        else:
            res = len(self.reqs) > 0
        return res

    def get_next_request(self):
        req = None
        if isinstance(self.reqs, Iterator):
            try:
                req = next(self.reqs)
            except StopIteration:
                pass
        else:
            try:
                req = self.reqs.pop(0)
            except Exception:
                pass
        return req

    def complete_request(self, x, y, epoch=None, f=None, c=None, pred=None, time=-1.0):
        assert x.shape[0] == self.prob.dim
        assert y.shape[0] == self.prob.n_objectives
        if self.optimize_mean_variance and pred is not None:
            if pred.shape[0] == self.prob.n_objectives:
                pred = np.column_stack((pred, np.zeros_like(pred)))
        if (f is not None) and (f.shape == 1):
            f = f.reshape((1, -1))
        entry = EvalEntry(epoch, x, y, f, c, pred, time)
        self.completed.append(entry)
        return entry

    def has_completed(self):
        return len(self.completed) > 0

    def _remove_duplicate_evals(self):
        is_duplicates = MOEA.get_duplicates(self.x)

        self.x = self.x[~is_duplicates]
        self.y = self.y[~is_duplicates]
        if self.f is not None:
            self.f = self.f[~is_duplicates]
        if self.c is not None:
            self.c = self.c[~is_duplicates]

    def _reduce_evals(self):
        self._remove_duplicate_evals()

        perm, _, _ = MOEA.orderMO(self.x, self.y)

        self.x = self.x[perm[0 : self.population_size], :]
        self.y = self.y[perm[0 : self.population_size], :]
        if self.c is not None:
            self.c = self.c[perm[0 : self.population_size], :]
        if self.f is not None:
            self.f = self.f[perm[0 : self.population_size]]

    def _update_evals(self):
        result = None

        if len(self.completed) > 0 and not self.has_requests():
            x_completed = np.vstack([x.parameters for x in self.completed])
            y_completed = np.vstack([x.objectives for x in self.completed])
            n_prediction = self.prob.n_objectives
            if self.optimize_mean_variance:
                n_prediction *= 2
            y_predicted = np.vstack(
                tuple(
                    map(
                        lambda x: [np.nan] * n_prediction if x is None else x,
                        [x.prediction for x in self.completed],
                    )
                )
            )

            f_completed = None
            if self.prob.n_features is not None:
                f_completed = np.concatenate(
                    [x.features for x in self.completed], axis=0
                )
            c_completed = None
            if self.prob.n_constraints is not None:
                c_completed = np.vstack([x.constraints for x in self.completed])

            assert x_completed.shape[1] == self.prob.dim
            assert y_completed.shape[1] == self.prob.n_objectives
            if self.prob.n_constraints is not None:
                assert c_completed.shape[1] == self.prob.n_constraints

            if self.x is None:
                self.x = x_completed
                self.y = y_completed
                self.f = f_completed
                self.c = c_completed
            else:
                self.x = np.vstack((self.x, x_completed))
                self.y = np.vstack((self.y, y_completed))

                if self.prob.n_features is not None:
                    self.f = np.concatenate((self.f, f_completed), axis=0)
                if self.prob.n_constraints is not None:
                    self.c = np.vstack((self.c, c_completed))

            t_completed = np.vstack([x.time for x in self.completed])
            if self.t is None:
                self.t = t_completed
            else:
                self.t = np.vstack((self.t, t_completed))
            ts = self.t[self.t > 0.0]
            if len(ts) > 0:
                self.stats.update(
                    {
                        "eval_min": np.min(ts),
                        "eval_max": np.max(ts),
                        "eval_mean": np.mean(ts),
                        "eval_std": np.std(ts),
                        "eval_sum": np.sum(ts),
                        "eval_median": np.median(ts),
                    }
                )
            else:
                self.stats.update(
                    {
                        "eval_min": -1,
                        "eval_max": -1,
                        "eval_mean": -1,
                        "eval_std": -1,
                        "eval_sum": -1,
                        "eval_median": -1,
                    }
                )

            self._remove_duplicate_evals()
            self.completed = []
            result = x_completed, y_completed, y_predicted, f_completed, c_completed

        return result

    def initialize_epoch(self, epoch_index):
        assert (
            self.opt_gen is None
        ), "Optimization generator is active in DistOptStrategy"

        optimizer_index = next(self.optimizer_iter)
        optimizer_kwargs = {}
        if self.optimizer_kwargs[optimizer_index] is not None:
            optimizer_kwargs.update(self.optimizer_kwargs[optimizer_index])
        if self.distance_metric is not None:
            optimizer_kwargs["distance_metric"] = self.distance_metric

        self._update_evals()

        assert epoch_index > self.epoch_index
        self.epoch_index = epoch_index
        self.opt_gen = opt.epoch(
            self.num_generations,
            self.prob.param_names,
            self.prob.objective_names,
            self.prob.lb,
            self.prob.ub,
            self.resample_fraction,
            self.x,
            self.y,
            self.c,
            pop=self.population_size,
            optimizer_name=self.optimizer_name[optimizer_index],
            optimizer_kwargs=optimizer_kwargs,
            surrogate_method_name=self.surrogate_method_name,
            surrogate_method_kwargs=self.surrogate_method_kwargs,
            surrogate_custom_training=self.surrogate_custom_training,
            surrogate_custom_training_kwargs=self.surrogate_custom_training_kwargs,
            sensitivity_method_name=self.sensitivity_method_name,
            sensitivity_method_kwargs=self.sensitivity_method_kwargs,
            feasibility_method_name=self.feasibility_method_name,
            feasibility_method_kwargs=self.feasibility_method_kwargs,
            optimize_mean_variance=self.optimize_mean_variance,
            termination=self.termination,
            local_random=self.local_random,
            logger=self.logger,
            file_path=self.file_path,
        )

        item = None
        try:
            item = next(self.opt_gen)
        except StopIteration as ex:
            self.opt_gen.close()
            result_dict = ex.args[0]
            self.opt_gen = result_dict

        if item is not None:
            x_gen, reduce_evals = item
            if reduce_evals:
                self._reduce_evals()

            for i in range(x_gen.shape[0]):
                self.append_request(EvalRequest(x_gen[i, :], None, self.epoch_index))

    def update_epoch(self, resample=False):
        assert self.opt_gen is not None, "Epoch not initialized"

        return_state = None
        return_value = None
        x_resample, y_pred, gen_index, x_sm, y_sm = None, None, None, None, None
        completed_evals = self._update_evals()
        reduce_evals = False

        if completed_evals is None:
            if self.has_requests():
                return_state = StrategyState.WaitingRequests
                return return_state, return_value, completed_evals
            try:
                if isinstance(self.opt_gen, dict):
                    raise StopIteration(self.opt_gen)
                else:
                    item, reduce_evals = next(self.opt_gen)
            except StopIteration as ex:
                if isinstance(self.opt_gen, GeneratorType):
                    self.opt_gen.close()
                self.opt_gen = None

                result_dict = ex.args[0]

                self.stats.update(result_dict.get("stats", {}))

                if "best_x" in result_dict:
                    best_x = result_dict["best_x"]
                    best_y = result_dict["best_y"]
                    gen_index = result_dict["gen_index"]
                    x, y = result_dict["x"], result_dict["y"]
                    optimizer = result_dict["optimizer"]

                    return_state = StrategyState.CompletedEpoch
                    return_value = EpochResults(
                        best_x, best_y, gen_index, x, y, optimizer
                    )
                else:
                    x_resample = result_dict["x_resample"]
                    y_pred = result_dict["y_pred"]
                    gen_index = result_dict["gen_index"]
                    x_sm, y_sm = result_dict["x_sm"], result_dict["y_sm"]
                    optimizer = result_dict["optimizer"]

                    if resample:
                        for i in range(x_resample.shape[0]):
                            self.append_request(
                                EvalRequest(
                                    x_resample[i, :], y_pred[i], self.epoch_index + 1
                                )
                            )

                    return_state = StrategyState.CompletedEpoch
                    return_value = EpochResults(
                        x_resample, y_pred, gen_index, x_sm, y_sm, optimizer
                    )
            else:
                if reduce_evals:
                    self._reduce_evals()
                x_gen = item
                for i in range(x_gen.shape[0]):
                    self.append_request(
                        EvalRequest(x_gen[i, :], None, self.epoch_index)
                    )
                return_state = StrategyState.EnqueuedRequests
                return_value = x_gen

        else:
            x_gen = completed_evals[0]
            y_gen = completed_evals[1]
            c_gen = completed_evals[4]

            try:
                if isinstance(self.opt_gen, dict):
                    raise StopIteration(self.opt_gen)
                else:
                    item, reduce_evals = self.opt_gen.send((x_gen, y_gen, c_gen))
            except StopIteration as ex:
                if isinstance(self.opt_gen, GeneratorType):
                    self.opt_gen.close()
                self.opt_gen = None

                result_dict = ex.args[0]

                self.stats.update(result_dict.get("stats", {}))

                x_resample = None
                y_pred = None
                x_sm = None
                y_sm = None
                gen_index = None
                x = None
                y = None

                if "best_x" in result_dict:
                    best_x = result_dict["best_x"]
                    best_y = result_dict["best_y"]
                    gen_index = result_dict["gen_index"]
                    x, y = result_dict["x"], result_dict["y"]
                    optimizer = result_dict["optimizer"]
                    return_state = StrategyState.CompletedEpoch
                    return_value = EpochResults(
                        best_x, best_y, gen_index, x, y, optimizer
                    )
                else:
                    x_resample = result_dict["x_resample"]
                    y_pred = result_dict["y_pred"]
                    gen_index = result_dict["gen_index"]
                    x_sm, y_sm = result_dict["x_sm"], result_dict["y_sm"]
                    optimizer = result_dict["optimizer"]

                    if resample and x_resample is not None:
                        for i in range(x_resample.shape[0]):
                            self.append_request(
                                EvalRequest(
                                    x_resample[i, :], y_pred[i], self.epoch_index + 1
                                )
                            )

                    return_state = StrategyState.CompletedEpoch
                    return_value = EpochResults(
                        x_resample, y_pred, gen_index, x, y, optimizer
                    )
            else:
                if reduce_evals:
                    self._reduce_evals()
                x_gen = item
                for i in range(x_gen.shape[0]):
                    self.append_request(
                        EvalRequest(x_gen[i, :], None, self.epoch_index)
                    )
                return_state = StrategyState.EnqueuedRequests
                return_value = x_gen

        return return_state, return_value, completed_evals

    def get_best_evals(self, feasible=True):
        if self.x is not None:
            bestx, besty, bestf, bestc, beste, perm = opt.get_best(
                self.x,
                self.y,
                self.f,
                self.c,
                self.prob.dim,
                self.prob.n_objectives,
                feasible=feasible,
            )
            return bestx, besty, self.prob.feature_constructor(bestf), bestc
        else:
            return None, None, None, None

    def get_evals(self, return_features=False, return_constraints=False):
        if return_features and return_constraints:
            return (self.x, self.y, self.f, self.c)
        elif return_features:
            return (self.x, self.y, self.f)
        elif return_constraints:
            return (self.x, self.y, self.c)
        else:
            return (self.x, self.y)

    def get_completed(self):
        if len(self.completed) > 0:
            x_completed = [x.parameters for x in self.completed]
            y_completed = [x.objectives for x in self.completed]
            f_completed = None
            c_completed = None
            if self.prob.n_features is not None:
                f_completed = [x.features for x in self.completed]
            if self.prob.n_constraints is not None:
                c_completed = [x.constraints for x in self.completed]

            return (x_completed, y_completed, f_completed, c_completed)
        else:
            return None


class DistOptimizer:
    def __init__(
        self,
        opt_id,
        obj_fun,
        obj_fun_args=None,
        objective_names=None,
        feature_dtypes=None,
        feature_class=None,
        constraint_names=None,
        n_initial=10,
        initial_maxiter=5,
        initial_method="slh",
        dynamic_initial_sampling=None,
        dynamic_initial_sampling_kwargs=None,
        verbose=False,
        reduce_fun=None,
        reduce_fun_args=None,
        problem_ids=None,
        problem_parameters=None,
        space=None,
        population_size=100,
        num_generations=200,
        resample_fraction=0.25,
        distance_metric=None,
        n_epochs=10,
        save_eval=10,
        file_path=None,
        save=False,
        save_surrogate_evals=False,
        save_optimizer_params=True,
        metadata=None,
        nested_parameter_space=False,
        surrogate_method_name="gpr",
        surrogate_method_kwargs={"anisotropic": False, "optimizer": "sceua"},
        surrogate_custom_training=None,
        surrogate_custom_training_kwargs=None,
        optimizer_name="nsga2",
        optimizer_kwargs={
            "mutation_prob": 0.1,
            "crossover_prob": 0.9,
        },
        sensitivity_method_name=None,
        sensitivity_method_kwargs={},
        optimize_mean_variance=False,
        local_random=None,
        random_seed=None,
        feasibility_method_name=None,
        feasibility_method_kwargs=None,
        termination_conditions=None,
        controller: Optional[distwq.MPIController] = None,
        **kwargs,
    ) -> None:
        """
        `Creates an optimizer based on the MO-ASMO optimizer. Supports
        distributed optimization runs via mpi4py.

        :param set problem_ids (optional): Set of problem ids.
        For solving sets of related problems with the same set of parameters.
        If this parameter is not None, it is expected that the objective function
        will return a dictionary of the form { problem_id: value }
        :param dict problem_parameters: Problem parameters.
        All hyperparameters and their values for the objective
        function, including those not being optimized over. E.g: ``{'beta': 0.44}``.
        Can be an empty dict.
        Can include hyperparameters being optimized over, but does not need to.
        If a hyperparameter is specified in both 'problem_parameters' and 'space', its value
        in 'problem_parameters' will be overridden.
        :param dict space: Hyperparameters to optimize over.
        Entries should be of the form:
        ``parameter: (Low_Bound, High_Bound)`` e.g:
        ``{'alpha': (0.65, 0.85), 'gamma': (1, 8)}``. If both bounds for a
        parameter are Ints, then only integers within the (inclusive) range
        will be sampled and tested.
        :param func obj_fun: function to minimize.
        Must take as argument every parameter specified in
        both 'problem_parameters' and 'space',  and return the result as float.
        :param int n_epochs: (optional) Number of epochs to sample and test params.
        :param int save_eval: (optional) How often to save progress.
        :param str file_path: (optional) File name for restoring and/or saving results and settings.
        :param bool save: (optional) Save settings and progress periodically.
        """

        if (random_seed is not None) and (local_random is not None):
            raise RuntimeError(
                "Both random_seed and local_random are specified! "
                "Only one or the other must be specified. "
            )

        if random_seed is not None:
            local_random = default_rng(seed=random_seed)

        self.controller = controller
        self.opt_id = opt_id
        self.verbose = verbose
        self.population_size = population_size
        self.num_generations = num_generations
        self.resample_fraction = resample_fraction
        self.distance_metric = distance_metric
        self.dynamic_initial_sampling = dynamic_initial_sampling
        self.dynamic_initial_sampling_kwargs = dynamic_initial_sampling_kwargs
        self.surrogate_method_name = surrogate_method_name
        self.surrogate_method_kwargs = surrogate_method_kwargs
        self.surrogate_custom_training = surrogate_custom_training
        self.surrogate_custom_training_kwargs = surrogate_custom_training_kwargs
        self.sensitivity_method_name = sensitivity_method_name
        self.sensitivity_method_kwargs = sensitivity_method_kwargs
        self.optimizer_name = (
            optimizer_name
            if isinstance(optimizer_name, Sequence)
            and not isinstance(optimizer_name, str)
            else (optimizer_name,)
        )
        self.optimizer_kwargs = (
            optimizer_kwargs
            if isinstance(optimizer_kwargs, Sequence)
            else (optimizer_kwargs,)
        )
        self.optimize_mean_variance = optimize_mean_variance
        self.feasibility_method_name = feasibility_method_name
        self.feasibility_method_kwargs = feasibility_method_kwargs
        self.termination_conditions = termination_conditions
        self.metadata = metadata
        self.local_random = local_random
        self.random_seed = random_seed
        if self.resample_fraction > 1.0:
            self.resample_fraction = 1.0

        self.logger = logging.getLogger(opt_id)
        if self.verbose:
            self.logger.setLevel(logging.INFO)

        # Verify inputs
        if file_path is None:
            if problem_parameters is None or space is None:
                raise ValueError(
                    "You must specify at least file name `file_path` or problem "
                    "parameters `problem_parameters` along with a hyperparameter space `space`."
                )
            if save:
                raise ValueError(
                    "If you want to save you must specify a file name `file_path`."
                )
        else:
            if not os.path.isfile(file_path):
                if problem_parameters is None or space is None:
                    raise FileNotFoundError(file_path)

        param_space = None
        if space is not None:
            param_space = ParameterSpace.from_dict(space)

        if problem_parameters is not None:
            problem_parameters = ParameterSpace.from_dict(
                problem_parameters, is_value_only=True
            )

        old_evals = {}
        max_epoch = -1
        stored_random_seed = None
        if file_path is not None:
            if os.path.isfile(file_path):
                (
                    stored_random_seed,
                    max_epoch,
                    old_evals,
                    param_space,
                    objective_names,
                    feature_dtypes,
                    constraint_names,
                    problem_parameters,
                    problem_ids,
                ) = init_from_h5(
                    file_path, param_space.parameter_names, opt_id, self.logger
                )
        if stored_random_seed is not None:
            if local_random is not None:
                if self.logger is not None:
                    self.logger.warning("Using saved random seed to create local RNG. ")
            self.local_random = default_rng(seed=stored_random_seed)

        if problem_parameters is not None:
            assert set(param_space.parameter_names).isdisjoint(
                set(problem_parameters.parameter_names)
            )

        assert param_space.n_parameters > 0
        self.param_space = param_space
        self.param_names = param_space.parameter_names

        assert objective_names is not None
        self.objective_names = objective_names

        has_problem_ids = problem_ids is not None
        if not has_problem_ids:
            problem_ids = set([0])

        self.n_initial = n_initial
        self.initial_maxiter = initial_maxiter
        self.initial_method = initial_method
        self.problem_parameters = problem_parameters
        self.file_path, self.save = file_path, save

        for optimizer_kwargs in self.optimizer_kwargs:
            di_crossover = optimizer_kwargs.get("di_crossover", None)
            if isinstance(di_crossover, dict):
                di_crossover = param_space.flatten(di_crossover)
                optimizer_kwargs["di_crossover"] = di_crossover

            di_mutation = optimizer_kwargs.get("di_mutation", None)
            if isinstance(di_mutation, dict):
                di_mutation = param_space.flatten(di_mutation)
                optimizer_kwargs["di_mutation"] = di_mutation

        self.epoch_count = 0
        self.start_epoch = 0
        if max_epoch > 0:
            self.start_epoch = max_epoch

        self.n_epochs = n_epochs
        self.save_eval = save_eval
        self.save_surrogate_evals_ = save_surrogate_evals
        self.save_optimizer_params_ = save_optimizer_params
        self.saved_eval_count = 0
        self.eval_count = 0

        self.obj_fun_args = obj_fun_args
        if has_problem_ids:
            self.eval_fun = partial(
                eval_obj_fun_mp,
                obj_fun,
                self.problem_parameters,
                self.param_space,
                nested_parameter_space,
                self.obj_fun_args,
                problem_ids,
            )
        else:
            self.eval_fun = partial(
                eval_obj_fun_sp,
                obj_fun,
                self.problem_parameters,
                self.param_space,
                nested_parameter_space,
                self.obj_fun_args,
                0,
            )

        self.reduce_fun = reduce_fun
        self.reduce_fun_args = reduce_fun_args

        self.eval_reqs = {problem_id: {} for problem_id in problem_ids}
        self.old_evals = old_evals

        self.has_problem_ids = has_problem_ids
        self.problem_ids = problem_ids

        self.optimizer_dict = {}
        self.storage_dict = {}

        self.feature_constructor = lambda x: x
        if feature_class is not None:
            self.feature_constructor = import_object_by_path(feature_class)
        self.feature_dtypes = feature_dtypes
        self.feature_names = None
        if feature_dtypes is not None:
            self.feature_names = [dt[0] for dt in feature_dtypes]

        self.constraint_names = constraint_names

        if self.save and file_path is not None:
            if not os.path.isfile(file_path):
                init_h5(
                    self.opt_id,
                    self.problem_ids,
                    self.has_problem_ids,
                    self.param_space,
                    self.objective_names,
                    self.feature_dtypes,
                    self.constraint_names,
                    self.problem_parameters,
                    self.metadata,
                    self.random_seed,
                    self.file_path,
                    surrogate_mean_variance=self.optimize_mean_variance,
                )

        self.stats = {}

    def get_stats(self):
        for problem_id in self.problem_ids:
            if problem_id in self.optimizer_dict:
                self.stats.update(
                    {
                        f"{problem_id}_{k}" if problem_id > 0 else k: v
                        for k, v in self.optimizer_dict[problem_id].stats.items()
                    }
                )

        result = {}
        for key in self.stats:
            if not key.endswith("_start") and not key.endswith("_end"):
                result[key] = self.stats[key]
                continue
            name, period = key.rsplit("_", 1)
            if period == "start":
                if f"{name}_end" in self.stats:
                    result[name] = self.stats[f"{name}_end"] - self.stats[key]

        if self.controller is not None:
            controller_stats = self.controller.stats
            n_processed = self.controller.n_processed
            total_time = self.controller.total_time

            call_times = np.array([s["this_time"] for s in controller_stats])
            call_quotients = np.array([s["time_over_est"] for s in controller_stats])
            cvar_call_quotients = call_quotients.std() / call_quotients.mean()

            result["results_collected"] = n_processed[1:].sum()
            result["total_evaluation_time"] = call_times.sum()
            result["mean_time_per_call"] = call_times.mean()
            result["stdev_time_per_call"] = call_times.std()
            result["cvar_actual_over_estd_time_per_call"] = cvar_call_quotients

            if self.controller.workers_available:
                total_time_est = self.controller.total_time_est
                worker_quotients = total_time / total_time_est
                cvar_worker_quotients = worker_quotients.std() / worker_quotients.mean()

                result["mean_calls_per_worker"] = n_processed[1:].mean()
                result["stdev_calls_per_worker"] = n_processed[1:].std()
                result["min_calls_per_worker"] = n_processed[1:].min()
                result["max_calls_per_worker"] = n_processed[1:].max()
                result["mean_time_per_worker"] = total_time.mean()
                result["stdev_time_per_worker"] = total_time.std()
                result["cvar_actual_over_estd_time_per_worker"] = cvar_worker_quotients

        return result

    def initialize_strategy(self):
        opt_prob = OptProblem(
            self.param_names,
            self.objective_names,
            self.feature_dtypes,
            self.feature_constructor,
            self.constraint_names,
            self.param_space,
            self.eval_fun,
            logger=self.logger,
        )
        dim = len(self.param_names)
        for problem_id in self.problem_ids:
            initial = None
            if problem_id in self.old_evals:
                old_eval_epochs = [e.epoch for e in self.old_evals[problem_id]]
                old_eval_xs = [e.parameters for e in self.old_evals[problem_id]]
                old_eval_ys = [e.objectives for e in self.old_evals[problem_id]]
                epochs = None
                if len(old_eval_epochs) > 0 and old_eval_epochs[0] is not None:
                    epochs = np.concatenate(old_eval_epochs, axis=None)
                x = np.vstack(old_eval_xs)
                y = np.vstack(old_eval_ys)
                f = None
                if self.feature_dtypes is not None:
                    e0 = self.old_evals[problem_id][0]
                    f_shape = e0.features.shape[0] if len(e0.features.shape) > 0 else 0
                    if f_shape == 0:
                        old_eval_fs = [[e.features] for e in self.old_evals[problem_id]]
                    elif f_shape == 1:
                        old_eval_fs = [e.features for e in self.old_evals[problem_id]]
                    else:
                        old_eval_fs = [
                            e.features.reshape((1, f_shape))
                            for e in self.old_evals[problem_id]
                        ]
                    f = self.feature_constructor(np.concatenate(old_eval_fs, axis=0))
                c = None
                if self.constraint_names is not None:
                    old_eval_cs = [e.constraints for e in self.old_evals[problem_id]]
                    c = np.vstack(old_eval_cs)
                initial = (epochs, x, y, f, c)
                if len(old_eval_xs) >= self.n_initial * dim:
                    self.start_epoch += 1

            opt_strategy = DistOptStrategy(
                opt_prob,
                self.n_initial,
                initial=initial,
                resample_fraction=self.resample_fraction,
                population_size=self.population_size,
                num_generations=self.num_generations,
                initial_maxiter=self.initial_maxiter,
                initial_method=self.initial_method,
                distance_metric=self.distance_metric,
                surrogate_method_name=self.surrogate_method_name,
                surrogate_method_kwargs=self.surrogate_method_kwargs,
                surrogate_custom_training=self.surrogate_custom_training,
                surrogate_custom_training_kwargs=self.surrogate_custom_training_kwargs,
                sensitivity_method_name=self.sensitivity_method_name,
                sensitivity_method_kwargs=self.sensitivity_method_kwargs,
                optimizer_name=self.optimizer_name,
                optimizer_kwargs=self.optimizer_kwargs,
                feasibility_method_name=self.feasibility_method_name,
                feasibility_method_kwargs=self.feasibility_method_kwargs,
                termination_conditions=self.termination_conditions,
                optimize_mean_variance=self.optimize_mean_variance,
                local_random=self.local_random,
                logger=self.logger,
                file_path=self.file_path,
            )
            self.optimizer_dict[problem_id] = opt_strategy
            self.storage_dict[problem_id] = []
        if initial is not None:
            self.print_best()

    def save_evals(self):
        """Store results of finished evals to file."""
        finished_evals = {}
        for problem_id in self.problem_ids:
            storage_evals = self.storage_dict[problem_id]
            if len(storage_evals) > 0:
                n = len(self.objective_names)
                epochs_completed = [x.epoch for x in storage_evals]
                x_completed = [x.parameters for x in storage_evals]
                y_completed = [x.objectives for x in storage_evals]
                if self.optimize_mean_variance:
                    y_pred_completed = map(
                        lambda x: [np.nan] * 2 * n if x is None else x,
                        [x.prediction for x in storage_evals],
                    )
                else:
                    y_pred_completed = map(
                        lambda x: [np.nan] * n if x is None else x,
                        [x.prediction for x in storage_evals],
                    )
                f_completed = None
                if self.feature_names is not None:
                    f_completed = [x.features for x in storage_evals]
                c_completed = None
                if self.constraint_names is not None:
                    c_completed = [x.constraints for x in storage_evals]
                finished_evals[problem_id] = (
                    epochs_completed,
                    x_completed,
                    y_completed,
                    f_completed,
                    c_completed,
                    y_pred_completed,
                )
                self.storage_dict[problem_id] = []

        if len(finished_evals) > 0:
            save_to_h5(
                self.opt_id,
                self.problem_ids,
                self.has_problem_ids,
                self.objective_names,
                self.feature_dtypes,
                self.constraint_names,
                self.param_space,
                finished_evals,
                self.problem_parameters,
                self.metadata,
                self.random_seed,
                self.file_path,
                self.logger,
                surrogate_mean_variance=self.optimize_mean_variance,
            )

    def save_surrogate_evals(self, problem_id, epoch, gen_index, x_sm, y_sm):
        """Store results of surrogate evals to file."""
        if x_sm.shape[0] > 0:
            save_surrogate_evals_to_h5(
                self.opt_id,
                problem_id,
                self.param_names,
                self.objective_names,
                epoch,
                gen_index,
                x_sm,
                y_sm,
                self.file_path,
                self.logger,
            )

    def save_optimizer_params(
        self, problem_id, epoch, optimizer_name, optimizer_params
    ):
        """Store optimizer hyper-parameters to file."""
        save_optimizer_params_to_h5(
            self.opt_id,
            problem_id,
            epoch,
            optimizer_name,
            optimizer_params,
            self.file_path,
            self.logger,
        )

    def save_stats(self, problem_id, epoch):
        stats = self.get_stats()

        save_stats_to_h5(
            self.opt_id,
            problem_id,
            epoch,
            self.file_path,
            self.logger,
            stats,
        )

    def get_best(self, feasible=True, return_features=False, return_constraints=False):
        best_results = {}
        for problem_id in self.problem_ids:
            best_x, best_y, best_f, best_c = self.optimizer_dict[
                problem_id
            ].get_best_evals(feasible=feasible)
            prms = list(zip(self.param_names, list(best_x.T)))
            lres = list(zip(self.objective_names, list(best_y.T)))
            lconstr = None
            if self.constraint_names is not None:
                lconstr = list(zip(self.constraint_names, list(best_c.T)))
            if return_features and return_constraints:
                best_results[problem_id] = (prms, lres, best_f, lconstr)
            elif return_features:
                best_results[problem_id] = (prms, lres, best_f)
            elif return_constraints:
                best_results[problem_id] = (prms, lres, lconstr)
            else:
                best_results[problem_id] = (prms, lres)
        if self.has_problem_ids:
            return best_results
        else:
            return best_results[0]

    def print_best(self, feasible=True):
        best_results = self.get_best(
            feasible=feasible, return_features=True, return_constraints=True
        )
        if self.has_problem_ids:
            for problem_id in self.problem_ids:
                prms, res, ftrs, constr = best_results[problem_id]
                prms_dict = dict(prms)
                res_dict = dict(res)
                constr_dict = None
                if constr is not None:
                    constr_dict = dict(constr)
                n_res = next(iter(res_dict.values())).shape[0]
                for i in range(n_res):
                    res_i = {k: res_dict[k][i] for k in res_dict}
                    prms_i = {k: prms_dict[k][i] for k in prms_dict}
                    constr_i = None
                    if constr_dict is not None:
                        constr_i = {k: constr_dict[k][i] for k in constr_dict}
                    ftrs_i = None
                    if ftrs is not None:
                        lftrs_i = ftrs[i]
                    if (ftrs_i is not None) and (constr_i is not None):
                        self.logger.info(
                            f"Best eval {i} so far for id {problem_id}: {res_i}@{prms_i} [{lftrs_i}] [constr: {constr_i}]"
                        )
                    elif constr_i is not None:
                        self.logger.info(
                            f"Best eval {i} so far for id {problem_id}: {res_i}@{prms_i} [constr: {constr_i}]"
                        )
                    elif ftrs_i is not None:
                        self.logger.info(
                            f"Best eval {i} so far for id {problem_id}: {res_i}@{prms_i} [{lftrs_i}]"
                        )
                    else:
                        self.logger.info(
                            f"Best eval {i} so far for id {problem_id}: {res_i}@{prms_i}"
                        )
        else:
            prms, res, ftrs, constr = best_results
            prms_dict = dict(prms)
            res_dict = dict(res)
            n_res = next(iter(res_dict.values())).shape[0]
            constr_dict = None
            if constr is not None:
                constr_dict = dict(constr)
            for i in range(n_res):
                res_i = {k: res_dict[k][i] for k in res_dict}
                prms_i = {k: prms_dict[k][i] for k in prms_dict}
                constr_i = None
                if constr_dict is not None:
                    constr_i = {k: constr_dict[k][i] for k in constr_dict}
                ftrs_i = None
                if ftrs is not None:
                    ftrs_i = ftrs[i]
                if (ftrs_i is not None) and (constr_i is not None):
                    self.logger.info(
                        f"Best eval {i} so far: {res_i}@{prms_i} [{ftrs_i}] [constr: {constr_i}]"
                    )
                elif constr_i is not None:
                    self.logger.info(
                        f"Best eval {i} so far: {res_i}@{prms_i} [constr: {constr_i}]"
                    )
                elif ftrs_i is not None:
                    self.logger.info(
                        f"Best eval {i} so far: {res_i}@{prms_i} [{ftrs_i}]"
                    )
                else:
                    self.logger.info(f"Best eval {i} so far: {res_i}@{prms_i}")

    def _process_requests(self):
        task_ids = []

        has_requests = False
        for problem_id in self.problem_ids:
            has_requests = (
                has_requests or self.optimizer_dict[problem_id].has_requests()
            )

        next_phase = False
        while (len(task_ids) > 0) or has_requests:
            self.controller.process()

            if (self.controller.time_limit is not None) and (
                time.time() - self.controller.start_time
            ) >= self.controller.time_limit:
                break

            if len(task_ids) > 0:
                rets = self.controller.probe_all_next_results()
                for ret in rets:
                    task_id, res = ret
                    if self.reduce_fun is None:
                        rres = res
                    else:
                        if self.reduce_fun_args is None:
                            rres = self.reduce_fun(res)
                        else:
                            rres = self.reduce_fun(res, *self.reduce_fun_args)

                    t = rres.pop("time", -1.0)
                    for problem_id in rres:
                        eval_req = self.eval_reqs[problem_id][task_id]
                        eval_x = eval_req.parameters
                        eval_pred = eval_req.prediction
                        eval_epoch = eval_req.epoch
                        if (
                            self.feature_names is not None
                            and self.constraint_names is not None
                        ):
                            entry = self.optimizer_dict[problem_id].complete_request(
                                eval_x,
                                rres[problem_id][0],
                                f=rres[problem_id][1],
                                c=rres[problem_id][2],
                                pred=eval_pred,
                                epoch=eval_epoch,
                                time=t,
                            )
                            self.storage_dict[problem_id].append(entry)
                        elif self.feature_names is not None:
                            entry = self.optimizer_dict[problem_id].complete_request(
                                eval_x,
                                rres[problem_id][0],
                                f=rres[problem_id][1],
                                pred=eval_pred,
                                epoch=eval_epoch,
                                time=t,
                            )
                            self.storage_dict[problem_id].append(entry)
                        elif self.constraint_names is not None:
                            entry = self.optimizer_dict[problem_id].complete_request(
                                eval_x,
                                rres[problem_id][0],
                                c=rres[problem_id][1],
                                pred=eval_pred,
                                epoch=eval_epoch,
                                time=t,
                            )
                            self.storage_dict[problem_id].append(entry)
                        else:
                            entry = self.optimizer_dict[problem_id].complete_request(
                                eval_x,
                                rres[problem_id],
                                pred=eval_pred,
                                epoch=eval_epoch,
                                time=t,
                            )
                            self.storage_dict[problem_id].append(entry)
                        prms = list(zip(self.param_names, list(eval_x.T)))
                        lftrs = None
                        lres = None
                        if (
                            self.feature_names is not None
                            and self.constraint_names is not None
                        ):
                            lres = list(
                                zip(self.objective_names, rres[problem_id][0].T)
                            )
                            lftrs = [x for x in rres[problem_id][1]]
                            lconstr = list(
                                zip(self.constraint_names, rres[problem_id][2].T)
                            )
                            logger.info(
                                f"problem id {problem_id}: optimization epoch {eval_epoch}: parameters {prms}: {lres} / {lftrs} constr: {lconstr}"
                            )
                        elif self.feature_names is not None:
                            lres = list(
                                zip(self.objective_names, rres[problem_id][0].T)
                            )
                            lftrs = [x for x in rres[problem_id][1]]
                            logger.info(
                                f"problem id {problem_id}: optimization epoch {eval_epoch}: parameters {prms}: {lres} / {lftrs}"
                            )
                        elif self.constraint_names is not None:
                            lres = list(
                                zip(self.objective_names, rres[problem_id][0].T)
                            )
                            lconstr = list(
                                zip(self.constraint_names, rres[problem_id][1].T)
                            )
                            logger.info(
                                f"problem id {problem_id}: optimization epoch {eval_epoch}: parameters {prms}: {lres} / constr: {lconstr}"
                            )
                        else:
                            lres = list(zip(self.objective_names, rres[problem_id].T))
                            logger.info(
                                f"problem id {problem_id}: optimization epoch {eval_epoch}: parameters {prms}: {lres}"
                            )

                    self.eval_count += 1
                    task_ids.remove(task_id)

            if (
                self.save
                and (self.eval_count > 0)
                and (self.saved_eval_count < self.eval_count)
                and ((self.eval_count - self.saved_eval_count) >= self.save_eval)
            ):
                self.save_evals()
                self.saved_eval_count = self.eval_count

            if (self.controller.time_limit is not None) and (
                time.time() - self.controller.start_time
            ) >= self.controller.time_limit:
                break

            task_args = []
            task_reqs = []
            while not next_phase:
                eval_req_dict = {}
                eval_x_dict = {}
                for problem_id in self.problem_ids:
                    eval_req = self.optimizer_dict[problem_id].get_next_request()
                    if eval_req is None:
                        next_phase = True
                        has_requests = False
                        break
                    else:
                        has_requests = True or has_requests
                        eval_req_dict[problem_id] = eval_req
                        eval_x_dict[problem_id] = eval_req.parameters

                if next_phase:
                    break
                else:
                    task_args.append(
                        (
                            self.opt_id,
                            eval_x_dict,
                        )
                    )
                    task_reqs.append(eval_req_dict)

            if (self.controller.time_limit is not None) and (
                time.time() - self.controller.start_time
            ) >= self.controller.time_limit:
                break

            if len(task_args) > 0:
                new_task_ids = self.controller.submit_multiple(
                    "eval_fun", module_name="dmosopt.dmosopt", args=task_args
                )
                for task_id, eval_req_dict in zip(new_task_ids, task_reqs):
                    task_ids.append(task_id)
                    for problem_id in self.problem_ids:
                        self.eval_reqs[problem_id][task_id] = eval_req_dict[problem_id]

        if (
            self.save
            and (self.eval_count > 0)
            and (self.saved_eval_count < self.eval_count)
        ):
            self.save_evals()
            self.saved_eval_count = self.eval_count

        assert len(task_ids) == 0
        return self.eval_count, self.saved_eval_count

    def run_epoch(self, completed_epoch=False):
        if self.controller is None:
            raise RuntimeError(
                "DistOptimizer: method epoch cannot be executed when controller is not set."
            )

        epoch = self.epoch_count + self.start_epoch
        advance_epoch = self.epoch_count < self.n_epochs - 1

        self.stats["init_sampling_start"] = time.time()
        eval_count, saved_eval_count = self._process_requests()

        for problem_id in self.problem_ids:
            distopt = self.optimizer_dict[problem_id]

            # dynamic sampling
            if self.dynamic_initial_sampling is not None and self.epoch_count == 0:
                dynamic_initial_sampler = import_object_by_path(
                    self.dynamic_initial_sampling
                )

                dyn_sample_iter_count = 0
                while True:
                    more_samples = dynamic_initial_sampler(
                        file_path=self.file_path,
                        iteration=dyn_sample_iter_count,
                        evaluated_samples=distopt.completed,
                        next_samples=opt.xinit(
                            self.n_initial,
                            distopt.prob.param_names,
                            distopt.prob.lb,
                            distopt.prob.ub,
                            nPrevious=None,
                            maxiter=self.initial_maxiter,
                            method=self.initial_method,
                            local_random=self.local_random,
                            logger=self.logger,
                        ),
                        sampler={
                            "n_initial": self.n_initial,
                            "maxiter": self.initial_maxiter,
                            "method": self.initial_method,
                            "param_names": distopt.prob.param_names,
                            "xlb": distopt.prob.lb,
                            "xub": distopt.prob.ub,
                        },
                        **(self.dynamic_initial_sampling_kwargs or {}),
                    )

                    if more_samples is None:
                        break

                    distopt.reqs.extend(
                        [
                            EvalRequest(more_samples[i, :], None, 0)
                            for i in range(more_samples.shape[0])
                        ]
                    )

                    self._process_requests()

                    dyn_sample_iter_count += 1

            distopt.initialize_epoch(epoch)

        self.stats["init_sampling_end"] = time.time()

        while not completed_epoch:
            eval_count, saved_eval_count = self._process_requests()

            for problem_id in self.problem_ids:
                ## Have we completed the evaluations for an epoch or a generation
                strategy_state, strategy_value, completed_evals = self.optimizer_dict[
                    problem_id
                ].update_epoch(resample=advance_epoch)
                completed_epoch = strategy_state == StrategyState.CompletedEpoch
                if completed_epoch:
                    res = strategy_value

                    ## Compute prediction accuracy of completed evaluations
                    if (completed_evals is not None) and (epoch > 1):
                        x_completed = completed_evals[0]
                        y_completed = completed_evals[1]
                        pred_completed = completed_evals[2]
                        c_completed = completed_evals[4]
                        if c_completed is not None:
                            feasible = np.argwhere(np.all(c_completed > 0.0, axis=1))
                            if len(feasible) > 0:
                                feasible = feasible.ravel()
                                x_completed = x_completed[feasible, :]
                                y_completed = y_completed[feasible, :]
                                pred_completed = pred_completed[feasible, :]

                        if x_completed.shape[0] > 0:
                            n_objectives = y_completed.shape[1]
                            mae = []
                            for i in range(n_objectives):
                                y_i = y_completed[:, i]
                                pred_i = pred_completed[:, i]
                                mae.append(
                                    np.mean(
                                        np.abs(
                                            y_i[~np.isnan(y_i)] - pred_i[~np.isnan(y_i)]
                                        )
                                    )
                                )
                            logger.info(
                                f"surrogate accuracy at epoch {epoch - 1} for problem {problem_id} was {mae}"
                            )

                    if advance_epoch and epoch > 0:
                        x_sm, y_sm = None, None
                        if self.save and self.save_surrogate_evals_:
                            gen_index = res.gen_index
                            x_sm, y_sm = res.x, res.y
                            self.save_surrogate_evals(
                                problem_id, epoch, gen_index, x_sm, y_sm
                            )
                        if self.save and self.save_optimizer_params_:
                            optimizer = res.optimizer
                            self.save_optimizer_params(
                                problem_id,
                                epoch,
                                optimizer.name,
                                optimizer.opt_parameters,
                            )
        if self.save:
            self.save_stats(problem_id, epoch)

        self.epoch_count = self.epoch_count + 1
        return self.epoch_count


def h5_get_group(h, groupname):
    if groupname in h.keys():
        g = h[groupname]
    else:
        g = h.create_group(groupname)
    return g


def h5_get_dataset(g, dsetname, **kwargs):
    if "shape" not in kwargs:
        kwargs["shape"] = (0,)
    if dsetname in g.keys():
        dset = g[dsetname]
    else:
        dset = g.create_dataset(dsetname, **kwargs)
    return dset


def h5_concat_dataset(dset, data):
    dsize = dset.shape[0]
    newshape = (dsize + data.shape[0],) + data.shape[1:]
    dset.resize(newshape)
    dset[dsize:] = data
    return dset


def create_param_paths_dtype(
    parameter_enum_dtype: np.dtype, max_depth: int = 10, max_name_length: int = 128
) -> np.dtype:
    """
    Create a NumPy dtype for storing parameter paths in HDF5.

    Args:
        parameter_enum_dtype: parameter enumeration type
        max_depth: Maximum nesting depth of parameters
        max_name_length: Maximum length of each parameter name component

    Returns:
        Structured dtype for parameter paths with fields:
        - path_length: Number of components in path
        - components: Fixed-size array of fixed-length strings

    """
    return np.dtype(
        [
            ("parameter", parameter_enum_dtype),
            ("path_length", np.int32),
            ("components", f"S{max_name_length}", (max_depth,)),
        ]
    )


def param_paths_to_array(
    param_mapping: Dict[str, int],
    parameter_enum_dtype: np.dtype,
    param_paths: Dict[str, List[str]],
    max_depth: int = 10,
    max_name_length: int = 128,
) -> np.ndarray:
    """
    Convert parameter paths dictionary to structured array.

    Args:
        param_paths: Dictionary mapping full parameter names to path components
        max_depth: Maximum allowed path depth
        max_name_length: Maximum allowed length for each path component

    Returns:
        Structured array containing parameter paths
    """
    dtype = create_param_paths_dtype(parameter_enum_dtype, max_depth, max_name_length)
    arr = np.zeros(len(param_paths), dtype=dtype)

    for i, (name, path) in enumerate(param_paths.items()):
        if len(path) > max_depth:
            raise ValueError(f"Path depth {len(path)} exceeds maximum {max_depth}")
        if any(len(comp) > max_name_length for comp in path):
            raise ValueError(f"Path component exceeds maximum length {max_name_length}")

        arr[i]["parameter"] = param_mapping[name]
        arr[i]["path_length"] = len(path)
        for j, component in enumerate(path):
            arr[i]["components"][j] = component.encode("ascii")

    return arr


def array_to_param_paths(arr: np.ndarray) -> Dict[str, List[str]]:
    """
    Convert structured array back to parameter paths dictionary.

    Args:
        arr: Structured array containing parameter paths

    Returns:
        Dictionary mapping full parameter names to path components
    """
    param_paths = {}

    for row in arr:
        path_length = row["path_length"]
        components = [
            comp.decode("ascii").rstrip("\x00")
            for comp in row["components"][:path_length]
        ]
        full_name = ".".join(components)
        param_paths[full_name] = components

    return param_paths


def h5_init_types(
    f,
    opt_id,
    objective_names,
    feature_dtypes,
    constraint_names,
    problem_parameters,
    parameter_space,
    surrogate_mean_variance=False,
):
    opt_grp = h5_get_group(f, opt_id)

    objective_keys = set(objective_names)
    feature_keys = None
    if feature_dtypes is not None:
        feature_keys = [feature_dtype[0] for feature_dtype in feature_dtypes]

    # create HDF5 types for the objectives and features
    objective_mapping = {name: idx for (idx, name) in enumerate(objective_keys)}
    feature_mapping = None
    if feature_keys is not None:
        feature_mapping = {name: idx for (idx, name) in enumerate(feature_keys)}

    dt = h5py.enum_dtype(objective_mapping, basetype=np.uint16)
    opt_grp["objective_enum"] = dt

    dt = np.dtype([("objective", opt_grp["objective_enum"])])
    opt_grp["objective_spec_type"] = dt

    dt = np.dtype(
        {"names": objective_names, "formats": [np.float32] * len(objective_names)}
    )
    opt_grp["objective_type"] = dt

    if surrogate_mean_variance:
        surrogate_objective_names = list(
            [f"{name} mean" for name in objective_names]
        ) + list([f"{name} variance" for name in objective_names])
        surrogate_dt = np.dtype(
            {
                "names": surrogate_objective_names,
                "formats": [np.float32] * len(surrogate_objective_names),
            }
        )
    else:
        surrogate_dt = np.dtype(
            {"names": objective_names, "formats": [np.float32] * len(objective_names)}
        )
    opt_grp["surrogate_objective_type"] = surrogate_dt

    dset = h5_get_dataset(
        opt_grp,
        "objective_spec",
        maxshape=(len(objective_names),),
        dtype=opt_grp["objective_spec_type"].dtype,
    )
    dset.resize((len(objective_names),))
    a = np.zeros(len(objective_names), dtype=opt_grp["objective_spec_type"].dtype)
    for idx, parm in enumerate(objective_names):
        a[idx]["objective"] = objective_mapping[parm]
    dset[:] = a

    if feature_mapping is not None:
        dt = h5py.enum_dtype(feature_mapping, basetype=np.uint16)
        opt_grp["feature_enum"] = dt

        dt = np.dtype([("feature", opt_grp["feature_enum"])])
        opt_grp["feature_spec_type"] = dt

        dt = np.dtype(feature_dtypes)
        opt_grp["feature_type"] = dt

        dset = h5_get_dataset(
            opt_grp,
            "feature_spec",
            maxshape=(len(feature_keys),),
            dtype=opt_grp["feature_spec_type"].dtype,
        )
        dset.resize((len(feature_keys),))
        a = np.zeros(len(feature_keys), dtype=opt_grp["feature_spec_type"].dtype)
        for idx, parm in enumerate(feature_keys):
            a[idx]["feature"] = feature_mapping[parm]
        dset[:] = a

    # create HDF5 types for the constraints
    constr_keys = None
    if constraint_names is not None:
        constr_keys = set(constraint_names)

        constr_mapping = {name: idx for (idx, name) in enumerate(constr_keys)}

        dt = h5py.enum_dtype(constr_mapping, basetype=np.uint16)
        opt_grp["constraint_enum"] = dt

        dt = np.dtype([("constraint", opt_grp["constraint_enum"])])
        opt_grp["constraint_spec_type"] = dt

        dt = np.dtype(
            {"names": constraint_names, "formats": [np.float32] * len(constraint_names)}
        )
        opt_grp["constraint_type"] = dt

        dset = h5_get_dataset(
            opt_grp,
            "constraint_spec",
            maxshape=(len(constraint_names),),
            dtype=opt_grp["constraint_spec_type"].dtype,
        )
        dset.resize((len(constraint_names),))
        a = np.zeros(len(constraint_names), dtype=opt_grp["constraint_spec_type"].dtype)
        for idx, parm in enumerate(constraint_names):
            a[idx]["constraint"] = constr_mapping[parm]
            dset[:] = a

    # create HDF5 types describing the parameter specification
    # while preserving parameter order from parameter_space.parameter_names
    param_keys = []
    for name in problem_parameters.parameter_names:
        if name not in param_keys:
            param_keys.append(name)
    for name in parameter_space.parameter_names:
        if name not in param_keys:
            param_keys.append(name)

    param_mapping = {name: idx for (idx, name) in enumerate(param_keys)}

    dt = h5py.enum_dtype(param_mapping, basetype=np.uint16)
    opt_grp["parameter_enum"] = dt

    dt = np.dtype(
        {
            "names": parameter_space.parameter_names,
            "formats": [np.float32] * len(parameter_space.parameter_names),
        }
    )
    opt_grp["parameter_space_type"] = dt

    dt = np.dtype(
        [
            ("parameter", opt_grp["parameter_enum"]),
            ("is_integer", bool),
            ("value", np.float32),
        ]
    )
    opt_grp["problem_parameters_type"] = dt

    dset = h5_get_dataset(
        opt_grp,
        "problem_parameters",
        maxshape=(problem_parameters.n_parameters,),
        dtype=opt_grp["problem_parameters_type"].dtype,
    )
    dset.resize((problem_parameters.n_parameters,))
    a = np.zeros(
        problem_parameters.n_parameters, dtype=opt_grp["problem_parameters_type"].dtype
    )
    idx = 0
    for idx, parm in enumerate(problem_parameters.items):
        a[idx]["parameter"] = param_mapping[parm.name]
        a[idx]["value"] = parm.value
        a[idx]["is_integer"] = parm.is_integer
    dset[:] = a

    parameter_spec_dtype = np.dtype(
        [
            ("parameter", opt_grp["parameter_enum"]),
            ("is_integer", bool),
            ("lower", np.float32),
            ("upper", np.float32),
        ]
    )
    opt_grp["parameter_spec_type"] = parameter_spec_dtype

    dset = h5_get_dataset(
        opt_grp,
        "parameter_spec",
        maxshape=(parameter_space.n_parameters,),
        dtype=opt_grp["parameter_spec_type"].dtype,
    )
    dset.resize((parameter_space.n_parameters,))
    a = np.zeros(
        parameter_space.n_parameters, dtype=opt_grp["parameter_spec_type"].dtype
    )
    for idx, parm in enumerate(parameter_space.items):
        a[idx]["parameter"] = param_mapping[parm.name]
        a[idx]["is_integer"] = parm.is_integer
        a[idx]["lower"] = parm.lower
        a[idx]["upper"] = parm.upper
    dset[:] = a

    parameter_path_dtype = create_param_paths_dtype(opt_grp["parameter_enum"])
    opt_grp["parameter_path_type"] = parameter_path_dtype
    all_parameter_paths = parameter_space.parameter_paths
    all_parameter_paths.update(problem_parameters.parameter_paths)
    param_path_array = param_paths_to_array(
        param_mapping, opt_grp["parameter_enum"], all_parameter_paths
    )

    dset = h5_get_dataset(
        opt_grp,
        "parameter_paths",
        maxshape=(len(all_parameter_paths),),
        dtype=opt_grp["parameter_path_type"].dtype,
    )
    dset.resize((len(param_path_array),))
    dset[:] = param_path_array


def h5_init_opt_group(
    f,
    opt_id,
    objective_names,
    feature_dtypes,
    constraint_names,
    problem_parameters,
    parameter_space,
    problem_ids,
    has_problem_ids,
    metadata,
    random_seed,
    surrogate_mean_variance=False,
):
    if opt_id in f.keys():
        return
    h5_init_types(
        f,
        opt_id,
        objective_names,
        feature_dtypes,
        constraint_names,
        problem_parameters,
        parameter_space,
        surrogate_mean_variance=surrogate_mean_variance,
    )
    opt_grp = h5_get_group(f, opt_id)
    if metadata is not None:
        opt_grp["metadata"] = metadata
    opt_grp["problem_ids"] = np.asarray(
        list(problem_ids) if has_problem_ids else [0], dtype=np.int32
    )
    if random_seed is not None:
        opt_grp["random_seed"] = np.asarray([random_seed], dtype=np.int32)


def h5_load_raw(input_file, opt_id):
    ## N is number of trials
    ## M is number of hyperparameters
    f = h5py.File(input_file, "r")
    opt_grp = h5_get_group(f, opt_id)

    objective_enum_dict = h5py.check_enum_dtype(opt_grp["objective_enum"].dtype)
    objective_enum_name_dict = {idx: parm for parm, idx in objective_enum_dict.items()}
    objective_names = [
        objective_enum_name_dict[spec[0]] for spec in iter(opt_grp["objective_spec"])
    ]

    constraint_names = None
    if "constraint_enum" in opt_grp:
        constraint_enum_dict = h5py.check_enum_dtype(opt_grp["constraint_enum"].dtype)
        constraint_idx_dict = {parm: idx for parm, idx in constraint_enum_dict.items()}
        constraint_name_dict = {idx: parm for parm, idx in constraint_idx_dict.items()}
        constraint_names = [
            constraint_name_dict[spec[0]] for spec in iter(opt_grp["constraint_spec"])
        ]

    feature_names = None
    if "feature_enum" in opt_grp:
        feature_enum_dict = h5py.check_enum_dtype(opt_grp["feature_enum"].dtype)
        feature_idx_dict = {parm: idx for parm, idx in feature_enum_dict.items()}
        feature_name_dict = {idx: parm for parm, idx in feature_idx_dict.items()}
        feature_names = [
            feature_name_dict[spec[0]] for spec in iter(opt_grp["feature_spec"])
        ]

    parameter_paths = None
    if "parameter_paths" in opt_grp:
        param_path_array = opt_grp["parameter_paths"][:]
        parameter_paths = array_to_param_paths(param_path_array)

    parameter_enum_dict = h5py.check_enum_dtype(opt_grp["parameter_enum"].dtype)
    parameters_idx_dict = {parm: idx for parm, idx in parameter_enum_dict.items()}
    parameters_name_dict = {idx: parm for parm, idx in parameters_idx_dict.items()}

    problem_parameters = {}
    problem_parameters_dset = opt_grp["problem_parameters"][:]
    problem_parameters_integer_flag = False
    if len(problem_parameters_dset) > 0:
        if len(problem_parameters_dset[0]) > 2:
            problem_parameters_integer_flag = True
    for entry in problem_parameters_dset:
        idx = entry[0]
        if problem_parameters_integer_flag:
            value = entry[2]
        else:
            value = entry[1]
        param_name = parameters_name_dict[idx]
        param_dict = problem_parameters
        if parameter_paths is not None:
            param_path = parameter_paths[param_name]
            for comp in param_path:
                if comp in param_dict:
                    param_dict = param_dict[comp]
                else:
                    new_param_dict = {}
                    param_dict[comp] = new_param_dict
                    param_dict = new_param_dict
        param_dict[param_name] = value

    parameter_specs = [
        (parameters_name_dict[spec[0]], tuple(spec)[1:])
        for spec in iter(opt_grp["parameter_spec"])
    ]

    problem_ids = None
    if "problem_ids" in opt_grp:
        problem_ids = set(opt_grp["problem_ids"])

    raw_results = {}
    for problem_id in problem_ids if problem_ids is not None else [0]:
        if str(problem_id) in opt_grp:
            raw_results[problem_id] = {
                "objectives": opt_grp[str(problem_id)]["objectives"][:],
                "parameters": opt_grp[str(problem_id)]["parameters"][:],
            }
            if "features" in opt_grp[str(problem_id)]:
                raw_results[problem_id]["features"] = opt_grp[str(problem_id)][
                    "features"
                ][:]
            if "constraints" in opt_grp[str(problem_id)]:
                raw_results[problem_id]["constraints"] = opt_grp[str(problem_id)][
                    "constraints"
                ][:]
            if "epochs" in opt_grp[str(problem_id)]:
                raw_results[problem_id]["epochs"] = opt_grp[str(problem_id)]["epochs"][
                    :
                ]
            if "predictions" in opt_grp[str(problem_id)]:
                raw_results[problem_id]["predictions"] = opt_grp[str(problem_id)][
                    "predictions"
                ][:]

    random_seed = None
    if "random_seed" in opt_grp:
        random_seed = opt_grp["random_seed"][0]

    f.close()

    raw_spec = {}
    param_names = []
    for param_name, spec in parameter_specs:
        param_names.append(param_name)

        param_path = None
        param_dict = raw_spec
        if parameter_paths is not None:
            param_path = parameter_paths[param_name]
            for comp in param_path[:-1]:
                if comp in param_dict:
                    param_dict = param_dict[comp]
                else:
                    new_param_dict = {}
                    param_dict[comp] = new_param_dict
                    param_dict = new_param_dict
            param_name = param_path[-1]

        is_int, lo, hi = spec
        param_dict[param_name] = [lo, hi, is_int]

    info = {
        "random_seed": random_seed,
        "objectives": objective_names,
        "features": feature_names,
        "constraints": constraint_names,
        "params": param_names,
        "problem_parameters": problem_parameters,
        "problem_ids": problem_ids,
    }

    return raw_spec, raw_results, info


def h5_load_all(file_path, opt_id):
    """
    Loads an HDF5 file containing
    (spec, results, info) where
      results: np.array of shape [N, M+1] where
        N is number of trials
        M is number of hyperparameters
        results[:, 0] is result/loss
        results[:, 1:] is [param1, param2, ...]
      spec: (is_integer, lower, upper)
        where each element is list of length M
      info: dict with keys
        params, problem
    Assumes the structure is located in group /{opt_id}
    Returns
    (param_space, function_eval, dict, prev_best)
      where prev_best: np.array[result, param1, param2, ...]
    """
    raw_spec, raw_problem_results, info = h5_load_raw(file_path, opt_id)
    evals = {problem_id: [] for problem_id in raw_problem_results}
    for problem_id in raw_problem_results:
        problem_evals = []
        raw_results = raw_problem_results[problem_id]
        epochs = raw_results.get("epochs", None)
        ys = raw_results["objectives"]
        xs = raw_results["parameters"]
        fs = raw_results.get("features", None)
        cs = raw_results.get("constraints", None)
        ypreds = raw_results.get("predictions", None)
        for i in range(ys.shape[0]):
            epoch_i = None
            if epochs is not None:
                epoch_i = epochs[i]
            x_i = list(xs[i])
            y_i = list(ys[i])
            y_pred_i = None
            if ypreds is not None:
                y_pred_i = list(ypreds[i])
            f_i = None
            if fs is not None:
                f_i = fs[i]
            c_i = None
            if cs is not None:
                c_i = list(cs[i])
            problem_evals.append(EvalEntry(epoch_i, x_i, y_i, f_i, c_i, y_pred_i))
        evals[problem_id] = problem_evals
    return raw_spec, evals, info


def init_from_h5(file_path, param_names, opt_id, logger=None):
    # Load progress and settings from file, then compare each
    # restored setting with settings specified by args (if any)
    raw_spec, old_evals, info = h5_load_all(file_path, opt_id)
    param_space = ParameterSpace.from_dict(raw_spec)
    saved_params = info["params"]
    max_epoch = -1
    for problem_id in old_evals:
        n_old_evals = len(old_evals[problem_id])
        if logger is not None:
            logger.info(f"Restored {n_old_evals} trials for problem {problem_id}")
        for ev in old_evals[problem_id]:
            if ev.epoch is not None:
                max_epoch = max(max_epoch, ev.epoch)
            else:
                break

    if (param_names is not None) and param_names != saved_params:
        # Switching parameters being optimized over would throw off the optimizer.
        # Must use restored parameters from specified
        raise RuntimeError(
            f"Saved parameters {saved_params} differ from currently specified "
            f"{param_names}. "
        )

    problem_parameters = ParameterSpace.from_dict(
        info["problem_parameters"], is_value_only=True
    )
    objective_names = info["objectives"]
    feature_names = info["features"]
    constraint_names = info["constraints"]
    problem_ids = info["problem_ids"] if "problem_ids" in info else None
    random_seed = info["random_seed"] if "random_seed" in info else None

    return (
        random_seed,
        max_epoch,
        old_evals,
        param_space,
        objective_names,
        feature_names,
        constraint_names,
        problem_parameters,
        problem_ids,
    )


def save_to_h5(
    opt_id,
    problem_ids,
    has_problem_ids,
    objective_names,
    feature_dtypes,
    constraint_names,
    parameter_space,
    evals,
    problem_parameters,
    metadata,
    random_seed,
    fpath,
    logger,
    surrogate_mean_variance=False,
):
    """
    Save progress and settings to an HDF5 file 'fpath'.
    """

    f = h5py.File(fpath, "a")
    h5_init_opt_group(
        f,
        opt_id,
        objective_names,
        feature_dtypes,
        constraint_names,
        problem_parameters,
        parameter_space,
        problem_ids,
        has_problem_ids,
        metadata,
        random_seed,
        surrogate_mean_variance=surrogate_mean_variance,
    )

    opt_grp = h5_get_group(f, opt_id)

    # parameter_enum_dict = h5py.check_enum_dtype(opt_grp["parameter_enum"].dtype)
    # parameters_idx_dict = {parm: idx for parm, idx in parameter_enum_dict.items()}
    # parameters_name_dict = {idx: parm for parm, idx in parameters_idx_dict.items()}

    for problem_id in problem_ids:
        (
            prob_evals_epoch,
            prob_evals_x,
            prob_evals_y,
            prob_evals_f,
            prob_evals_c,
            prob_evals_y_pred,
        ) = evals[problem_id]
        opt_prob = h5_get_group(opt_grp, str(problem_id))

        if logger is not None:
            logger.info(
                f"Saving {len(prob_evals_y)} evaluations for problem id {problem_id} to {fpath}."
            )

        dset = h5_get_dataset(opt_prob, "epochs", maxshape=(None,), dtype=np.uint32)
        data = np.asarray(prob_evals_epoch, dtype=np.uint32)
        h5_concat_dataset(dset, data)

        dset = h5_get_dataset(
            opt_prob, "objectives", maxshape=(None,), dtype=opt_grp["objective_type"]
        )
        data = np.array(
            [tuple(y) for y in prob_evals_y], dtype=opt_grp["objective_type"]
        )
        h5_concat_dataset(dset, data)

        dset = h5_get_dataset(
            opt_prob,
            "parameters",
            maxshape=(None,),
            dtype=opt_grp["parameter_space_type"],
        )
        data = np.array(
            [tuple(x) for x in prob_evals_x], dtype=opt_grp["parameter_space_type"]
        )
        h5_concat_dataset(dset, data)

        if prob_evals_f is not None:
            data = np.concatenate(prob_evals_f, dtype=opt_grp["feature_type"], axis=0)
            n_feature_measurements = 1
            if len(data.shape) > 1:
                n_feature_measurements = data.shape[1]
            dset = h5_get_dataset(
                opt_prob,
                "features",
                maxshape=(None,)
                if n_feature_measurements == 1
                else (None, n_feature_measurements),
                shape=(0,) if n_feature_measurements == 1 else (0, 0),
                dtype=opt_grp["feature_type"],
            )
            h5_concat_dataset(dset, data)

        if prob_evals_c is not None:
            dset = h5_get_dataset(
                opt_prob,
                "constraints",
                maxshape=(None,),
                dtype=opt_grp["constraint_type"],
            )
            data = np.array(
                [tuple(c) for c in prob_evals_c], dtype=opt_grp["constraint_type"]
            )
            h5_concat_dataset(dset, data)

        dset = h5_get_dataset(
            opt_prob,
            "predictions",
            maxshape=(None,),
            dtype=opt_grp["surrogate_objective_type"],
        )

        data = np.array(
            [tuple(y) for y in prob_evals_y_pred],
            dtype=opt_grp["surrogate_objective_type"],
        )
        h5_concat_dataset(dset, data)

    f.close()


def save_optimizer_params_to_h5(
    opt_id,
    problem_id,
    epoch,
    optimizer_name,
    optimizer_params,
    fpath,
    logger,
):
    """
    Save optimizer hyper-parameters to an HDF5 file 'fpath'.
    """

    f = h5py.File(fpath, "a")

    opt_grp = h5_get_group(f, opt_id)

    opt_params_grp = h5_get_group(opt_grp, "optimizer_params")
    opt_params_epoch_grp = h5_get_group(opt_params_grp, f"{epoch}")

    if logger is not None:
        logger.info(
            f"Saving optimizer hyper-parameters for problem id {problem_id} epoch {epoch} to {fpath}."
        )
    if "optimizer_name" not in opt_params_epoch_grp:
        opt_params_epoch_grp["optimizer_name"] = optimizer_name
    for k, v in optimizer_params.items():
        if v is not None and k not in opt_params_epoch_grp:
            opt_params_epoch_grp[k] = v

    f.close()


def save_surrogate_evals_to_h5(
    opt_id,
    problem_id,
    param_names,
    objective_names,
    epoch,
    gen_index,
    x_sm,
    y_sm,
    fpath,
    logger,
):
    """
    Save surrogate evaluations to an HDF5 file 'fpath'.
    """

    f = h5py.File(fpath, "a")

    opt_grp = h5_get_group(f, opt_id)

    # opt_prob = h5_get_group(opt_grp, str(problem_id))
    opt_sm = h5_get_group(opt_grp, "surrogate_evals")

    n_evals = x_sm.shape[0]
    if logger is not None:
        logger.info(
            f"Saving {n_evals} surrogate evaluations for problem id {problem_id} to {fpath}."
        )

    dset = h5_get_dataset(opt_sm, "epochs", maxshape=(None,), dtype=np.uint32)
    data = np.asarray([epoch] * n_evals, dtype=np.uint32)
    h5_concat_dataset(dset, data)

    dset = h5_get_dataset(opt_sm, "generations", maxshape=(None,), dtype=np.uint32)
    h5_concat_dataset(dset, gen_index)

    dset = h5_get_dataset(
        opt_sm,
        "objectives",
        maxshape=(None,),
        dtype=opt_grp["surrogate_objective_type"],
    )
    data = np.array([tuple(y) for y in y_sm], dtype=opt_grp["surrogate_objective_type"])
    h5_concat_dataset(dset, data)

    dset = h5_get_dataset(
        opt_sm, "parameters", maxshape=(None,), dtype=opt_grp["parameter_space_type"]
    )
    data = np.array([tuple(x) for x in x_sm], dtype=opt_grp["parameter_space_type"])
    h5_concat_dataset(dset, data)

    f.close()


def save_stats_to_h5(
    opt_id,
    problem_id,
    epoch,
    fpath,
    logger,
    stats,
):
    """
    Save optimizer statistics to an HDF5 file 'fpath'.
    """

    f = h5py.File(fpath, "a")

    opt_grp = h5_get_group(f, opt_id)

    dtype = np.dtype(
        {"names": [k for k in sorted(stats)], "formats": [np.float64] * len(stats)}
    )

    opt_stats_grp = h5_get_group(opt_grp, "optimizer_stats")
    opt_stats_epoch_grp = h5_get_group(opt_stats_grp, f"{epoch}")

    if logger is not None:
        logger.info(
            f"Saving optimizer stats for problem id {problem_id} epoch {epoch} to {fpath}."
        )

    dset = h5_get_dataset(
        opt_stats_epoch_grp,
        "stats",
        maxshape=(None,),
        dtype=dtype,
    )
    h5_concat_dataset(
        dset,
        np.array([tuple(map(float, [stats[k] for k in sorted(stats)]))], dtype=dtype),
    )

    f.close()


def init_h5(
    opt_id,
    problem_ids,
    has_problem_ids,
    parameter_space,
    objective_names,
    feature_dtypes,
    constraint_names,
    problem_parameters,
    metadata,
    random_seed,
    fpath,
    surrogate_mean_variance=False,
):
    """
    Save progress and settings to an HDF5 file 'fpath'.
    """

    f = h5py.File(fpath, "a")
    h5_init_opt_group(
        f,
        opt_id,
        objective_names,
        feature_dtypes,
        constraint_names,
        problem_parameters,
        parameter_space,
        problem_ids,
        has_problem_ids,
        metadata,
        random_seed,
        surrogate_mean_variance=surrogate_mean_variance,
    )

    f.close()


def eval_obj_fun_sp(
    obj_fun,
    pp,
    param_space,
    nested_parameter_space,
    obj_fun_args,
    problem_id,
    space_vals,
):
    """
    Objective function evaluation (single problem).
    """

    this_space_vals = space_vals[problem_id]
    if nested_parameter_space:
        this_pp = update_nested_dict(
            pp.unflatten(), param_space.unflatten(this_space_vals)
        )
    else:
        this_pp = {}
        this_pp.update(
            [
                (item.name, int(item.value) if item.is_integer else item.value)
                for item in pp.items
            ]
        )
        this_pp.update(
            [
                (param_name, this_space_vals[i])
                for i, param_name in enumerate(param_space.parameter_names)
            ]
        )
    if obj_fun_args is None:
        obj_fun_args = ()
    t = time.time()
    result = obj_fun(this_pp, *obj_fun_args)
    return {problem_id: result, "time": time.time() - t}


def eval_obj_fun_mp(
    obj_fun,
    pp,
    param_space,
    nested_parameter_space,
    obj_fun_args,
    problem_ids,
    space_vals,
):
    """
    Objective function evaluation (multiple problems).
    """

    mpp = {}
    for problem_id in problem_ids:
        this_space_vals = space_vals[problem_id]
        if nested_parameter_space:
            this_pp = update_nested_dict(
                pp.unflatten(), param_space.unflatten(this_space_vals)
            )
        else:
            this_pp = {}
            this_pp.update(
                [
                    (item.name, int(item.value) if item.is_integer else item.value)
                    for item in pp.items
                ]
            )
            this_pp.update(
                [
                    (param_name, this_space_vals[i])
                    for i, param_name in enumerate(param_space.parameter_names)
                ]
            )
        mpp[problem_id] = this_pp

    if obj_fun_args is None:
        obj_fun_args = ()

    t = time.time()
    result_dict = obj_fun(mpp, *obj_fun_args)
    result_dict["time"] = time.time() - t

    return result_dict


def reducefun(xs):
    return xs[0]


def dopt_init(
    dopt_params,
    worker=None,
    nprocs_per_worker=None,
    verbose=False,
    initialize_strategy=False,
):
    objfun = None
    objfun_name = dopt_params.get("obj_fun_name", None)
    if distwq.is_worker:
        if objfun_name is not None:
            objfun = import_object_by_path(objfun_name)
        else:
            objfun_init_name = dopt_params.get("obj_fun_init_name", None)
            objfun_init_args = dopt_params.get("obj_fun_init_args", None)
            if objfun_init_name is None:
                raise RuntimeError("dmosopt.soptinit: objfun is not provided")
            objfun_init = import_object_by_path(objfun_init_name)
            objfun = objfun_init(**objfun_init_args, worker=worker)
    else:
        ctrl_init_fun_name = dopt_params.get("controller_init_fun_name", None)
        ctrl_init_fun_args = dopt_params.get("controller_init_fun_args", {})
        if ctrl_init_fun_name is not None:
            ctrl_init_fun = import_object_by_path(ctrl_init_fun_name)
            ctrl_init_fun(**ctrl_init_fun_args)

    dopt_params["obj_fun"] = objfun
    reducefun_name = dopt_params.get("reduce_fun_name", None)
    if reducefun_name is not None:
        reduce_fun = import_object_by_path(reducefun_name)
        dopt_params["reduce_fun"] = reduce_fun
    else:
        # If using MPI with 1 process per worker, then each worker
        # will always return a list containing one element, and
        # therefore we can apply a reduce function that returns the
        # first element of the list.
        if distwq.is_controller and distwq.workers_available:
            if nprocs_per_worker == 1:
                dopt_params["reduce_fun"] = reducefun
            elif nprocs_per_worker > 1:
                raise RuntimeError(
                    "When nprocs_per_workers > 1, a reduce function must be specified."
                )

    dopt = DistOptimizer(**dopt_params, verbose=verbose)
    if initialize_strategy:
        dopt.initialize_strategy()
    dopt_dict[dopt.opt_id] = dopt
    return dopt


def dopt_ctrl(controller, dopt_params, nprocs_per_worker, verbose=True):
    """Controller for distributed surrogate optimization."""
    logger = logging.getLogger(dopt_params["opt_id"])
    logger.info("Initializing optimization controller...")
    if verbose:
        logger.setLevel(logging.INFO)
    dopt_params["controller"] = controller
    dopt = dopt_init(
        dopt_params,
        nprocs_per_worker=nprocs_per_worker,
        verbose=verbose,
        initialize_strategy=True,
    )
    logger.info(f"Optimizing for {dopt.n_epochs} epochs...")

    if dopt.n_epochs <= 0:
        # initial sampling only
        return dopt.run_epoch(completed_epoch=True)

    while dopt.epoch_count < dopt.n_epochs:
        dopt.run_epoch()


def dopt_work(worker, dopt_params, verbose=False, debug=False):
    """Worker for distributed surrogate optimization."""
    if worker.worker_id > 1 and (not debug):
        verbose = False
    dopt_init(dopt_params, worker=worker, verbose=verbose, initialize_strategy=False)


def eval_fun(opt_id, *args):
    return dopt_dict[opt_id].eval_fun(*args)


def run(
    dopt_params,
    time_limit=None,
    feasible=True,
    return_features=False,
    return_constraints=False,
    spawn_workers=False,
    sequential_spawn=False,
    spawn_startup_wait=None,
    spawn_executable=None,
    spawn_args=[],
    nprocs_per_worker=1,
    collective_mode="gather",
    verbose=True,
    worker_debug=False,
):
    if distwq.is_controller:
        distwq.run(
            fun_name="dopt_ctrl",
            module_name="dmosopt.dmosopt",
            verbose=verbose,
            args=(
                dopt_params,
                nprocs_per_worker,
                verbose,
            ),
            worker_grouping_method="spawn" if spawn_workers else "split",
            broker_is_worker=True,
            sequential_spawn=sequential_spawn,
            spawn_startup_wait=spawn_startup_wait,
            spawn_executable=spawn_executable,
            spawn_args=spawn_args,
            nprocs_per_worker=nprocs_per_worker,
            collective_mode=collective_mode,
            time_limit=time_limit,
        )
        opt_id = dopt_params["opt_id"]
        dopt = dopt_dict[opt_id]
        dopt.print_best()
        return dopt.get_best(
            feasible=feasible,
            return_features=return_features,
            return_constraints=return_constraints,
        )
    else:
        if "file_path" in dopt_params:
            del dopt_params["file_path"]
        if "save" in dopt_params:
            del dopt_params["save"]
        distwq.run(
            fun_name="dopt_work",
            module_name="dmosopt.dmosopt",
            broker_fun_name=dopt_params.get("broker_fun_name", None),
            broker_module_name=dopt_params.get("broker_module_name", None),
            verbose=verbose,
            args=(
                dopt_params,
                verbose,
                worker_debug,
            ),
            worker_grouping_method="spawn" if spawn_workers else "split",
            broker_is_worker=True,
            sequential_spawn=sequential_spawn,
            spawn_startup_wait=spawn_startup_wait,
            spawn_executable=spawn_executable,
            spawn_args=spawn_args,
            nprocs_per_worker=nprocs_per_worker,
            collective_mode=collective_mode,
            time_limit=time_limit,
        )
        return None
