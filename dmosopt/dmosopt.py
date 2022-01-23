import os, sys, importlib, logging, pprint, copy
from functools import partial
from collections import namedtuple
import numpy as np
from numpy.random import default_rng
import distwq
import dmosopt.MOASMO as opt
from dmosopt.datatypes import OptProblem, ParamSpec, EvalEntry, EvalRequest
from dmosopt.termination import MultiObjectiveStdTermination

logger = logging.getLogger('dmosopt')

try:
    import h5py
except ImportError as e:
    logger.warning(f'unable to import h5py: {e}')

sopt_dict = {}

def anyclose(a, b, rtol=1e-4, atol=1e-4):
    for i in range(b.shape[0]):
        if np.allclose(a, b[i, :]):
            return True
    return False
        
    
class SOptStrategy():
    def __init__(self, prob, n_initial=10, initial=None, initial_maxiter=5, initial_method="glp",
                 population_size=100, resample_fraction=0.25, num_generations=100,
                 crossover_rate=0.9, mutation_rate=None, di_crossover=1., di_mutation=20.,
                 surrogate_method='gpr', surrogate_options={'anisotropic': False, 'optimizer': "sceua"},
                 distance_metric=None,  optimizer="nsga2",
                 feasibility_model=False, termination_conditions=None, local_random=None,
                 logger=None):
        if local_random is None:
            local_random = default_rng()
        self.local_random = local_random
        self.logger = logger
        self.feasibility_model = feasibility_model
        self.surrogate_options = surrogate_options
        self.surrogate_method = surrogate_method
        self.optimizer = optimizer
        self.distance_metric = distance_metric
        self.prob = prob
        self.completed = []
        self.reqs = []
        if initial is None:
            self.x = None
            self.y = None
            self.f = None
            self.c = None
        else:
            epochs, self.x, self.y, self.f, self.c = initial
        self.resample_fraction = resample_fraction
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.di_mutation = di_mutation
        self.di_crossover = di_crossover
        self.termination = None
        if termination_conditions:
            termination_kwargs = { 'x_tol': 1e-6,
                                   'f_tol': 0.0001,
                                   'nth_gen': 5,
                                   'n_max_gen': num_generations,
                                   'n_last': 50 }
            if isinstance(termination_conditions, dict):
                termination_kwargs.update(termination_conditions)
            self.termination = MultiObjectiveStdTermination(prob, **termination_kwargs)
        nPrevious = None
        if self.x is not None:
            nPrevious = self.x.shape[0]
        xinit = opt.xinit(n_initial, prob.dim, prob.n_objectives, prob.lb, prob.ub, nPrevious=nPrevious,
                          maxiter=initial_maxiter, method=initial_method, local_random=self.local_random,
                          logger=self.logger)
        self.reqs = []
        if xinit is not None:
            assert(xinit.shape[1] == prob.dim)
            if initial is None:
                self.reqs = [ EvalRequest(xinit[i,:], None) for i in range(xinit.shape[0]) ]
            else:
                self.reqs = list(filter(lambda req: not anyclose(req.parameters, self.x), 
                                        [ EvalRequest(xinit[i,:], None) for i in range(xinit.shape[0]) ]))
            
    def get_next_request(self):
        req = None
        if len(self.reqs) > 0:
            req = self.reqs.pop(0)
        return req

    def complete_request(self, x, y, epoch=None, f=None, c=None, pred=None):
        assert(x.shape[0] == self.prob.dim)
        assert(y.shape[0] == self.prob.n_objectives)
        entry = EvalEntry(epoch,x,y,f,c,pred)
        self.completed.append(entry)
        return entry
    
    def step(self, return_sm=False):
        if len(self.completed) > 0:
            x_completed = np.vstack([x.parameters for x in self.completed])
            y_completed = np.vstack([x.objectives for x in self.completed])

            f_completed = None
            if self.prob.n_features is not None:
                f_completed = np.concatenate([x.features for x in self.completed], axis=None)
            c_completed = None
            if self.prob.n_constraints is not None:
                c_completed = np.vstack([x.constraints for x in self.completed])
            
            assert(x_completed.shape[1] == self.prob.dim)
            assert(y_completed.shape[1] == self.prob.n_objectives)
            if self.prob.n_constraints is not None:
                assert(c_completed.shape[1] == self.prob.n_constraints)
                
            if self.x is None:
                self.x = x_completed
                self.y = y_completed
                self.f = f_completed
                self.c = c_completed
            else:
                self.x = np.vstack((self.x, x_completed))
                self.y = np.vstack((self.y, y_completed))

                if self.prob.n_features is not None:
                    self.f = np.concatenate((self.f, f_completed))
                if self.prob.n_constraints is not None:
                    self.c = np.vstack((self.c, c_completed))
            self.completed = []
        optimizer_kwargs={'gen': self.num_generations,
                          'crossover_rate': self.crossover_rate,
                          'mutation_rate': self.mutation_rate,
                          'di_crossover': self.di_crossover,
                          'di_mutation': self.di_mutation}
        if self.distance_metric is not None:
            optimizer_kwargs['distance_metric'] = self.distance_metric
        if self.termination is not None:
            self.termination.reset()

        x_sm = None
        y_sm = None
        x_resample = None
        res = opt.onestep(self.prob.dim, self.prob.n_objectives,
                          self.prob.lb, self.prob.ub, self.resample_fraction,
                          self.x, self.y, self.c, pop=self.population_size,
                          optimizer=self.optimizer,
                          optimizer_kwargs=optimizer_kwargs,
                          surrogate_method=self.surrogate_method,
                          surrogate_options=self.surrogate_options,
                          feasibility_model=self.feasibility_model,
                          termination=self.termination,
                          local_random=self.local_random,
                          logger=self.logger, return_sm=return_sm)
        
        if return_sm:
            x_resample, y_pred, gen_index, x_sm, y_sm = res
        else:
            x_resample, y_pred = res
            
        for i in range(x_resample.shape[0]):
            self.reqs.append(EvalRequest(x_resample[i,:], y_pred[i,:]))
            
        if return_sm:
            return x_resample, y_pred, gen_index, x_sm, y_sm
        else:
            return x_resample, y_pred
        
    def get_best_evals(self, feasible=True):
        if self.x is not None:
            bestx, besty, bestf, bestc, beste, perm =  opt.get_best(self.x, self.y, self.f, self.c, self.prob.dim, 
                                                                    self.prob.n_objectives, feasible=feasible)
            return bestx, besty, bestf, bestc
        else:
            return None, None, None, None

    def get_evals(self, return_features=False, return_constraints=False):
        if return_features and return_constraints:
            return (self.x, self.y, self.f, self.c)
        elif return_features:
            return (self.x, self.y, self.f)
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

class DistOptimizer():
    def __init__(
        self,
        opt_id,
        obj_fun,
        obj_fun_args=None,
        objective_names=None,
        feature_dtypes=None,
        constraint_names=None,
        n_initial=10,
        initial_maxiter=5,
        initial_method="glp",
        verbose=False,
        reduce_fun=None,
        reduce_fun_args=None,
        problem_ids=None,
        problem_parameters=None,
        space=None,
        population_size=100,
        num_generations=200,
        resample_fraction=0.25,
        mutation_rate=None,
        crossover_rate=0.9,
        di_crossover=1.0,
        di_mutation=20.0,
        distance_metric=None,
        n_epochs=10,
        save_eval=10,
        file_path=None,
        save=False,
        save_surrogate_eval=False,
        metadata=None,
        surrogate_method="gpr",
        surrogate_options={'anisotropic': False,
                           'optimizer': "sceua" },
        optimizer="nsga2",
        local_random=None,
        feasibility_model=False,
        termination_conditions=None,
        **kwargs
    ):
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

        self.opt_id = opt_id
        self.verbose = verbose
        self.population_size = population_size
        self.num_generations = num_generations
        self.resample_fraction = resample_fraction
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.di_crossover = di_crossover
        self.di_mutation = di_mutation
        self.distance_metric = distance_metric
        self.surrogate_method = surrogate_method
        self.surrogate_options = surrogate_options
        self.optimizer = optimizer
        self.feasibility_model = feasibility_model
        self.termination_conditions = termination_conditions
        self.metadata = metadata
        self.local_random = local_random
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

        dim = 0
        param_names, is_int, lo_bounds, hi_bounds = [], [], [], []
        if space is not None:
            dim = len(space)
            for parm, conf in space.items():
                param_names.append(parm)
                lo, hi = conf
                is_int.append(type(lo) == int and type(hi) == int)
                lo_bounds.append(lo)
                hi_bounds.append(hi)
                if parm in problem_parameters:
                    del(problem_parameters[parm])
        old_evals = {}
        max_epoch = -1
        if file_path is not None:
            if os.path.isfile(file_path):
                max_epoch, old_evals, param_names, is_int, lo_bounds, hi_bounds, objective_names, feature_dtypes, constraint_names, problem_parameters, problem_ids = \
                    init_from_h5(file_path, param_names, opt_id, self.logger)
                    

        assert(dim > 0)
        param_spec = ParamSpec(bound1=np.asarray(lo_bounds), bound2=np.asarray(hi_bounds), is_integer=is_int)
        self.param_spec = param_spec

        assert(objective_names is not None)
        self.objective_names = objective_names
        
        has_problem_ids = (problem_ids is not None)
        if not has_problem_ids:
            problem_ids = set([0])

        self.n_initial = n_initial
        self.initial_maxiter = initial_maxiter
        self.initial_method = initial_method
        self.problem_parameters, self.param_names = problem_parameters, param_names
        self.is_int = is_int
        self.file_path, self.save = file_path, save

        self.start_epoch = max_epoch
        if self.start_epoch != 0:
            self.start_epoch += 1
            
        self.n_epochs = n_epochs
        self.save_eval = save_eval
        self.save_surrogate_eval = save_surrogate_eval
        
        self.obj_fun_args = obj_fun_args
        if has_problem_ids:
            self.eval_fun = partial(eval_obj_fun_mp, obj_fun, self.problem_parameters, self.param_names, self.is_int, self.obj_fun_args, problem_ids)
        else:
            self.eval_fun = partial(eval_obj_fun_sp, obj_fun, self.problem_parameters, self.param_names, self.is_int, self.obj_fun_args, 0)
            
        self.reduce_fun = reduce_fun
        self.reduce_fun_args = reduce_fun_args
        
        self.eval_reqs = { problem_id: {} for problem_id in problem_ids }
        self.old_evals = old_evals

        self.has_problem_ids = has_problem_ids
        self.problem_ids = problem_ids

        self.optimizer_dict = {}
        self.storage_dict = {}

        self.feature_dtypes = feature_dtypes
        self.feature_names = None
        if feature_dtypes is not None:
            self.feature_names = [dt[0] for dt in feature_dtypes]

        self.constraint_names = constraint_names

        if file_path is not None:
            if not os.path.isfile(file_path):
                init_h5(self.opt_id, self.problem_ids, self.has_problem_ids,
                        self.param_spec, self.param_names, self.objective_names, 
                        self.feature_dtypes, self.constraint_names,
                        self.problem_parameters, self.metadata, self.file_path)
                    


    def init_strategy(self):
        opt_prob = OptProblem(self.param_names, self.objective_names, self.feature_dtypes,
                              self.constraint_names, self.param_spec, self.eval_fun,
                              logger=self.logger)
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
                    old_eval_fs = [e.features for e in self.old_evals[problem_id]]
                    f = np.concatenate(old_eval_fs, axis=None)
                c = None
                if self.constraint_names is not None:
                    old_eval_cs = [e.constraints for e in self.old_evals[problem_id]]
                    c = np.vstack(old_eval_cs)
                initial = (epochs, x, y, f, c)

            opt_strategy = SOptStrategy(opt_prob, self.n_initial, initial=initial, 
                                        population_size=self.population_size, 
                                        resample_fraction=self.resample_fraction,
                                        num_generations=self.num_generations,
                                        initial_maxiter=self.initial_maxiter,
                                        initial_method=self.initial_method,
                                        mutation_rate=self.mutation_rate,
                                        crossover_rate=self.crossover_rate,
                                        di_mutation=self.di_mutation,
                                        di_crossover=self.di_crossover,
                                        distance_metric=self.distance_metric,
                                        surrogate_method=self.surrogate_method,
                                        surrogate_options=self.surrogate_options,
                                        optimizer=self.optimizer,
                                        feasibility_model=self.feasibility_model,
                                        termination_conditions=self.termination_conditions,
                                        local_random=self.local_random,
                                        logger=self.logger)
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
                y_pred_completed = map(lambda x: [np.nan]*n if x is None else x,
                                       [x.prediction for x in storage_evals])
                f_completed = None
                if self.feature_names is not None:
                    f_completed = [x.features for x in storage_evals]
                c_completed = None
                if self.constraint_names is not None:
                    c_completed = [x.constraints for x in storage_evals]
                finished_evals[problem_id] = (epochs_completed, x_completed, y_completed, f_completed, c_completed, y_pred_completed)
                self.storage_dict[problem_id] = []

        if len(finished_evals) > 0:
            save_to_h5(self.opt_id, self.problem_ids, self.has_problem_ids,
                       self.param_names, self.objective_names, self.feature_dtypes, self.constraint_names,
                       self.param_spec, finished_evals, self.problem_parameters, 
                       self.metadata, self.file_path, self.logger)


    def save_surrogate_evals(self, problem_id, epoch, gen_index, x_sm, y_sm):
        """Store results of surrogate evals to file."""
        if x_sm.shape[0] > 0:
            save_surrogate_evals_to_h5(self.opt_id, problem_id, 
                                       self.param_names, self.objective_names, 
                                       epoch, gen_index, x_sm, y_sm,
                                       self.file_path, self.logger)

    def get_best(self, feasible=True, return_features=False, return_constraints=False):
        best_results = {}
        for problem_id in self.problem_ids:
            best_x, best_y, best_f, best_c = self.optimizer_dict[problem_id].get_best_evals(feasible=feasible)
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
        best_results = self.get_best(feasible=feasible, return_features=True, return_constraints=True)
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
                    res_i = { k: res_dict[k][i] for k in res_dict }
                    prms_i = { k: prms_dict[k][i] for k in prms_dict }
                    constr_i = None
                    if constr_dict is not None:
                        constr_i = { k: constr_dict[k][i] for k in constr_dict }
                    ftrs_i = None
                    if ftrs is not None:
                        ftrs_i = ftrs[i]
                        lftrs_i = dict(zip(ftrs_i.dtype.names, ftrs_i))
                    if (ftrs_i is not None) and (constr_i is not None):
                        self.logger.info(f"Best eval {i} so far for id {problem_id}: {res_i}@{prms_i} [{lftrs_i}] [constr: {constr_i}]")
                    elif (constr_i is not None): 
                        self.logger.info(f"Best eval {i} so far for id {problem_id}: {res_i}@{prms_i} [constr: {constr_i}]")
                    elif (ftrs_i is not None): 
                        self.logger.info(f"Best eval {i} so far for id {problem_id}: {res_i}@{prms_i} [{lftrs_i}]")
                    else:
                        self.logger.info(f"Best eval {i} so far for id {problem_id}: {res_i}@{prms_i}")
        else:
            prms, res, ftrs, constr = best_results
            prms_dict = dict(prms)
            res_dict = dict(res)
            n_res = next(iter(res_dict.values())).shape[0]
            constr_dict = None
            if constr is not None:
                constr_dict = dict(constr)
            for i in range(n_res):
                res_i = { k: res_dict[k][i] for k in res_dict }
                prms_i = { k: prms_dict[k][i] for k in prms_dict }
                constr_i = None
                if constr_dict is not None:
                    constr_i = { k: constr_dict[k][i] for k in constr_dict }
                ftrs_i = None
                if ftrs is not None:
                    ftrs_i = ftrs[i]
                if (ftrs_i is not None) and (constr_i is not None):
                    self.logger.info(f"Best eval {i} so far: {res_i}@{prms_i} [{ftrs_i}] [constr: {constr_i}]")
                elif (constr_i is not None): 
                    self.logger.info(f"Best eval {i} so far: {res_i}@{prms_i} [constr: {constr_i}]")
                elif (ftrs_i is not None): 
                    self.logger.info(f"Best eval {i} so far: {res_i}@{prms_i} [{ftrs_i}]")
                else:
                    self.logger.info(f"Best eval {i} so far: {res_i}@{prms_i}")
            

def h5_get_group (h, groupname):
    if groupname in h.keys():
        g = h[groupname]
    else:
        g = h.create_group(groupname)
    return g

def h5_get_dataset (g, dsetname, **kwargs):
    if dsetname in g.keys():
        dset = g[dsetname]
    else:
        dset = g.create_dataset(dsetname, (0,), **kwargs)
    return dset

def h5_concat_dataset(dset, data):
    dsize = dset.shape[0]
    newshape = (dsize+len(data),)
    dset.resize(newshape)
    dset[dsize:] = data
    return dset


def h5_init_types(f, opt_id, param_names, objective_names, feature_dtypes, constraint_names, problem_parameters, spec):
    
    opt_grp = h5_get_group(f, opt_id)

    objective_keys = set(objective_names)
    feature_keys = None
    if feature_dtypes is not None:
        feature_keys = [ feature_dtype[0] for feature_dtype in feature_dtypes ]

    # create HDF5 types for the objectives and features
    objective_mapping = { name: idx for (idx, name) in
                          enumerate(objective_keys) }
    feature_mapping = None
    if feature_keys is not None:
        feature_mapping = { name: idx for (idx, name) in
                            enumerate(feature_keys) }

    dt = h5py.enum_dtype(objective_mapping, basetype=np.uint16)
    opt_grp['objective_enum'] = dt

    dt = np.dtype([("objective", opt_grp['objective_enum'])])
    opt_grp['objective_spec_type'] = dt

    dt = np.dtype({ 'names': objective_names,
                    'formats': [np.float32]*len(objective_names) })
    opt_grp['objective_type'] = dt

    dset = h5_get_dataset(opt_grp, 'objective_spec', maxshape=(len(objective_names),),
                          dtype=opt_grp['objective_spec_type'].dtype)
    dset.resize((len(objective_names),))
    a = np.zeros(len(objective_names), dtype=opt_grp['objective_spec_type'].dtype)
    for idx, parm in enumerate(objective_names):
        a[idx]["objective"] = objective_mapping[parm]
    dset[:] = a

    if feature_mapping is not None:
        dt = h5py.enum_dtype(feature_mapping, basetype=np.uint16)
        opt_grp['feature_enum'] = dt

        dt = np.dtype([("feature", opt_grp['feature_enum'])])
        opt_grp['feature_spec_type'] = dt

        dt = np.dtype(feature_dtypes)
        opt_grp['feature_type'] = dt

        dset = h5_get_dataset(opt_grp, 'feature_spec', maxshape=(len(feature_keys),),
                              dtype=opt_grp['feature_spec_type'].dtype)
        dset.resize((len(feature_keys),))
        a = np.zeros(len(feature_keys), dtype=opt_grp['feature_spec_type'].dtype)
        for idx, parm in enumerate(feature_keys):
            a[idx]["feature"] = feature_mapping[parm]
        dset[:] = a

    # create HDF5 types for the constraints
    constr_keys = None
    if constraint_names is not None:
        constr_keys = set(constraint_names)

        constr_mapping = { name: idx for (idx, name) in
                           enumerate(constr_keys) }

        dt = h5py.enum_dtype(constr_mapping, basetype=np.uint16)
        opt_grp['constraint_enum'] = dt

        dt = np.dtype([("constraint", opt_grp['constraint_enum'])])
        opt_grp['constraint_spec_type'] = dt

        dt = np.dtype({ 'names': constraint_names,
                        'formats': [np.float32]*len(constraint_names) })
        opt_grp['constraint_type'] = dt

        dset = h5_get_dataset(opt_grp, 'constraint_spec', maxshape=(len(constraint_names),),
                              dtype=opt_grp['constraint_spec_type'].dtype)
        dset.resize((len(constraint_names),))
        a = np.zeros(len(constraint_names), dtype=opt_grp['constraint_spec_type'].dtype)
        for idx, parm in enumerate(constraint_names):
            a[idx]["constraint"] = constr_mapping[parm]
            dset[:] = a

    # create HDF5 types describing the parameter specification
    param_keys = set(param_names)
    param_keys.update(problem_parameters.keys())

    param_mapping = { name: idx for (idx, name) in
                      enumerate(param_keys) }

    dt = h5py.enum_dtype(param_mapping, basetype=np.uint16)
    opt_grp['parameter_enum'] = dt

    dt = np.dtype({ 'names': param_names, 'formats': [np.float32]*len(param_names) })
    opt_grp['parameter_space_type'] = dt
    
    dt = np.dtype([("parameter", opt_grp['parameter_enum']),
                   ("value", np.float32)])
    opt_grp['problem_parameters_type'] = dt

    dset = h5_get_dataset(opt_grp, 'problem_parameters', maxshape=(len(param_mapping),),
                          dtype=opt_grp['problem_parameters_type'].dtype)
    dset.resize((len(param_mapping),))
    a = np.zeros(len(param_mapping), dtype=opt_grp['problem_parameters_type'].dtype)
    idx = 0
    for idx, (parm, val) in enumerate(problem_parameters.items()):
        a[idx]["parameter"] = param_mapping[parm]
        a[idx]["value"] = val
    dset[:] = a
    
    dt = np.dtype([("parameter", opt_grp['parameter_enum']),
                   ("is_integer", np.bool),
                   ("lower", np.float32),
                   ("upper", np.float32)])
    opt_grp['parameter_spec_type'] = dt

    is_integer = np.asarray(spec.is_integer, dtype=np.bool)
    upper = np.asarray(spec.bound2, dtype=np.float32)
    lower = np.asarray(spec.bound1, dtype=np.float32)

    dset = h5_get_dataset(opt_grp, 'parameter_spec', maxshape=(len(param_names),),
                          dtype=opt_grp['parameter_spec_type'].dtype)
    dset.resize((len(param_names),))
    a = np.zeros(len(param_names), dtype=opt_grp['parameter_spec_type'].dtype)
    for idx, (parm, is_int, hi, lo) in enumerate(zip(param_names, is_integer, upper, lower)):
        a[idx]["parameter"] = param_mapping[parm]
        a[idx]["is_integer"] = is_int
        a[idx]["lower"] = lo
        a[idx]["upper"] = hi
    dset[:] = a

    

    
def h5_load_raw(input_file, opt_id):
    ## N is number of trials
    ## M is number of hyperparameters
    f = h5py.File(input_file, 'r')
    opt_grp = h5_get_group(f, opt_id)

    objective_enum_dict = h5py.check_enum_dtype(opt_grp['objective_enum'].dtype)
    objective_idx_dict = { parm: idx for parm, idx in objective_enum_dict.items() }
    objective_name_dict = { idx: parm for parm, idx in objective_idx_dict.items() }
    n_objectives = len(objective_enum_dict)
    objective_names = [ objective_name_dict[spec[0]]
                        for spec in iter(opt_grp['objective_spec']) ]

    n_constraints = 0
    constraint_names = None
    if 'constraint_enum' in opt_grp:
        constraint_enum_dict = h5py.check_enum_dtype(opt_grp['constraint_enum'].dtype)
        constraint_idx_dict = { parm: idx for parm, idx in constraint_enum_dict.items() }
        constraint_name_dict = { idx: parm for parm, idx in constraint_idx_dict.items() }
        n_constraints = len(constraint_enum_dict)
        constraint_names = [ constraint_name_dict[spec[0]]
                             for spec in iter(opt_grp['constraint_spec']) ]

    n_features = 0
    feature_names = None
    if 'feature_enum' in opt_grp:
        feature_enum_dict = h5py.check_enum_dtype(opt_grp['feature_enum'].dtype)
        feature_idx_dict = { parm: idx for parm, idx in feature_enum_dict.items() }
        feature_name_dict = { idx: parm for parm, idx in feature_idx_dict.items() }
        n_features = len(feature_enum_dict)
        feature_names = [ feature_name_dict[spec[0]]
                          for spec in iter(opt_grp['feature_spec']) ]


    parameter_enum_dict = h5py.check_enum_dtype(opt_grp['parameter_enum'].dtype)
    parameters_idx_dict = { parm: idx for parm, idx in parameter_enum_dict.items() }
    parameters_name_dict = { idx: parm for parm, idx in parameters_idx_dict.items() }
    
    problem_parameters = { parameters_name_dict[idx]: val
                           for idx, val in opt_grp['problem_parameters'] }
    parameter_specs = [ (parameters_name_dict[spec[0]], tuple(spec)[1:])
                        for spec in iter(opt_grp['parameter_spec']) ]

    problem_ids = None
    if 'problem_ids' in opt_grp:
        problem_ids = set(opt_grp['problem_ids'])
    
    M = len(parameter_specs)
    P = n_objectives

    raw_results = {}
    for problem_id in problem_ids if problem_ids is not None else [0]:
        if str(problem_id) in opt_grp:
            raw_results[problem_id] = { 'objectives': opt_grp[str(problem_id)]['objectives'][:],
                                        'parameters': opt_grp[str(problem_id)]['parameters'][:] }
            if 'features' in opt_grp[str(problem_id)]:
                raw_results[problem_id]['features'] = opt_grp[str(problem_id)]['features'][:]
            if 'constraints' in opt_grp[str(problem_id)]:
                raw_results[problem_id]['constraints'] = opt_grp[str(problem_id)]['constraints'][:]
            if 'epochs' in opt_grp[str(problem_id)]:
                raw_results[problem_id]['epochs'] = opt_grp[str(problem_id)]['epochs'][:]
            if 'predictions' in opt_grp[str(problem_id)]:
                raw_results[problem_id]['predictions'] = opt_grp[str(problem_id)]['predictions'][:]
                                        
    f.close()
    
    param_names = []
    is_integer = []
    lower = []
    upper = []
    for parm, spec in parameter_specs:
        param_names.append(parm)
        is_int, lo, hi = spec
        is_integer.append(is_int)
        lower.append(lo)
        upper.append(hi)
        
    raw_spec = (is_integer, lower, upper)
    info = { 'objectives': objective_names,
             'features': feature_names,
             'constraints': constraint_names,
             'params': param_names,                            
             'problem_parameters': problem_parameters,
             'problem_ids': problem_ids }
    
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
    (param_spec, function_eval, dict, prev_best)
      where prev_best: np.array[result, param1, param2, ...]
    """
    raw_spec, raw_problem_results, info = h5_load_raw(file_path, opt_id)
    is_integer, lo_bounds, hi_bounds = raw_spec
    param_names = info['params']
    objective_names = info['objectives']
    feature_names = info['features']
    constraint_names = info['constraints']
    n_objectives = len(objective_names)
    n_features = 0
    if feature_names is not None:
        n_features = len(feature_names)
    n_constraints = 0
    if constraint_names is not None:
        n_constraints = len(constraint_names)
    spec = ParamSpec(bound1=lo_bounds, bound2=hi_bounds, is_integer=is_integer)
    evals = { problem_id: [] for problem_id in raw_problem_results }
    for problem_id in raw_problem_results:
        problem_evals = []
        raw_results = raw_problem_results[problem_id]
        epochs = raw_results.get('epochs', None)
        ys = raw_results['objectives']
        xs = raw_results['parameters']
        fs = raw_results.get('features', None)
        cs = raw_results.get('constraints', None)
        ypreds = raw_results.get('predictions', None)
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
    return raw_spec, spec, evals, info
    
def init_from_h5(file_path, param_names, opt_id, logger=None):        
    # Load progress and settings from file, then compare each
    # restored setting with settings specified by args (if any)
    old_raw_spec, old_spec, old_evals, info = h5_load_all(file_path, opt_id)
    saved_params = info['params']
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
        # Switching params being optimized over would throw off the optimizer.
        # Must use restore params from specified
        if logger is not None:
            logger.warning(
                f"Saved params {saved_params} differ from currently specified "
                f"{param_names}. Using saved.")
    params = saved_params
    raw_spec = old_raw_spec
    is_int, lo_bounds, hi_bounds = raw_spec
    if len(params) != len(is_int):
        raise ValueError(
            f"Params {params} and spec {raw_spec} are of different length"
            )
    problem_parameters = info['problem_parameters']
    objective_names = info['objectives']
    feature_names = info['features']
    constraint_names = info['constraints']
    problem_ids = info['problem_ids'] if 'problem_ids' in info else None

    
    return max_epoch, old_evals, params, is_int, lo_bounds, hi_bounds, objective_names, feature_names, constraint_names, problem_parameters, problem_ids


def save_to_h5(opt_id, problem_ids, has_problem_ids, param_names, objective_names, feature_names, constraint_names, spec, evals, problem_parameters, metadata, fpath, logger):
    """
    Save progress and settings to an HDF5 file 'fpath'.
    """

    f = h5py.File(fpath, "a")
    if opt_id not in f.keys():
        h5_init_types(f, opt_id, param_names, objective_names, problem_parameters, constraint_names, spec)
        opt_grp = h5_get_group(f, opt_id)
        if metadata is not None:
            opt_grp['metadata'] = metadata
        if has_problem_ids:
            opt_grp['problem_ids'] = np.asarray(list(problem_ids), dtype=np.int32)
        else:
            opt_grp['problem_ids'] = np.asarray([0], dtype=np.int32)
            
    opt_grp = h5_get_group(f, opt_id)

    parameter_enum_dict = h5py.check_enum_dtype(opt_grp['parameter_enum'].dtype)
    parameters_idx_dict = { parm: idx for parm, idx in parameter_enum_dict.items() }
    parameters_name_dict = { idx: parm for parm, idx in parameters_idx_dict.items() }

    for problem_id in problem_ids:

        prob_evals_epoch, prob_evals_x, prob_evals_y, prob_evals_f, prob_evals_c, prob_evals_y_pred = evals[problem_id]
        opt_prob = h5_get_group(opt_grp, str(problem_id))

        if logger is not None:
            logger.info(f"Saving {len(prob_evals_y)} evaluations for problem id {problem_id} to {fpath}.")

        dset = h5_get_dataset(opt_prob, 'epochs', maxshape=(None,),
                              dtype=np.uint32)
        data = np.asarray(prob_evals_epoch, dtype=np.uint32)
        h5_concat_dataset(dset, data)

        dset = h5_get_dataset(opt_prob, 'objectives', maxshape=(None,),
                              dtype=opt_grp['objective_type'])
        data = np.array([tuple(y) for y in prob_evals_y], dtype=opt_grp['objective_type'])
        h5_concat_dataset(dset, data)

        dset = h5_get_dataset(opt_prob, 'parameters', maxshape=(None,),
                              dtype=opt_grp['parameter_space_type'])
        data = np.array([tuple(x) for x in prob_evals_x], dtype=opt_grp['parameter_space_type'])
        h5_concat_dataset(dset, data)

        if prob_evals_f is not None:
            dset = h5_get_dataset(opt_prob, 'features', maxshape=(None,),
                                  dtype=opt_grp['feature_type'])
            data = np.concatenate(prob_evals_f, dtype=opt_grp['feature_type'])
            h5_concat_dataset(dset, data)
        
        if prob_evals_c is not None:
            dset = h5_get_dataset(opt_prob, 'constraints', maxshape=(None,),
                                  dtype=opt_grp['constraint_type'])
            data = np.array([tuple(c) for c in prob_evals_c], dtype=opt_grp['constraint_type'])
            h5_concat_dataset(dset, data)
            
        dset = h5_get_dataset(opt_prob, 'predictions', maxshape=(None,),
                              dtype=opt_grp['objective_type'])
        data = np.array([tuple(y) for y in prob_evals_y_pred], dtype=opt_grp['objective_type'])
        h5_concat_dataset(dset, data)
        
    f.close()


def save_surrogate_evals_to_h5(opt_id, problem_id, param_names, objective_names, epoch, gen_index, x_sm, y_sm, fpath, logger):
    """
    Save surrogate evaluations to an HDF5 file 'fpath'.
    """

    f = h5py.File(fpath, "a")
            
    opt_grp = h5_get_group(f, opt_id)

    opt_prob = h5_get_group(opt_grp, str(problem_id))
    opt_sm = h5_get_group(opt_grp, 'surrogate_evals')
        
    n_evals = x_sm.shape[0]
    if logger is not None:
        logger.info(f"Saving {n_evals} surrogate evaluations for problem id {problem_id} to {fpath}.")

    dset = h5_get_dataset(opt_sm, 'epochs', maxshape=(None,),
                          dtype=np.uint32)
    data = np.asarray([epoch]*n_evals, dtype=np.uint32)
    h5_concat_dataset(dset, data)

    dset = h5_get_dataset(opt_sm, 'generations', maxshape=(None,),
                          dtype=np.uint32)
    h5_concat_dataset(dset, gen_index)

    dset = h5_get_dataset(opt_sm, 'objectives', maxshape=(None,),
                          dtype=opt_grp['objective_type'])
    data = np.array([tuple(y) for y in y_sm], dtype=opt_grp['objective_type'])
    h5_concat_dataset(dset, data)

    dset = h5_get_dataset(opt_sm, 'parameters', maxshape=(None,),
                          dtype=opt_grp['parameter_space_type'])
    data = np.array([tuple(x) for x in x_sm], dtype=opt_grp['parameter_space_type'])
    h5_concat_dataset(dset, data)
        
    f.close()

    
def init_h5(opt_id, problem_ids, has_problem_ids, spec, param_names, objective_names, feature_dtypes, constraint_names, problem_parameters, metadata, fpath):
    """
    Save progress and settings to an HDF5 file 'fpath'.
    """

    f = h5py.File(fpath, "a")
    if opt_id not in f.keys():
        h5_init_types(f, opt_id, param_names, objective_names, feature_dtypes, constraint_names, problem_parameters, spec)
        opt_grp = h5_get_group(f, opt_id)
        if has_problem_ids:
            opt_grp['problem_ids'] = np.asarray(list(problem_ids), dtype=np.int32)
        if metadata is not None:
            opt_grp['metadata'] = metadata


    f.close()

    

def eval_obj_fun_sp(obj_fun, pp, space_params, is_int, obj_fun_args, problem_id, space_vals):
    """
    Objective function evaluation (single problem).
    """
    
    this_space_vals = space_vals[problem_id]
    for j, key in enumerate(space_params):
        pp[key] = int(this_space_vals[j]) if is_int[j] else this_space_vals[j]
        
    if obj_fun_args is None:
        obj_fun_args = ()
        
    result = obj_fun(pp, *obj_fun_args)
    return { problem_id: result }


def eval_obj_fun_mp(obj_fun, pp, space_params, is_int, obj_fun_args, problem_ids, space_vals):
    """
    Objective function evaluation (multiple problems).
    """

    mpp = {}
    for problem_id in problem_ids:
        this_pp = copy.deepcopy(pp)
        this_space_vals = space_vals[problem_id]
        for j, key in enumerate(space_params):
            this_pp[key] = int(this_space_vals[j]) if is_int[j] else this_space_vals[j]
        mpp[problem_id] = this_pp

    if obj_fun_args is None:
        obj_fun_args = ()

    result_dict = obj_fun(mpp, *obj_fun_args)
    return result_dict


def sopt_init(sopt_params, worker=None, verbose=False, init_strategy=False):
    objfun = None
    objfun_module = sopt_params.get('obj_fun_module', '__main__')
    objfun_name = sopt_params.get('obj_fun_name', None)
    if distwq.is_worker:
        if objfun_name is not None:
            if objfun_module not in sys.modules:
                importlib.import_module(objfun_module)
                
            objfun = eval(objfun_name, sys.modules[objfun_module].__dict__)
        else:
            objfun_init_module = sopt_params.get('obj_fun_init_module', '__main__')
            objfun_init_name = sopt_params.get('obj_fun_init_name', None)
            objfun_init_args = sopt_params.get('obj_fun_init_args', None)
            if objfun_init_name is None:
                raise RuntimeError("dmosopt.soptinit: objfun is not provided")
            if objfun_init_module not in sys.modules:
                importlib.import_module(objfun_init_module)
            objfun_init = eval(objfun_init_name, sys.modules[objfun_init_module].__dict__)
            objfun = objfun_init(**objfun_init_args, worker=worker)
            
    sopt_params['obj_fun'] = objfun
    reducefun_module = sopt_params.get('reduce_fun_module', '__main__')
    reducefun_name = sopt_params.get('reduce_fun_name', None)
    if reducefun_module not in sys.modules:
        importlib.import_module(reducefun_module)
    if reducefun_name is not None:
        reducefun = eval(reducefun_name, sys.modules[reducefun_module].__dict__)
        sopt_params['reduce_fun'] = reducefun        
    sopt = DistOptimizer(**sopt_params, verbose=verbose)
    if init_strategy:
        sopt.init_strategy()
    sopt_dict[sopt.opt_id] = sopt
    return sopt


def sopt_ctrl(controller, sopt_params, verbose=True):
    """Controller for distributed surrogate optimization."""
    logger = logging.getLogger(sopt_params['opt_id'])
    logger.info(f"Initializing optimization controller...")
    if verbose:
        logger.setLevel(logging.INFO)
    sopt = sopt_init(sopt_params, verbose=verbose, init_strategy=True)
    logger.info(f"Optimizing for {sopt.n_epochs} epochs...")
    start_epoch = sopt.start_epoch
    epoch_count = 0
    eval_count = 0
    saved_eval_count = 0
    task_ids = []
    next_epoch = False
    while epoch_count < sopt.n_epochs:

        epoch = epoch_count + start_epoch
        controller.recv()

        if len(task_ids) > 0:
            rets = controller.probe_all_next_results()
            for ret in rets:

                task_id, res = ret

                if sopt.reduce_fun is None:
                    rres = res
                else:
                    if sopt.reduce_fun_args is None:
                        rres = sopt.reduce_fun(res)
                    else:
                        rres = sopt.reduce_fun(res, *sopt.reduce_fun_args)

                for problem_id in rres:
                    eval_req = sopt.eval_reqs[problem_id][task_id]
                    eval_x = eval_req.parameters
                    eval_pred = eval_req.prediction
                    if sopt.feature_names is not None and sopt.constraint_names is not None:
                        entry = sopt.optimizer_dict[problem_id].complete_request(eval_x, rres[problem_id][0],
                                                                                 f=rres[problem_id][1], c=rres[problem_id][2],
                                                                                 pred=eval_pred, epoch=epoch)
                        sopt.storage_dict[problem_id].append(entry)
                    elif sopt.feature_names is not None:
                        entry = sopt.optimizer_dict[problem_id].complete_request(eval_x, rres[problem_id][0], f=rres[problem_id][1],
                                                                                 pred=eval_pred, epoch=epoch)
                        sopt.storage_dict[problem_id].append(entry)
                    elif sopt.constraint_names is not None:
                        entry = sopt.optimizer_dict[problem_id].complete_request(eval_x, rres[problem_id][0], c=rres[problem_id][1],
                                                                                 pred=eval_pred, epoch=epoch)
                        sopt.storage_dict[problem_id].append(entry)
                    else:
                        entry = sopt.optimizer_dict[problem_id].complete_request(eval_x, rres[problem_id], pred=eval_pred, epoch=epoch)
                        sopt.storage_dict[problem_id].append(entry)
                    prms = list(zip(sopt.param_names, list(eval_x.T)))
                    lftrs = None
                    lres = None
                    if sopt.feature_names is not None and sopt.constraint_names is not None:
                        lres = list(zip(sopt.objective_names, rres[problem_id][0].T))
                        lftrs = [dict(zip(rres[problem_id][1].dtype.names, x)) for x in rres[problem_id][1]]
                        lconstr = list(zip(sopt.constraint_names, rres[problem_id][2].T))
                        logger.info(f"problem id {problem_id}: optimization epoch {epoch_count}: parameters {prms}: {lres} / {lftrs} constr: {lconstr}")
                    elif sopt.feature_names is not None:
                        lres = list(zip(sopt.objective_names, rres[problem_id][0].T))
                        lftrs = [dict(zip(rres[problem_id][1].dtype.names, x)) for x in rres[problem_id][1]]
                        logger.info(f"problem id {problem_id}: optimization epoch {epoch_count}: parameters {prms}: {lres} / {lftrs}")
                    elif sopt.constraint_names is not None:
                        lres = list(zip(sopt.objective_names, rres[problem_id][0].T))
                        lconstr = list(zip(sopt.constraint_names, rres[problem_id][1].T))
                        logger.info(f"problem id {problem_id}: optimization epoch {epoch_count}: parameters {prms}: {lres} / constr: {lconstr}")
                    else:
                        lres = list(zip(sopt.objective_names, rres[problem_id].T))
                        logger.info(f"problem id {problem_id}: optimization epoch {epoch_count}: parameters {prms}: {lres}")
                
                eval_count += 1
                task_ids.remove(task_id)

        if sopt.save and (eval_count > 0) and (saved_eval_count < eval_count) and ((eval_count - saved_eval_count) >= sopt.save_eval):
            sopt.save_evals()
            saved_eval_count = eval_count

        if not next_epoch:
            n_workers = 1
            if controller.workers_available:
                n_workers = len(controller.ready_workers)
            for w in range(n_workers):
                eval_req_dict = {}
                eval_x_dict = {}
                for problem_id in sopt.problem_ids:
                    eval_req = sopt.optimizer_dict[problem_id].get_next_request()
                    if eval_req is None:
                        next_epoch = True
                    else:
                        eval_req_dict[problem_id] = eval_req
                        eval_x_dict[problem_id] = eval_req.parameters
                if next_epoch:
                    break

                task_id = controller.submit_call("eval_fun", module_name="dmosopt.dmosopt",
                                                 args=(sopt.opt_id, eval_x_dict,))
                task_ids.append(task_id)
                for problem_id in sopt.problem_ids:
                    sopt.eval_reqs[problem_id][task_id] = eval_req_dict[problem_id]

        if next_epoch and (len(task_ids) == 0):
            if sopt.save and (eval_count > 0) and (saved_eval_count < eval_count):
                sopt.save_evals()
                saved_eval_count = eval_count
                    
            for problem_id in sopt.problem_ids:
                if epoch_count > 0:
                    completed_epoch_evals = list(filter(lambda x: x.epoch == epoch_count, sopt.optimizer_dict[problem_id].completed))
                    if len(completed_epoch_evals) > 0:
                        y_completed = np.vstack([x.objectives for x in completed_epoch_evals])
                        pred_completed = np.vstack([x.prediction for x in completed_epoch_evals])
                        n_objectives = y_completed.shape[1]
                        mae = []
                        for i in range(n_objectives):
                            mae.append(np.mean(np.abs(y_completed[:,i] - pred_completed[:,i])))
                        logger.info(f"surrogate accuracy at step {epoch_count} for problem {problem_id} was {mae}")
                logger.info(f"performing optimization step {epoch_count+1} for problem {problem_id} ...")
                x_sm, y_sm = None, None
                if sopt.save and sopt.save_surrogate_eval:
                    _, _, gen_index, x_sm, y_sm = sopt.optimizer_dict[problem_id].step(return_sm=True)
                    sopt.save_surrogate_evals(problem_id, epoch+1, gen_index, x_sm, y_sm)
                else:
                    sopt.optimizer_dict[problem_id].step()
                logger.info(f"completed optimization step {epoch_count+1} for problem {problem_id} ...")
            controller.info()
            sys.stdout.flush()
            next_epoch = False
            if eval_count > 0:
                epoch_count += 1

    if sopt.save:
        sopt.save_evals()
    controller.info()

def sopt_work(worker, sopt_params, verbose=False, debug=False):
    """Worker for distributed surrogate optimization."""
    if worker.worker_id > 1 and (not debug):
        verbose = False
    sopt_init(sopt_params, worker=worker, verbose=verbose, init_strategy=False)

def eval_fun(opt_id, *args):
    return sopt_dict[opt_id].eval_fun(*args)

def run(sopt_params, feasible=True, return_features=False, return_constraints=False, spawn_workers=False, sequential_spawn=False, spawn_startup_wait=None, nprocs_per_worker=1, collective_mode="gather", verbose=True, worker_debug=False):
    if distwq.is_controller:
        distwq.run(fun_name="sopt_ctrl", module_name="dmosopt.dmosopt",
                   verbose=verbose, args=(sopt_params, verbose,),
                   spawn_workers=spawn_workers,
                   sequential_spawn=sequential_spawn,
                   spawn_startup_wait=spawn_startup_wait,
                   nprocs_per_worker=nprocs_per_worker,
                   collective_mode=collective_mode)
        opt_id = sopt_params['opt_id']
        sopt = sopt_dict[opt_id]
        sopt.print_best()
        return sopt.get_best(feasible=feasible, return_features=return_features, return_constraints=return_constraints)
    else:
        if 'file_path' in sopt_params:
            del(sopt_params['file_path'])
        if 'save' in sopt_params:
            del(sopt_params['save'])
        distwq.run(fun_name="sopt_work", module_name="dmosopt.dmosopt",
                   broker_fun_name=sopt_params.get("broker_fun_name", None),
                   broker_module_name=sopt_params.get("broker_module_name", None),
                   verbose=verbose, args=(sopt_params, verbose, worker_debug, ),
                   spawn_workers=spawn_workers,
                   sequential_spawn=sequential_spawn,
                   spawn_startup_wait=spawn_startup_wait,
                   nprocs_per_worker=nprocs_per_worker,
                   collective_mode=collective_mode)
        return None
        


