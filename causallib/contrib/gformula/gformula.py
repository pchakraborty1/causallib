#!/usr/bin/env python3

import warnings

import pandas as pd
from typing import Optional, Any, Union, OrderedDict, Callable
import numpy as np
from joblib import Parallel, delayed

from causallib.utils import general_tools as g_tools

from . import constants as C
from .base import GMethodBase
from .treament_strategy import TreatmentStrategy, Observational, CFBernoulli


def calculate_residual(model, X, y, index):
    yhat = model.predict(X)
    resid = y - np.reshape(yhat, y.shape)
    resid_df = pd.concat((resid, index), 
                         axis=1).set_index(list(index.columns)) 
    return resid_df


class GFormula(GMethodBase):
    """
        GFormula class that is based on Monte Carlo Simulation.
    """
    def __init__(self, 
                 treatment_models: OrderedDict, 
                 covariate_models: OrderedDict, 
                 outcome_model: Optional[Any]=None, 
                 refit_models: Optional[bool]=True, 
                 mode: Optional[str]=C.DEFAULT_SIM_MODE, 
                 random_state: Optional[int]=None, 
                 ):
        super(GFormula, self).__init__(treatment_models, covariate_models, outcome_model, refit_models)
        self.mode = mode
        self.random_state = random_state
        #  self.n_obsv = n_obsv
        #  self.n_sims = n_sims
        #  self.n_steps = n_steps
        #  self.covariate_cols = None
        #  self.treatment_cols = None
        #  self.time_col = None
        #  self.y_col = None
        #  self.prev_covariate_cols = None
        #  self.prev_treatment_cols = None
        #  self.all_cols = None
    
    @property
    def id_col(self):
        try:
            return self.id_col_
        except AttributeError:
            self.id_col = C.DEFAULT_ID_NAME
            return self.id_col

    @id_col.setter
    def id_col(self, value):
        self.id_col_ = value
        return

    @property
    def time_col(self):
        try:
            return self.time_col_
        except AttributeError:
            self.time_col = C.DEFAULT_TIME_NAME
            return self.time_col

    @time_col.setter
    def time_col(self, value):
        self.time_col_ = value
        return

    def set_resid_val(self, var_name, value):
        if not hasattr(self, 'resid_val_'):
            self.resid_val_ = dict()

        if type(value) is np.ndarray and value.ndim == 3:
            warnings.warn('numpy type detected. saving supplied residuals in passed format')
            residuals = value
        elif isinstance(value, pd.DataFrame):
            # pandas dataframe passed. processing to numpy array
            assert list(value.index.names) == [self.id_col, self.time_col], f"Dataframe indices should be {[self.id_col, self.time_col]}"
            _n_id = len(value.index.levels[0])
            _n_time = len(value.index.levels[1])

            residuals = value.values.reshape(_n_id, _n_time, -1).swapaxes(0, 1)  # N_TIMEPOINTS x N_IDS X 
            self.resid_val_[var_name] = residuals
        else:
            raise AttributeError('Unsupported data passed. can either be a 3D array or a dataframe')
        return

    def fit(self,
            X: pd.DataFrame,
            a: pd.Series,
            t: Optional[pd.Series] = None,
            y: Optional[Any] = None,
            id_col: str = None,
            **kwargs
            ):

        X_val = kwargs.pop('X_val', None)
        a_val = kwargs.pop('a_val', None)
        t_val = kwargs.pop('t_val', None)
        y_val = kwargs.pop('y_val', None)
       
        # Book-keeping the covariate names
        if id_col:
            self.id_col = id_col
        self._set_cols(X, a, t, y)

        # Reading the datasets
        (index, treatment_X, treatment_Y, 
         covariate_X, covariate_Y, 
         outcome_X, outcome_Y) = self._prepare_data(X, a, t, y)

        is_residual_calculatable = (X_val is not None) and (a_val is not None)
        if is_residual_calculatable:
            (val_index, val_treatment_X, val_treatment_Y, 
             val_covariate_X, val_covariate_Y, 
             val_outcome_X, val_outcome_Y) = self._prepare_data(X_val, a_val, t_val, y_val)
        else:
            warnings.warn('Residuals cannot be computed. validation data not provided. depending on the simulation mode may need to manually supply residuals')

        for treat in self.treatment_models:
            treatment_model_is_not_fitted = not g_tools.check_learner_is_fitted(self.treatment_models[treat])
            if self.refit_models or treatment_model_is_not_fitted:
                # FIXME: support for multiple treatments
                self.treatment_models[treat].fit(treatment_X, treatment_Y, **kwargs)
            if is_residual_calculatable and self.d_type_dict_[treat] is not bool:
                self.set_resid_val(treat,
                                   calculate_residual(self.treatment_models[treat],
                                                      val_treatment_X, val_treatment_Y, val_index)
                                  )

        for cov in self.covariate_models:
            cov_model_is_not_fitted = not g_tools.check_learner_is_fitted(self.covariate_models[cov])
            if self.refit_models or cov_model_is_not_fitted:
                self.covariate_models[cov].fit(covariate_X[cov], covariate_Y[cov],  **kwargs)
            if is_residual_calculatable and self.d_type_dict_[cov] is not bool:
                self.set_resid_val(cov, 
                                   calculate_residual(self.covariate_models[cov],
                                                      val_covariate_X[cov], val_covariate_Y[cov], val_index)
                                  )

        if y:
            outcome_model_is_not_fitted = not g_tools.check_learner_is_fitted(self.outcome_model)
            if self.refit_models or outcome_model_is_not_fitted:
                self.outcome_model.fit(outcome_X, outcome_Y, **kwargs)

        # TODO calculate residual_values
        return self

    def estimate_individual_outcome(self,
                                    X: pd.DataFrame,
                                    a: pd.Series,
                                    t: Optional[pd.Series] = None,
                                    y: Optional[Any] = None,
                                    treatment_strategy: Optional[Union[Callable, str]] = C.DEFAULT_OBSV_STRATEGY,
                                    timeline_start: Optional[int] = None,
                                    timeline_end: Optional[int] = None,
                                    n_jobs: Optional[int] = C.DEFAULT_N_JOBS,
                                    n_sims: Optional[int] = C.DEFAULT_N_SIMS
                                    ) -> pd.DataFrame:
        """
            Returns dataframe (individual estimated curves) for each subject in (t * (X and a))

        Steps:
            1. For each sample,
                a. Get the simulation outputs ('covariates', 'actions' & 'time')
                   from _estimate_individual_outcome_single_sample
                b. For covariate (n_sim * n_steps * cov_cols) & action (n_sim * n_steps * act_cols),
                   take mean across 'n_sim', which will result (n_steps * cov_cols) & (n_steps * act_cols)
                c. Concatenate these two arrays across column
                d. Convert the result to df
                e. Add 'sample_id' and 'time' to df
                f. Append each sample to list
            2. Finally, concatenate list of dfs and return
        """

        if self.random_state is not None:
            np.random.seed(self.random_state)
        unique_sample_ids = X[self.id_col].unique()
        if not callable(treatment_strategy):
            if treatment_strategy == C.DEFAULT_OBSV_STRATEGY:
                treatment_strategy = Observational(treatment_models=self.treatment_models)
            elif treatment_strategy == C.DEFAULT_CF_STRATEGY:
                treatment_strategy = CFBernoulli(p=0.)
            else:
                raise AttributeError('unknown strategy passed. neither a callable nor of pre-defined strategy') 

        def _run_single_id(sample_id):
            sample_X = X.loc[X[self.id_col] == sample_id]
            sample_a = a.loc[a[self.id_col] == sample_id]
            sample_y = y.loc[y[self.id_col] == sample_id] if y else None
            sample_t = t.loc[t[self.id_col] == sample_id] if t else None 
            sample_sim = self._estimate_individual_outcome_single_sample(X=sample_X,
                                                                         a=sample_a,
                                                                         t=sample_t,
                                                                         y=sample_y,
                                                                         treatment_strategy=treatment_strategy,
                                                                         timeline_start=timeline_start,
                                                                         timeline_end=timeline_end,
                                                                         n_sims=n_sims
                                                                        )

            sample_sim_cov = sample_sim['covariates'].mean(axis=0)  # n_steps * cov_cols
            sample_sim_act = sample_sim['actions'].mean(axis=0)  # n_steps * act_cols

            sample_sim_res = pd.DataFrame(np.concatenate([sample_sim_cov, sample_sim_act], axis=1),
                                          columns=list(self.covariate_cols_) + list(self.treatment_cols_))
            sample_sim_res[self.id_col] = sample_id
            sample_sim_res[self.time_col] = sample_sim['time']
            return sample_sim_res
       
        if n_jobs is None:
            warnings.warn('Not using parallel backend')
            all_sim_result = [_run_single_id(sample_id) for sample_id in unique_sample_ids]
        else:
            all_sim_result = Parallel(n_jobs=n_jobs, verbose=10)(delayed(_run_single_id)(sample_id) 
                                                                for sample_id in unique_sample_ids)


        return pd.concat(all_sim_result)

    def _estimate_individual_outcome_single_sample(self, X, a, t, y, treatment_strategy, timeline_start, timeline_end, n_sims) -> dict:
        """
            Simulates the outcome for each sample across 't' steps.
            Returns:
                 sample_sim = [
                {
                    'actions' = n_sim * n_steps * cov_cols
                    'covariates' = n_sim * n_steps * act_cols
                    'time' = 1 * n_steps
                    'pat_id' = str
                }
            ]
        """
        # TODO remove redundancy : n_obsv and n_steps
        if t is None:
            t = X[self.time_col]
        min_time = timeline_start if timeline_start is not None else int(t.min() + 1)  # need to observe at least one variable
        max_time = timeline_end if timeline_end is not None else int(t.max())
        contiguous_times = pd.Series(data=range(min_time, max_time), name=t.name)

        n_steps = len(contiguous_times)
        n_obsv = min_time 

        simulation = dict(actions=list(),
                          covariates=list(),
                          time=list(),
                          pat_id=X.iloc[0][self.id_col],
                          )
        X = X.drop([self.id_col, self.time_col], axis=1)
        a = a.drop([self.id_col, self.time_col], axis=1)
        y = y.drop([self.id_col, self.time_col], axis=1) if y else None

        X = np.expand_dims(X.to_numpy(), axis=0)
        a = np.expand_dims(a.to_numpy(), axis=0)
        y = np.expand_dims(y.to_numpy(), axis=0) if y else None
        lengths = np.array([len(X), ])

        # assert y.shape[1] >= n_obsv, "n_obsv bigger than observed data"

        X = X.repeat(n_sims, 0)                 # N_sims * T * F
        a = a.repeat(n_sims, 0)                 # N_sims * T * F
        y = y.repeat(n_sims, 0) if y else None  # N_sims * T * F

        x_t = X[:, :n_obsv, :]
        a_t = a[:, :n_obsv, :]
        y_t = y[:, :n_obsv, :] if y else None

        act_t = treatment_strategy(prev_x=x_t[:, -1, :], 
                                   all_x=x_t[:, :-1, :], 
                                   prev_a=a_t[:, -1, :])
        a_t = np.concatenate((a_t[:, :-1, :], act_t), axis=1)

        # init all the models
        self._init_models()  # TODO

        # Simulate
        for _idx in range(n_steps):
            _idx = _idx + n_obsv
            sim_t = self._predict(x_t, a_t, y_t, _idx)  # TODO add support for RNN later

            # update x_t and a_t
            x_t = np.concatenate([x_t, sim_t], axis=1)

            # get a new treatment_action
            act_t = treatment_strategy(prev_x=x_t[:, -1, :], all_x=x_t[:, :-1, :], prev_a=a_t[:, -1, :])
            # a_t = np.concatenate((a_t[:, :-1, :], act_t), axis=1)
            a_t = np.concatenate([a_t, act_t], axis=1)
            
            simulation['actions'].append(act_t)
            simulation['covariates'].append(sim_t)
            simulation['time'].append(_idx)

        simulation['actions'] = np.concatenate(simulation['actions'], axis=1)  # N_sim * n_steps * F-act
        simulation['covariates'] = np.concatenate(simulation['covariates'], axis=1)  # N_sim * n_steps * F-act
        return simulation

    def estimate_population_outcome(self,
                                    X: pd.DataFrame,
                                    a: pd.Series,
                                    t: pd.Series = None,
                                    y: Optional[Any] = None,
                                    treatment_strategy: Optional[Union[Callable, str]] = C.DEFAULT_OBSV_STRATEGY,
                                    timeline_start: Optional[int] = None,
                                    timeline_end: Optional[int] = None,
                                    n_jobs: Optional[int] = C.DEFAULT_N_JOBS,
                                    n_sims: Optional[int] = C.DEFAULT_N_SIMS
                                    ) -> pd.DataFrame:

        """
            Calculates population outcome for each covariate across 't' times.
            Backlog: Support for multiple treatment strategies. Like a list of  "always_treated"  and "never_treated".
        """
        individual_prediction_curves = self.estimate_individual_outcome(X=X,
                                                                        a=a,
                                                                        t=t,
                                                                        y=y,
                                                                        treatment_strategy=treatment_strategy,
                                                                        timeline_start=timeline_start,
                                                                        timeline_end=timeline_end)

        # Averaging across time
        res = individual_prediction_curves.groupby(individual_prediction_curves.time).mean()
        return res

    def _apply_noise(self, out, t, var_name):

        # TODO: Convert Torch to np
        # TODO: infer box_type from data
        mode = self.mode
        box_type = self.d_type_dict_[var_name]

        is_residual_computed = True
        try:
            residuals = self.resid_val_[var_name]
        except Exception as res_e:
            is_residual_computed = False
        """adding noise to a box output
           out: bs * 1 * <F-box>
           t: time
           box: box id
           mode: mode of operation
        """
        #  _device = out.device
        # (first_box_var, last_box_var, _, _) = self.model.box_dim[box]
        #  box_type = self.model.box_type[box]

        if box_type == 'boolean':
            sim = (np.random.rand(*out.shape) < out).astype('int')
            #  sim = T.from_numpy(_sim).float().to(out.device)  # , requires_grad=False).to(device)
            #  sim.requires_grad_(False)

        elif box_type == 'float':
            if mode == 'empirical':
                if not is_residual_computed:
                    raise Exception(f'residuals are not computed. see full error={res_e}')
                _resid_dist = residuals[t, :, :]  # bs * F
                _sim_noise_t = self._batch_choice(_resid_dist, num_samples=out.shape[0])
                #  _sim_noise_t = T.from_numpy(sim_noise).float()  # , requires_grad=False)  # bs * <F-box>
                #  _sim_noise_t.requires_grad_(False)

                #  _sim_noise_t = np.expand_dims(sim_noise, axis=1)  # bs * 1 * <F-box>
                # clamping values between 0 and 1
                # sim.clamp_(0, 1)
            elif mode == 'normal':
                _sim_noise_t = 1.0 * np.random.randn(*out.shape)
            elif mode == 'emp_std':
                if not is_residual_computed:
                    raise Exception(f'residuals are not computed. see full error={res_e}')
                _std = residuals[t, :, :].std(axis=0)
                # TODO: chech the shape of the generated random variable
                _sim_noise_t = _std * np.random.randn(*out.shape)
            elif mode == 'emp_mean_std':
                if not is_residual_computed:
                    raise Exception(f'residuals are not computed. see full error={res_e}')
                _std = residuals[t, :, :].std(axis=0)
                _mean = residuals[t, :, :].mean(axis=0)
                # TODO: chech the shape of the generated random variable
                _sim_noise_t = _mean + _std * np.random.randn(*out.shape)
            elif mode == 'tdist':
                #  _dist =  T.distributions.StudentT(df=residuals.shape[1] - 1)
                raise NotImplementedError()
                # TODO: chech the shape of the generated t-dist.
                # https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_t.html
                _sim_noise_t = 1.0 * np.random.standard_t(df=residuals.shape[1] -1, size=out.shape)  # _dist.sample(out.shape)
            else:
                raise AttributeError()
            #  _sim_noise_t = _sim_noise_t.to(_device)
            sim = out + _sim_noise_t

        else:
            raise AttributeError()
        return sim

    def _batch_choice(self, arr, num_samples):
        val_arr = arr[~np.isnan(arr).any(axis=1)]
        idx = val_arr.shape[0]
        choice = np.random.choice(idx, num_samples)
        sample = val_arr[choice, :]
        return sample

    def _prepare_data(self, X, a, t, y):

        #  data = X.join(a)
        data = X.merge(a, on=[self.id_col, self.time_col])
        for cov in self.covariate_cols_:
            data['prev_' + cov] = data.groupby(self.id_col)[cov].shift(1)
        for a in self.treatment_cols_:
            data['prev_' + a] = data.groupby(self.id_col)[a].shift(1)

        data.dropna(inplace=True)  # dropping first row
        for var in self.covariate_cols_ + self.treatment_cols_:
            data['prev_' + var] = data['prev_' + var].astype(data[var].dtype)

        index = data[[self.id_col, self.time_col]]
        data = data[self.all_cols_]

        self.d_type_dict_ = dict(data.dtypes)

        X_treatment, Y_treatment = self.extract_treatment_model_data(data)
        X_covariates, Y_covariates = self.extract_covariate_models_data(data)
        X_outcome, Y_outcome = self.extract_outcome_model_data(data)

        return index, X_treatment, Y_treatment, X_covariates, Y_covariates, X_outcome, Y_outcome

    def _init_models(self, *args, **kwargs):
        return

        # TODO Write actual code for model initialization
        # TODO Initially do it for only sklearn
        # for model in self.covariate_models:
        #     model.init()
        # self.treatment_model.init()
        # self.outcome_model.init()

        # self.treatment_model = sklearn.base.clone(self.treatment_model)
        # todo add args

    def _predict(self, X, a, y, step):

        _input = np.concatenate((X, a), axis=2)
        # roll/shift right to number of 'col_dim'
        _input = np.concatenate([_input, np.roll(_input, -_input.shape[2])], axis=2)
        #FIXME: make this a call to support longitudinal data
        _input = pd.DataFrame(_input[:, -1, :], columns=self.all_cols_)

        X_sim = []
        d_type_dict = self.d_type_dict_ # dict(_input.dtypes)
        #  assert d_type_dict == self.d_type_dict_
        default_cols = self.prev_covariate_cols_ + self.prev_treatment_cols_
        _input = _input[default_cols]
        for i, cov in enumerate(self.covariate_models):
            if d_type_dict[cov] == 'float':
                _pred = self.covariate_models[cov].predict(_input)
            elif d_type_dict[cov] == 'bool':
                _pred = self.covariate_models[cov].predict_proba(_input)
            else:
                raise ValueError("Data type error. {0}, is not supported for {1}".format(d_type_dict[cov], cov))

            _pred = np.expand_dims(_pred, axis=1)
            _pred = self._apply_noise(_pred, step, cov)  # bs * 1 
            #FIXME: make this a call to support longitudinal data
            _input = pd.DataFrame(np.concatenate((_input, _pred), axis=1), 
                                  columns=self.all_cols_[:len(default_cols)+i+1])
            X_sim.append(_pred)

        X_sim = np.concatenate(X_sim, axis=1)
        X_sim = np.expand_dims(X_sim, axis=1)
        return X_sim

    def extract_treatment_model_data(self, X):
        all_columns = self.all_cols_
        treatment_cols = self.treatment_cols_
        #FIXME: support for multiple treatments here
        _columns = [col for col in all_columns if col not in treatment_cols]
        X_treatment = X[_columns]
        Y_treatment = X[treatment_cols]
        return X_treatment, Y_treatment

    def extract_covariate_models_data(self, X):
        all_columns = self.all_cols_
        default_cols = self.prev_covariate_cols_ + self.prev_treatment_cols_

        X_covariates = {}
        Y_covariates = {}
        for i, cov in enumerate(self.covariate_models):
            _columns = all_columns[: len(default_cols) + i]
            X_covariates[cov] = X[_columns]
            Y_covariates[cov] = X[cov]
        return X_covariates, Y_covariates

    def extract_outcome_model_data(self, X):
        if self.y_col_: 
            all_columns = self.all_cols_ 
            y_cols = self.y_col_

            _columns = [col for col in all_columns if col not in y_cols]
            X_outcome = X[_columns]
            Y_outcome = X[y_cols]
        else:
            X_outcome = None
            Y_outcome = None
        return X_outcome, Y_outcome

    def _set_cols(self, X, a, t=None, y=None):
        self.covariate_cols_ = list(self.covariate_models.keys())
        self.treatment_cols_ = list(self.treatment_models.keys())
        self.y_col_ = y.name if y else None

        self.prev_covariate_cols_ = ['prev_' + cov for cov in self.covariate_cols_]
        self.prev_treatment_cols_ = ['prev_' + a for a in self.treatment_cols_]
        self.all_cols_ = self.prev_covariate_cols_ + self.prev_treatment_cols_ + self.covariate_cols_ + self.treatment_cols_

        if t:
            self.time_col = t.name
        return





