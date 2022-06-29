#!/usr/bin/env python3

from typing import Optional, Any, OrderedDict
import pandas as pd
from causallib.time_varying.GFormulaBase import TimeVaryingBaseEstimator
from ..estimation.base_estimator import IndividualOutcomeEstimator


#TODO
class CovariateModels(OrderedDict):
    covariate: str
    model: IndividualOutcomeEstimator


class GFormulaBase(TimeVaryingBaseEstimator):
    """
        GFormula base Estimator
    """
    def __init__(self,
                 outcome_model: Any,
                 treatment_model: Any,
                 covariate_models: CovariateModels,
                 refit_models=True):
        """
            outcome_model(IndividualOutcomeEstimator): A causal model that estimate on individuals level
                                                      (e.g. Standardization).
            treatment_model (???):
            covariate_models (dict): {
                        ‘x1’: Standardization(LogisticRegression()),
                        ‘x2’: Standardization(LinearRegression()),
                            so forth
                        },
            refit_models (bool): if True, re-fit a the treatment model and covariate models.

        """
        super(GFormulaBase, self).__init__(lambda **x: None)

        self.outcome_model = outcome_model
        self.treatment_model = treatment_model
        self.covariate_models = covariate_models
        self.refit_models = refit_models

    def fit(self,
            X: pd.DataFrame,
            a: pd.Series,
            t: pd.Series,
            y: Optional[Any] = None,
            refit_models=True,
            **kwargs
            ):
        """
            Fits parametric models and calculates prediction curves.

            Args:
                X (pd.DataFrame): Baseline covariate matrix of size (num_subjects, num_features).
                a (pd.Series): Treatment assignment of size (num_subjects,).
                t (pd.Series): Followup duration, size (num_subjects,).
                y (pd.Series): Observed outcome (1) or right censoring event (0), size (num_subjects,).
                refit_models (bool): if True, re-fit a the treatment model and covariate models.
                kwargs (dict): Optional kwargs for fit call of survival model

            Returns:
                self
        """
        raise NotImplementedError

    def estimate_individual_outcome(self,
                                    X: pd.DataFrame,
                                    a: pd.Series,
                                    t: pd.Series,
                                    y: Optional[Any] = None,
                                    timeline_start: Optional[int] = None,
                                    timeline_end: Optional[int] = None
                                    ) -> pd.DataFrame:
        """
              Returns individual estimated curves for each subject row in X/a/t

              Args:
                  X (pd.DataFrame): Baseline covariate matrix of size (num_subjects, num_features).
                  a (pd.Series): Treatment assignment of size (num_subjects,).
                  t (pd.Series): Followup durations, size (num_subjects,).
                  y: NOT USED (for API compatibility only).
                  timeline_start (int): Common start time-step. If provided, will generate survival curves starting
                                        from 'timeline_start' for all patients.
                                        If None, will predict from first observed event (t.min()).
                  timeline_end (int): Common end time-step. If provided, will generate survival curves up to 'timeline_end'
                                      for all patients.
                                      If None, will predict up to last observed event (t.max()).

              Returns:
                  pd.DataFrame: with time-step index, subject IDs (X.index) as columns and ??
              """
        pass

    def estimate_population_outcome(self,
                                    X: pd.DataFrame,
                                    a: pd.Series,
                                    t: pd.Series,
                                    y: Optional[Any] = None,
                                    timeline_start: Optional[int] = None,
                                    timeline_end: Optional[int] = None
                                    ) -> pd.DataFrame:
        """
              Returns individual estimated curves for each subject row in X/a/t

              Args:
                  X (pd.DataFrame): Baseline covariate matrix of size (num_subjects, num_features).
                  a (pd.Series): Treatment assignment of size (num_subjects,).
                  t (pd.Series): Followup durations, size (num_subjects,).
                  y: NOT USED (for API compatibility only).
                  timeline_start (int): Common start time-step. If provided, will generate survival curves starting
                                        from 'timeline_start' for all patients.
                                        If None, will predict from first observed event (t.min()).
                  timeline_end (int): Common end time-step. If provided, will generate survival curves up to 'timeline_end'
                                      for all patients.
                                      If None, will predict up to last observed event (t.max()).

              Returns:
                  pd.DataFrame: with time-step index, subject IDs (X.index) as columns and ??
              """
        pass


class GFormula(GFormulaBase):
    """
        GFormula class that is based on Monte Carlo Simulation for creating noise. #TODO more on this
    """
    def __init__(self,
                 outcome_model,
                 treatment_model,
                 covariate_models,
                 refit_models=True
                 ):

        super().__init__(outcome_model, treatment_model, covariate_models, refit_models)

    def fit(self,
            X: pd.DataFrame,
            a: pd.Series,
            t: pd.Series,
            y: Optional[Any] = None,
            refit_models=True,
            **kwargs
            ):
        if refit_models:
            self.treatment_model.fit(X, a)
            for cov in self.covariate_models:
                self.covariate_models[cov].fit(X, a)

        self.outcome_model.fit(X, a, t, y)
        return self

    def predict(self, X, a, t):
        pass


