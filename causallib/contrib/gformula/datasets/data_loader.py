#!/usr/bin/env python3

import os
import pandas as pd
from causallib.datasets.data_loder import load_data_file

DATA_DIR_NAME = os.path.dirname(__file__)

def load_synth_dynamic_treat(observational=True):
    """
    Loads and pre-processes the synthetic dynamic treatment dataset.

    See https://github.com/pchakraborty1/causallib/blob/test_cases_issue_1/causallib/datasets/data/synthetic_dynamic_treatment/README.md
    for details on how data was generated

    Args:
        observational (bool): which strategy to load
                    If True, loads the train, validation, and test folds for the observational strategy
                    If False, loads the never treat strategy data. This is only test

    Returns:
        Dictionary with folds as keys. each fold contains
        X (pd.DataFrame): covariate matrix of size ((num_subjects x num_timepoints), num_features).
        a (pd.Series): Treatment assignment of size ((num_subjects x num_timepoints),). treatment dose
    """
    dir_name = os.path.join(DATA_DIR_NAME, "synthetic_dynamic_treatment")

    payload = dict()
    expt_name = 'simx2'
    if observational:
        sections = ['train', 'val', 'test']
        for s in sections:
            fName = f"{expt_name}_observational_{s}.csv"
            data = load_data_file(fName, dir_name)
            X = data[['id', 'time', 'cov1', 'cov2']].copy()
            a = data[['id', 'time', 'treatment']].copy()
            payload[s] = (X, a)
    else:
        fName = f"{expt_name}_never-treat.csv"
        data = load_data_file(fName, dir_name)
        X = data[['id', 'time', 'cov1', 'cov2']].copy()
        a = data[['id', 'time', 'treatment']].copy()
        payload['test'] = (X, a)
    return payload


