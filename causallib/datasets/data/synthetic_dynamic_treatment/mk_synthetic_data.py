#!/usr/bin/env python
_desc_doc = """
Script to create synthetic dataset with dynamic treatment effect

Data contains 5 attributes: id, time, treatment (A), cov1 (X1), cov2 (X2)

::latex::
    A \sim Binomial(\text{invLogit}(X1_{t-1} - \bar{x1})/10 - A_{t-1})
    X1 \sim Normal(A_t + X1_{t-1}, 1)
    X2 \sim Normal(0, 1)

A shifted dataset is also generated with previous values of past covariates.  
Attributes:  
    - independent: id, time, prev_treatment, prev_cov1, prev_cov2
    - dependent: treatment (boolean), prev_cov1 (float), prev_cov2 (float)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from toolz import functoolz
from sklearn.model_selection import train_test_split

import sys
import argparse
import logging
log = logging.getLogger()

# CONSTANTS
EPS = np.finfo(float).eps
EXPT_NAME = 'simx2'
NUM_TIMEPOINTS = 50
NUM_SAMPLES = 1000
SEED = 20
OBSV_STRATEGY = 'observational'
CF_STRATEGY = 'never-treat'

IDX_COLS = ['id', 'time']
COV_COLS = ['cov1', 'cov2']
TREAT_COLS = ['treatment']
SHIFTED_FEAT_COLS = ['prev_treatment', 'prev_cov1', 'prev_cov2']
SHIFTED_TGT_COLS = ['treatment', 'cov1', 'cov2']
TIME_ORDER_COL = ['time_history']

invLogit = lambda x: (np.exp(x) / (1 + np.exp(x)))      # same as sigmoid function (-\inf, +\inf) -> (0, 1) 

# SAMPLES
def generate_sample(rep, strategy=OBSV_STRATEGY):
    A = np.empty(rep)
    X = np.empty(rep)
    X2 = np.empty(rep)

    A[0] = np.random.binomial(1, 0.5)
    X[0] = np.random.normal(A[0])
    X2[0] = np.random.normal()
    
    min_x = X[1]
    mean_x = X[1]
    mean_A = A[1]

    for i in range(1, rep):
        if strategy == OBSV_STRATEGY:
            A[i] = np.random.binomial(1, invLogit((X[i-1] - mean_x) / 10 - A[i-1]))
        elif strategy == CF_STRATEGY:
            # never treat
            A[i] = 0
        else:
            raise NotImplementedError()

        X[i] = np.random.normal(A[i] + X[i-1])
        X2[i] = np.random.normal()

        mean_A = (mean_A * (i - 1) + A[i]) / i
        mean_x = (mean_x * (i - 1) + X[i]) / i
        min_x = min(min_x, X[i])
    time = np.arange(rep)
    sample = np.transpose(np.vstack((time, A, X, X2)))
    return sample

@functoolz.curry
def create_shifted_targets(df, shift_index,
                           shift_suffix='_history'
                           #shift_columns=['treatment', 'cov1']
                           ):
    tmp = pd.merge(df[[shift_index]], df,how='cross', suffixes=('', shift_suffix))
    tmp = tmp[tmp[f"{shift_index}"] >= tmp[f"{shift_index}{shift_suffix}"]]  
    tmp.set_index([f"{shift_index}", f"{shift_index}{shift_suffix}"], inplace=True)
    return tmp

def create_train_validation_splits(fIn, 
                                   shifted=False
                                   #  idx_cols=['id', 'time'],
                                   #  feat_cols=['prev_treat', 'prev_cov1', 'prev_cov2'],
                                   #  tgt_cols=['treatment', 'cov1', 'cov2'],
                                   #  time_order_col=['time_history']
                                  ):
    if shifted:
        _create_train_validation_splits_shifted(fIn,
                                                idx_cols=IDX_COLS,
                                                feat_cols=SHIFTED_FEAT_COLS,
                                                tgt_cols=SHIFTED_TGT_COLS,
                                                time_order_col=TIME_ORDER_COL)
    else:
        _create_train_validation_splits_original(fIn,
                                                 idx_cols=IDX_COLS,
                                                 cov_cols=COV_COLS,
                                                 tgt_cols=TREAT_COLS)
    return


def _create_train_validation_splits_original(fOut,
                                             idx_cols,
                                             cov_cols,
                                             tgt_cols):
    # -------------------------------------------------
    # Splitting the data into segments
    # -------------------------------------------------
    data = pd.read_csv(fOut)
    data.set_index(idx_cols, inplace=True)

    split_df = data.reset_index().groupby(idx_cols[0])[TREAT_COLS].sum()
    split_df = split_df.sum(axis=1)
    split_df = pd.cut(split_df, bins=4, labels=False)

    split_size = 0.1
    _train, test = train_test_split(split_df.index, 
                                    test_size=split_size, stratify=split_df)
    train, val = train_test_split(_train, test_size=split_size,
                                  stratify=split_df.loc[_train])
    log.debug(f'Splits created. #Train: {len(train)} #Val: {len(val)} #Test: {len(test)}')
    split_idx = dict(train=train, test=test, val=val)
    
    # Splitting the data frames
    for segment in ['train', 'val', 'test']:   
        log.info(f"creating split for {segment}. #individual:{len(split_idx[segment])}")
        segment_data = data.loc[pd.IndexSlice[split_idx[segment]], :]
        _fname = fOut.parent / f'{fOut.name.rstrip(".csv")}_{segment}.csv'
        segment_data.to_csv(_fname, header=True)
    return


def _create_train_validation_splits_shifted(fOut_shifted, 
                                            idx_cols,
                                            feat_cols,
                                            tgt_cols,
                                            time_order_col):
    data_shifted = pd.read_csv(fOut_shifted)
    data_shifted.set_index(idx_cols, inplace=True)

    # Splitting dataframe to target and feature such that for each index in target, there is a history of covariates for feature
    group_idx, shift_idx = idx_cols
    
    tgt_df = data_shifted[tgt_cols]
    feat_df = (data_shifted[feat_cols].reset_index()
               .groupby(group_idx)
               .apply(create_shifted_targets(shift_index=shift_idx))
              )[feat_cols]
    feat_df.sort_index(inplace=True)
    tgt_df.to_csv(fOut_shifted.parent / f'{fOut_shifted.name.rstrip(".csv")}_shifted_TGT.csv')
    feat_df.to_csv(fOut_shifted.parent / f'{fOut_shifted.name.rstrip(".csv")}_shifted_FEAT.csv')
    log.info('Shifted data split into target and feat')

    # -------------------------------------------------
    # Splitting the data into segments
    # -------------------------------------------------
    split_size = 0.1
    _train, test = train_test_split(tgt_df.index, 
                                    test_size=split_size, stratify=tgt_df['treatment'])
    train, val = train_test_split(_train, test_size=split_size,
                                  stratify=tgt_df['treatment'].loc[_train])
    log.debug(f'Splits created. #Train: {len(train)} #Val: {len(val)} #Test: {len(test)}')
    split_idx = dict(train=train, test=test, val=val)
    
    # Setting up the feature df for split
    feat_df = feat_df.reset_index().set_index(idx_cols)
    for segment in ['train', 'val', 'test']:   
        log.info(f"creating split for {segment}")
        segment_y = tgt_df.loc[pd.IndexSlice[split_idx[segment]], tgt_cols]
        _fname = fOut_shifted.parent / f'{fOut_shifted.name.rstrip(".csv")}_shifted_TGT_{segment}.csv'
        segment_y.to_csv(_fname, header=True)

        segment_x = feat_df.loc[pd.IndexSlice[split_idx[segment]], :]
        _fname = fOut_shifted.parent / f'{fOut_shifted.name.rstrip(".csv")}_shifted_FEAT_{segment}.csv'
        segment_x.to_csv(_fname, header=True)
        log.debug(f"Target Sizes for {segment}: {segment_y.shape, segment_x.shape}")
    return

def parse_args():
    f"""{_desc_doc}"""
    ap = argparse.ArgumentParser(f'{Path(__file__).name}')
    # Main options
    ap.add_argument("-s", "--seed", metavar='SEED', required=False,
                    type=int, default=SEED,
                    help=f"seed value to sample the data. Default={SEED}")
    ap.add_argument("-t", "--num_timepoints", metavar='NUM_TIMEPOINTS', required=False,
                    type=int, default=NUM_TIMEPOINTS,
                    help=f"Number of timepoints to sample from. Default={NUM_TIMEPOINTS}")
    ap.add_argument("-n", "--num_samples", metavar='NUM_SAMPLES', required=False,
                    type=int, default=NUM_SAMPLES,
                    help=f"Number of samples. Default={NUM_SAMPLES}")
    ap.add_argument("-c", "--counterfactual", action="store_true",
                    help=f"Strategy to use. if provided use '{CF_STRATEGY}' else uses '{OBSV_STRATEGY}'")
    ap.add_argument('--shifted', action="store_true",
                    help="if provided, also generate shifted samples of chosen strategy")
    # Log options
    ap.add_argument('-v', '--verbose', action="store_true",
                    help="Log option: Write debug messages.")

    arg = ap.parse_args()
    return arg

def init_logs(arg, log):
    if arg and vars(arg).get('verbose', False):
        l = logging.DEBUG
    else:
        l = logging.INFO
    
    # printing to stdout
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    log.addHandler(ch)
    log.setLevel(l)
    return


def main():
    arg = parse_args()
    init_logs(arg, log)

    # Grabbing the hyper-params
    seed = arg.seed
    num_timepoints = arg.num_timepoints
    num_samples = arg.num_samples
    strategy = OBSV_STRATEGY if not arg.counterfactual else CF_STRATEGY
    log.info(f'Using strategy {strategy}. Create shifted={arg.shifted}')

    fOut = Path(__file__).absolute().parent / f'{EXPT_NAME}_{strategy}.csv'
    fOut_shifted = fOut.parent / f'{EXPT_NAME}_{strategy}_shifted.csv'

    np.random.seed(seed)
    data = np.vstack([generate_sample(num_timepoints)  # generate data for each sample
                      for i in range(num_samples)])

    # Massaging data into proper dataframe
    data = pd.DataFrame(data, columns=[IDX_COLS[1]] + TREAT_COLS + COV_COLS)
    data[IDX_COLS[0]] = np.repeat(np.arange(num_samples), num_timepoints)
    data = (data[IDX_COLS + TREAT_COLS + COV_COLS]
            .astype({TREAT_COLS[0]: int})     # converting treatment to integer
           )

    if not fOut.parent.is_dir():
        fOut.parent.mkdir(parents=True)
    data.to_csv(fOut, index=False)
    log.info(f"Sample genereated to {fOut}\tShape: ({num_samples} x {num_timepoints}) x 4")

    if strategy == OBSV_STRATEGY:
        create_train_validation_splits(fOut)
        log.info(f"Splitted samples outputted to {fOut.parent})")
   
    if arg.shifted:
        # generating shifted sampled data
        grouped = data.groupby(IDX_COLS[0])
        for x in COV_COLS:
            data[f'prev_{x}'] = grouped[x].shift(1)
        for x in TREAT_COLS:
            data[f'prev_{x}'] = grouped[x].shift(1)

        data = data[IDX_COLS + SHIFTED_FEAT_COLS + SHIFTED_TGT_COLS]
        data.set_index(IDX_COLS, inplace=True)
        data = data.loc[pd.IndexSlice[:, 1:], :]
        data.to_csv(fOut_shifted, index=True)
        log.info(f"Shifted Sample genereated to {fOut}\tShape: ({num_samples} x {num_timepoints}) x {data.shape[1]}")
        
        if strategy == OBSV_STRATEGY:
            create_train_validation_splits(fOut_shifted, shifted=True)
            log.info(f"Shifted and splitted samples outputted to {fOut.parent})")


if __name__ == "__main__":
    main()
