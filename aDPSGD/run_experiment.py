#!/usr/bin/env ipython
# This is the script which runs the experiment! (trains a model!)

import argparse
from time import time
from tensorflow.keras.backend import clear_session
import numpy as np
import pandas as pd
from itertools import product

from model_utils import build_model, prep_for_training, train_model
from data_utils import load_data
from results_utils import ExperimentIdentifier, get_available_results
from cfg_utils import load_cfg, get_model_init_path
from experiment_metadata import get_dataset_size


def find_gaps_in_grid(cfg) -> list:
    """
    We want to have run every seed against every replace index.
    This will look at what we've already run and identify missing pairs.
    """
    df = get_available_results(cfg['cfg_name'], cfg['model']['architecture'],
                               replace_index=None, seed=None, diffinit=True)
    xtab = pd.crosstab(df['seed'], df['replace'])
    # Every value in the crosstabulation should be 2
    xtab_miss = np.where(xtab < 2)
    missing_pairs = [(x[0], x[1]) for x in np.array(xtab_miss).T]
    print(f'Identified {len(missing_pairs)} missing pairs:\n{missing_pairs}')
    print('WARNING: This might be a large number!')
    return missing_pairs


def propose_seeds_and_replaces(cfg, num_seeds, num_replaces) -> list:
    """
    We want to run num_seeds *new* seeds and num_replaces *new* replaces.
    This function will look at what we've already run, and propose new pairs.

    If we haven't run any experiments yet, this will just create a "new" grid.
    """
    df = get_available_results(cfg['cfg_name'], cfg['model']['architecture'],
                               replace_index=None, seed=None, diffinit=True)
    known_seeds = df['seed'].unique()
    known_replaces = df['replace'].unique()

    candidate_seeds = [x for x in range(99999) if x not in known_seeds]
    new_seeds = np.random.choice(candidate_seeds, num_seeds, replace=False)

    N = get_dataset_size(cfg['data'])
    candidate_replaces = [x for x in range(N) if x not in known_replaces]
    new_replaces = np.random.choice(candidate_replaces, num_replaces, replace=False)

    pairs = list(product(new_seeds, new_replaces))
    return pairs


def add_new_seeds_to_grid(cfg, num_seeds: int, num_replaces: int) -> list:
    """
    Specifically run more seeds.
    First get existing seeds and replaces.
    Sample new seeds, then run new seeds for each existing replace.
    """
    df = get_available_results(cfg['cfg_name'], cfg['model']['architecture'],
                               replace_index=None, seed=None, diffinit=True)
    known_seeds = df['seed'].unique()
    replaces_with_counts = df['replace'].value_counts()
    num_known_replaces = replaces_with_counts.shape[0]
    if num_replaces > num_known_replaces:
        print(f'Asked for {num_replaces} replaces but only {num_known_replaces} known- taking these.')
        replaces = df['replace'].unique()
    else:
        print(f'Asked for {num_replaces} and there are {num_known_replaces} so we are taking the top ones.')
        # this is to enrich a smaller set of replaces with seeds
        replaces = replaces_with_counts.iloc[:num_replaces].index
        print(f'These are: {replaces}')

    candidate_seeds = [x for x in range(99999) if x not in known_seeds]
    new_seeds = np.random.choice(candidate_seeds, num_seeds, replace=False)

    pairs = list(product(new_seeds, replaces))
    return pairs


def run_single_experiment(cfg, diffinit, seed, replace_index):
    t0 = time()
    # how we convert the cfg into a path and such is defined in ExperimentIdentifier
    exp = ExperimentIdentifier(seed=seed, replace_index=replace_index, diffinit=diffinit)
    exp.init_from_cfg(cfg)
    exp.ensure_directory_exists(verbose=True)
    path_stub = exp.path_stub()
    print('Running experiment with path', path_stub)
    # load data
    x_train, y_train, x_vali, y_vali, x_test, y_test = load_data(options=cfg['data'], replace_index=replace_index)
    # define model
    init_path = get_model_init_path(cfg, diffinit)
    model = build_model(**cfg['model'], init_path=init_path)
    # prep model for training
    prep_for_training(model, seed=seed,
                      optimizer_settings=cfg['training']['optimization_algorithm'],
                      task_type=cfg['model']['task_type'])
    # now train
    train_model(model, cfg['training'], cfg['logging'],
                x_train, y_train, x_vali, y_vali,
                path_stub=path_stub)
    # clean up
    del model
    clear_session()
    print('Finished after', time() - t0, 'seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='Name of yaml cfg of experiment')
    parser.add_argument('--diffinit', type=bool, help='Allow initialisation to vary with seed?', default=False)
    parser.add_argument('--seed', type=int, help='Random seed used for SGD', default=1)
    parser.add_argument('--replace_index', type=int, help='Which training example to replace with x0', default=None)
    args = parser.parse_args()
    cfg = load_cfg(args.cfg)

    run_single_experiment(cfg, args.diffinit, args.seed, args.replace_index)
