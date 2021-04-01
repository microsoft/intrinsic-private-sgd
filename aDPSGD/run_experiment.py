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
from cfg_utils import load_cfg, get_model_init_path, get_dataset_size


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


def add_new_seeds_to_grid(cfg, num_seeds) -> list:
    """
    Specifically run more seeds.
    First get existing seeds and replaces.
    Sample new seeds, then run new seeds for each existing replace.
    """
    df = get_available_results(cfg['cfg_name'], cfg['model']['architecture'],
                               replace_index=None, seed=None, diffinit=True)
    known_seeds = df['seed'].unique()
    known_replaces = df['replace'].unique()

    candidate_seeds = [x for x in range(99999) if x not in known_seeds]
    new_seeds = np.random.choice(candidate_seeds, num_seeds, replace=False)

    pairs = list(product(new_seeds, known_replaces))
    return pairs


def get_grid_to_run(cfg, num_seeds, num_replaces):
    # Get existing results - we will add to these
    df = get_available_results(cfg['cfg_name'], cfg['model']['architecture'],
                               replace_index=None, seed=None, diffinit=True)
    known_seeds = df['seed'].unique()
    known_replaces = df['replace'].unique()

    exp = ExperimentIdentifier()
    exp.init_from_cfg(cfg)
    grid_path = exp.derived_path_stub() / 'grid.csv'
    try:
        grid = pd.read_csv(grid_path)
        print(f'Loaded grid seeds and replace indices from {grid_path}')
        seeds = grid[grid['what'] == 'seed']['value']
        replaces = grid[grid['what'] == 'replace_index']['value']
    except FileNotFoundError:
        seeds = []
        replaces = []
    if len(seeds) < num_seeds:
        candidate_seeds = [x for x in range(99999) if x not in seeds]
        new_seeds = np.random.choice(candidate_seeds, num_seeds - len(seeds), replace=False)
        seeds = np.concatenate([seeds, new_seeds])
    if len(replaces) < num_replaces:
        N = get_dataset_size(cfg['data'])
        candidate_replaces = [x for x in range(N) if x not in replaces]
        new_replaces = np.random.choice(candidate_replaces, num_replaces - len(replaces))
        replaces = np.concatenate([replaces, new_replaces])
    seeds = np.int32(seeds)
    replaces = np.int32(replaces)
    what = ['seed']*len(seeds) + ['replace_index']*len(replaces)
    values = np.concatenate([seeds, replaces])
#    grid = pd.DataFrame({'value': values, 'what': what})
#    grid.to_csv(grid_path, index=False)
#    print(f'Saved grid seeds and replace indices to {grid_path}')
    return seeds, replaces


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
