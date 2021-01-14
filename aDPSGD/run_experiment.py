#!/usr/bin/env ipython
# This is the script which runs the experiment! (trains a model!)

import argparse
from time import time
from tensorflow.keras.backend import clear_session

from model_utils import build_model, prep_for_training, train_model
from data_utils import load_data
from results_utils import ExperimentIdentifier
from cfg_utils import load_cfg, get_model_init_path


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
