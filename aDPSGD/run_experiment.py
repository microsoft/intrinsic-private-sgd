#!/usr/bin/env ipython
# This is the script which runs the experiment! (trains a model!)

import argparse
import yaml
import os
from pathlib import Path
from time import time

import model_utils
from data_utils import load_data
from results_utils import ExperimentIdentifier


def check_cfg_for_consistency(cfg):
    if cfg['data']['binary']:
        assert cfg['model']['task_type'] == 'binary'
        assert cfg['model']['output_size'] == 1

    if 'flatten' in cfg['data']:
        if cfg['data']['flatten'] is True:
            assert type(cfg['model']['input_size']) == int
        else:
            assert len(cfg['model']['input_size']) > 1

    if cfg['model']['architecture'] in ['logistic', 'linear']:
        if cfg['model']['hidden_size'] is not None:
            print('WARNING: Hidden size specified for logistic or linear model!')
    print('cfg passed checks')


def get_model_init_path(cfg, diffinit):
    if diffinit:
        init_path = None
    else:
        architecture = cfg['model']['architecture']
        cfg_name = cfg['cfg_name']
        init_path = f'{architecture}_{cfg_name}_init.h5'
        init_path = (Path('./models') / init_path).resolve()

    return init_path


def run_experiment(cfg, diffinit, seed, replace_index):
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
    model = model_utils.build_model(**cfg['model'], init_path=init_path)
    # prep model for training
    model_utils.prep_for_training(model, seed=seed,
                                  optimizer_settings=cfg['training']['optimization_algorithm'],
                                  task_type=cfg['model']['task_type'])
    # now train
    model_utils.train_model(model, cfg['training'], cfg['logging'],
                            x_train, y_train, x_vali, y_vali,
                            path_stub=path_stub)
    # clean up
    del model
    print('Finished after', time() - t0, 'seconds')


def load_cfg(cfg_identifier):
    if '.yaml' in cfg_identifier:
        cfg_name = cfg_identifier.rstrip('.yaml')
    else:
        cfg_name = cfg_identifier
    # cfg = yaml.safe_load(open(os.path.join('cfgs', args.cfg + '.yaml')))
    cfg = yaml.load(open(os.path.join('cfgs', cfg_identifier + '.yaml')))
    cfg['cfg_name'] = cfg_name
    check_cfg_for_consistency(cfg)
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='Name of yaml cfg of experiment')
    parser.add_argument('--diffinit', type=bool, help='Allow initialisation to vary with seed?', default=True)
    parser.add_argument('--seed', type=int, help='Random seed used for SGD', default=1)
    parser.add_argument('--replace_index', type=int, help='Which training example to replace with x0', default=None)
    args = parser.parse_args()
    cfg = load_cfg(args.cfg)

    run_experiment(cfg, args.diffinit, args.seed, args.replace_index)
