#!/usr/bin/env ipython
# This is the script which runs the experiment! (trains a model!)

import argparse
import yaml
import os
from pathlib import Path
from time import time

import ipdb

import model_utils
from data_utils import load_data

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

def get_model_init_path(cfg):
    if cfg['model']['diffinit'] is True:
        init_path = None
    else:
        architecture = cfg['model']['architecture']
        cfg_name = cfg['cfg_name']
        init_path = os.path.join('models', architecture + '_' + cfg_name + '_init.h5')
    return init_path

def get_experiment_identifier(cfg, seed, replace_index):
    identifier_pieces = [cfg['cfg_name'],
            cfg['model']['architecture'],
            'diffinit'*cfg['model']['diffinit'],
            'replace' + str(replace_index),
            'seed' + str(seed)]
    experiment_identifier = '_'.join(identifier_pieces)
    return experiment_identifier

def run_experiment(cfg, seed, replace_index):
    t0 = time()
    experiment_identifier = get_experiment_identifier(cfg, seed, replace_index)
    print('Running experiment with identifier', experiment_identifier)
    # load data
    x_train, y_train, x_vali, y_vali, x_test, y_test = load_data(options=cfg['data'], replace_index=replace_index)
    # define model
    init_path = get_model_init_path(cfg)
    model = model_utils.build_model(**cfg['model'], init_path=init_path)
    # prep model for training
    model_utils.prep_for_training(model, seed=seed,
            optimizer_settings=cfg['training']['optimization_algorithm'],
            task_type=cfg['model']['task_type'])
    # now train
    model_utils.train_model(model, cfg['training'], cfg['logging'], 
            x_train, y_train, x_vali, y_vali,
            experiment_identifier=experiment_identifier)
    # clean up
    del model
    print('Finished after', time() - t0, 'seconds')

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, help='Name of yaml cfg of experiment')
parser.add_argument('--seed', type=int, help='Random seed used for SGD', default=1)
parser.add_argument('--replace_index', type=int, help='Which training example to replace with x0', default=None)
args = parser.parse_args()
if '.yaml' in args.cfg:
    args.cfg = args.cfg.rstrip('.yaml')
#cfg = yaml.safe_load(open(os.path.join('cfgs', args.cfg + '.yaml')))
cfg = yaml.load(open(os.path.join('cfgs', args.cfg + '.yaml')))
cfg['cfg_name'] = args.cfg
check_cfg_for_consistency(cfg)
run_experiment(cfg, args.seed, args.replace_index)
