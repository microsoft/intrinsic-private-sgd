#!/usr/bin/env ipython

import yaml
import os
from pathlib import Path


def check_cfg_for_consistency(cfg, verbose=False):
    if cfg['data']['binary']:
        assert cfg['model']['task_type'] == 'binary'
        assert cfg['model']['output_size'] == 1

    if 'flatten' in cfg['data']:
        if cfg['data']['flatten'] is True:
            assert type(cfg['model']['input_size']) == int
        else:
            assert len(cfg['model']['input_size']) > 1

    if verbose:
        if cfg['model']['architecture'] in ['logistic', 'linear']:
            if cfg['model']['hidden_size'] is not None:
                print('WARNING: Hidden size specified for logistic or linear model!')
        print('cfg passed checks')


def load_cfg(cfg_identifier):
    if '.yaml' in cfg_identifier:
        cfg_name = cfg_identifier.rstrip('.yaml')
    else:
        cfg_name = cfg_identifier
    cfg = yaml.load(open(os.path.join('cfgs', cfg_identifier + '.yaml')))
    cfg['cfg_name'] = cfg_name
    check_cfg_for_consistency(cfg)
    return cfg


def get_model_init_path(cfg, diffinit):
    if diffinit:
        init_path = None
    else:
        architecture = cfg['model']['architecture']
        cfg_name = cfg['cfg_name']
        init_path = f'{architecture}_{cfg_name}_init.h5'
        init_path = (Path('./models') / init_path).resolve()

    return init_path
