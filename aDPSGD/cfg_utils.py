#!/usr/bin/env ipython
# This is the script which runs the experiment! (trains a model!)

import yaml
import os
from pathlib import Path


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
