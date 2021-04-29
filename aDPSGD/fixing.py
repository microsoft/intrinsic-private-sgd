#!/usr/bin/env ipython
# The functions in this file relate to evaluating the performance of the model
# Specifically we are interested in the utility of models with different privacy levels

import numpy as np
import ipdb

import data_utils
import model_utils
from results_utils import ExperimentIdentifier, get_available_results
from run_experiment import load_cfg
from sklearn.metrics import log_loss


def fix_losses(cfg_name: str, max_t: int = 10000, cadence: int = 500, model: str = 'logistic'):
    for diffinit in [False, True]:
        df = get_available_results(cfg_name, model=model, replace_index=None,
                                   seed=None, diffinit=diffinit, data_privacy='all')
        n_exp = df.shape[0]
        print(f'Processing {n_exp} experiments...')
        for i, row in df.iterrows():
            if i % 100 == 0:
                print(i)
            seed = row['seed']
            replace_index = row['replace']
            recompute_performance_for_model(cfg_name, seed, replace_index, diffinit, max_t=max_t, cadence=cadence)
    return


def recompute_performance_for_model(cfg_name: str, seed: int, replace_index: int,
                                    diffinit: bool = False, verbose: bool =False,
                                    max_t: int = 10000, cadence: int = 500):
    cfg = load_cfg(cfg_name)
    exp = ExperimentIdentifier(cfg_name=cfg_name, seed=seed, replace_index=replace_index, diffinit=diffinit, model=cfg['model']['architecture'])
    weights_path = exp.path_stub().with_name(exp.path_stub().name + '.weights.csv')
    out_path = exp.path_stub().with_name(exp.path_stub().name + '.loss_FIXED.csv')
    if out_path.exists():
        print(f'WARNING: {out_path} already exists - skipping!')
        return
    assert not out_path.exists()
    fo = open(out_path, 'w')
    fo.write('t,minibatch_id,binary_crossentropy,binary_accuracy\n')
    # Now for the data
    x_train, y_train, x_vali, y_vali, x_test, y_test = data_utils.load_data(cfg['data'], replace_index=replace_index)
    # Time steps...
    time_steps = np.arange(0, max_t + 1, cadence)
    for t in time_steps:
        try:
            model = model_utils.build_model(**cfg['model'], init_path=weights_path, t=t)
        except ValueError:
            print(f'Out of time steps at t = {t}?')
            break
        # evaluate
        yhat_train = model(x_train).numpy().flatten()
        accuracy_train = ((yhat_train > 0.5)*1 == y_train).mean()
        ce_train = log_loss(y_train, yhat_train)
        yhat_vali = model(x_vali).numpy().flatten()
        accuracy_vali = ((yhat_vali > 0.5)*1 == y_vali).mean()
        ce_vali = log_loss(y_vali, yhat_vali)
        yhat_test = model(x_test).numpy().flatten()
        accuracy_test = ((yhat_test > 0.5)*1 == y_test).mean()
        ce_test = log_loss(y_test, yhat_test)
        if verbose:
            print(f'{t},ALL,{ce_train},{accuracy_train}')
            print(f'{t},VALI,{ce_vali},{accuracy_vali}')
            print(f'{t},TEST,{ce_test},{accuracy_test}')
        fo.write(f'{t},ALL,{ce_train},{accuracy_train}\n')
        fo.write(f'{t},VALI,{ce_vali},{accuracy_vali}\n')
        fo.write(f'{t},TEST,{ce_test},{accuracy_test}\n')
    fo.close()
    return
