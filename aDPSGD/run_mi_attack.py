#!/usr/bin/env ipython
# This is the script which runs the experiment! (trains a model!)

import argparse
from pathlib import Path
from time import time
from tensorflow.keras.backend import clear_session

from model_utils import build_model, prep_for_training, train_model
from data_utils import load_data
from results_utils import ExperimentIdentifier
from cfg_utils import load_cfg
#from attacks import simple_membership_inference, obtain_loss, stats_membership_inference, mia
# from derived_results import find_convergence_point
from test_private_model import get_loss_for_mi_attack



def get_model_init_path(cfg, diffinit):
    if diffinit:
        init_path = None
    else:
        architecture = cfg['model']['architecture']
        cfg_name = cfg['cfg_name']
        init_path = f'{architecture}_{cfg_name}_init.h5'
        init_path = (Path('./models') / init_path).resolve()

    return init_path


def run_mi_attack(cfg, diffinit, seed, replace_index, t):
    cfg_name=cfg['cfg_name']

    
    print('Loading dataset')
    # load data
    x_train, y_train, x_vali, y_vali, x_test, y_test = load_data(options=cfg['data'], replace_index=replace_index)
   
    # if t is None:
    #     t, valid_frac = find_convergence_point(cfg_name, model, diffinit=diffinit,
    #                                            tolerance=3, metric='binary_accuracy', data_privacy='all')

    #     if valid_frac < 0.5:
    #         raise ValueError(f'Convergence point not good, valid fraction: {valid_frac}')
    #     else:
    #         print(f'Selecting t as convergence point {t}, valid fraction {valid_frac}')

    results_train = get_loss_for_mi_attack(cfg_name=cfg_name, x_value=x_train, y_value=y_train, replace_index=replace_index,
                                                                       seed=seed, t=t, epsilon=None,
                                                                       delta=None,
                                                                       sens_from_bound=True,
                                                                       metric_to_report='binary_crossentropy',
                                                                       verbose=False,
                                                                       num_deltas='max',
                                                                       multivariate=False)

    results_test = get_loss_for_mi_attack(cfg_name=cfg_name, x_value=x_test, y_value=y_test, replace_index=replace_index,
                                                                       seed=seed, t=t, epsilon=None,
                                                                       delta=None,
                                                                       sens_from_bound=True,
                                                                       metric_to_report='binary_crossentropy',
                                                                       verbose=False,
                                                                       num_deltas='max',
                                                                       multivariate=False)

    print(len(results_train))
    #simple_membership_inference(train_loss, test_loss, frac, clipping_norm, noise_multiplier, epochs, dataset_name, model_type) #linear classifcation
            
    #stats_membership_inference(train_loss, test_loss, frac, clipping_norm, noise_multiplier, epochs, dataset_name, model_type) #based on quartiles
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='Name of yaml cfg of experiment')
    parser.add_argument('--diffinit', type=bool, help='Allow initialisation to vary with seed?', default=False)
    parser.add_argument('--seed', type=int, help='Random seed used for SGD', default=1)
    parser.add_argument('--replace_index', type=int, help='Which training example to replace with x0', default=None)
    parser.add_argument('--t', type=int, help='Time point at which to run derived experiments', default=None)
    args = parser.parse_args()
    cfg = load_cfg(args.cfg)

    
    run_mi_attack(cfg, args.diffinit, args.seed, args.replace_index, args.t)
