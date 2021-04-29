#!/usr/bin/env ipython
# This is the script which runs the experiment! (trains a model!)

import argparse
import numpy as np
from pathlib import Path
from time import time
from tensorflow.keras.backend import clear_session
import statistics

from model_utils import build_model, prep_for_training, train_model
from data_utils import load_data
from results_utils import ExperimentIdentifier
from cfg_utils import load_cfg
from attacks import simple_membership_inference #, stats_membership_inference, mia
# from derived_results import find_convergence_point
from test_private_model import get_loss_for_mi_attack
from results_utils import get_available_results



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
    model = cfg['model']['architecture']

    noise_options = ['noiseless' ,'bolton'] #,'augment_sgd','augment_sgd_diffinit']
    frac = 0.6

    df = get_available_results(cfg_name, model, replace_index=None, seed=None, diffinit=diffinit)
    available_seeds = df['seed'].unique().tolist()
    replace_indices = df['replace'].unique().tolist()

    print("no. of replace index found", len(replace_indices))

    all_eps = {}

    for setting in noise_options:
        all_eps[setting] = {'max':[], 'median': [], 'mean': []}

    for i in range(0,20):
        loss = get_loss_for_mi_attack(cfg_name=cfg_name, replace_index=replace_index,
                                                                        seed=available_seeds[i], t=t, epsilon=None,
                                                                        delta=None,
                                                                        sens_from_bound=True,
                                                                        metric_to_report='binary_crossentropy',
                                                                        verbose=False,
                                                                        num_deltas='max',
                                                                        multivariate=False)


        for setting in noise_options:
            print("Mi attack accuracy for "+setting+ " model")
            eps_median, eps_mean = simple_membership_inference(loss[setting][0], loss[setting][1], frac)# , clipping_norm, noise_multiplier, epochs, dataset_name, model_type) #linear classifcation
            print("Eps_median, eps_mean", eps_median, eps_mean)
            # all_eps[setting]['max'].append(eps_max)
            all_eps[setting]['median'].append(eps_median)
            all_eps[setting]['mean'].append(eps_mean)

    for setting in noise_options:
        print("For "+setting+" :")
        for thresh in ['mean', 'median']:
            print("Eps_"+thresh+" :", statistics.mean(all_eps[setting][thresh]), " stddev:", statistics.stdev(all_eps[setting][thresh]))

                    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='Name of yaml cfg of experiment')
    parser.add_argument('--diffinit', type=bool, help='Allow initialisation to vary with seed?', default=False)
    parser.add_argument('--seed', type=int, help='Random seed used for SGD', default=10)
    parser.add_argument('--replace_index', type=int, help='Which training example to replace with x0', default=None)
    parser.add_argument('--t', type=int, help='Time point at which to run derived experiments', default=None)
    args = parser.parse_args()
    cfg = load_cfg(args.cfg)

    
    run_mi_attack(cfg, args.diffinit, args.seed, args.replace_index, args.t)
