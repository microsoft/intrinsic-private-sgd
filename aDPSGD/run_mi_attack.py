#!/usr/bin/env ipython
# This is the script which runs the experiment! (trains a model!)

import argparse
import numpy as np
from pathlib import Path
from time import time
from tensorflow.keras.backend import clear_session
import statistics
import random
import csv
import os.path

from cfg_utils import load_cfg
from attacks import get_threshold, get_mi_attack_accuracy, get_epsilon 
from test_private_model import get_loss_for_mi_attack
from results_utils import get_available_results
from experiment_metadata import lr_convergence_points, nn_convergence_points


def run_mi_attack(cfg, exptype, t, runs, outputfile):
    results = {}
    cfg_name=cfg['cfg_name']
    model = cfg['model']['architecture']



    # Get the convergence point from experiment_metadata.py
    if t == None:
        if 'lr' in cfg_name:
            t = lr_convergence_points[cfg_name] 
        elif 'mlp' in cfg_name:
            t = nn_convergence_points[cfg_name]
        else:
            print('ERROR: Config file is neither a LR nor an MLP')


    if 'vi' in exptype:
        diffinit = False
    else:
        diffinit = True

    results['cfg_name'] = cfg_name
    results['model'] = model
    results['exptype'] = exptype
    results['diffinit'] = diffinit
    results['t'] = t

    noise_options = ['noiseless' ,'bolton'] #,'augment_sgd','augment_sgd_diffinit']

    df = get_available_results(cfg_name, model, replace_index=None, seed=None, diffinit=diffinit)
    available_seeds = df['seed'].value_counts().loc[lambda x : x>4].to_frame()
    available_seeds = df['seed'].tolist()

    for i in range(runs):
        threshold_seed = random.choice(available_seeds)
        available_seeds.remove(threshold_seed)
        new_seed = threshold_seed

        threshold_replace_indices = df[df['seed'] == threshold_seed]['replace'].unique().tolist() 
        th_replace_index = random.choice(threshold_replace_indices)

        threshold_replace_indices.remove(th_replace_index)
        new_replace_index = random.choice(threshold_replace_indices)

        for setting in noise_options:
            results['setting'] = setting
            loss = get_loss_for_mi_attack(cfg_name=cfg_name, replace_index=th_replace_index,
                                                                        seed=threshold_seed, t=t, epsilon=None,
                                                                        delta=None,
                                                                        sens_from_bound=True,
                                                                        metric_to_report='binary_crossentropy',
                                                                        verbose=False,
                                                                        num_deltas='max',
                                                                        multivariate=False)

            threshold = get_threshold(loss[setting][0])

            if 'vs' in exptype:
                new_available_seeds_with_threshold_ri = df[df['replace'] == th_replace_index]['seed'].unique().tolist()
                new_available_seeds_with_threshold_ri.remove(threshold_seed)
                new_seed = random.choice(new_available_seeds_with_threshold_ri)

                new_available_replace_indices = df[df['seed'] == new_seed]['replace'].unique().tolist()
                new_available_replace_indices.remove(th_replace_index)
                new_replace_index = random.choice(new_available_replace_indices)

                # Evaluate with different seed and same replace index
                loss = get_loss_for_mi_attack(cfg_name=cfg_name, replace_index=th_replace_index,
                                                                        seed=new_seed, t=t, epsilon=None,
                                                                        delta=None,
                                                                        sens_from_bound=True,
                                                                        metric_to_report='binary_crossentropy',
                                                                        verbose=False,
                                                                        num_deltas='max',
                                                                        multivariate=False)



            
            
            # Evaluate attack with the same replace index 
            attack_acc_with_same_ri = get_mi_attack_accuracy(loss[setting][0], loss[setting][1], threshold)
            epsilon_with_same_ri = get_epsilon(attack_acc_with_same_ri)           

            results['attack_acc_with_same_ri'] = attack_acc_with_same_ri
            results['epsilon_with_same_ri'] = epsilon_with_same_ri  
            
            # Evaluate attack with a different replace index 
            new_loss = get_loss_for_mi_attack(cfg_name=cfg_name, replace_index=new_replace_index,
                                                                        seed=new_seed, t=t, epsilon=None,
                                                                        delta=None,
                                                                        sens_from_bound=True,
                                                                        metric_to_report='binary_crossentropy',
                                                                        verbose=False,
                                                                        num_deltas='max',
                                                                        multivariate=False)
                
            attack_acc_with_diff_ri = get_mi_attack_accuracy(new_loss[setting][0], new_loss[setting][1], threshold)
            epsilon_with_diff_ri = get_epsilon(attack_acc_with_diff_ri)

            results['attack_acc_with_diff_ri'] = attack_acc_with_diff_ri
            results['epsilon_with_diff_ri'] = epsilon_with_diff_ri

            fieldnames = list(results.keys())
            append = os.path.exists(outputfile)

            if append:
                with open(outputfile, 'r', newline='') as logfile:
                    reader = csv.DictReader(logfile)
                    if not (reader.fieldnames == fieldnames):
                        print("Expected {0}".format(reader.fieldnames))
                        print("Got      {0}".format(fieldnames))
                        exit("Dictionary headers do not coincide")

            with open(outputfile, 'a', newline='') as logfile:
                csv_writer = csv.DictWriter(logfile, fieldnames=fieldnames)
                if not append: 
                    csv_writer.writeheader()
                csv_writer.writerow(results)


                    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='Name of yaml cfg of experiment')
    parser.add_argument('--exptype', type=str, help='fsfi, vsfi, fsvi, vsvi')
    parser.add_argument('--runs', type=int, default=5, help='Number of times to repeat an experiment')
    parser.add_argument('--output', type=str, default='all_mi_results.csv', help='Log file to store all the results')
    parser.add_argument('--t', type=int, default=None, help='Numer of iterations')
    args = parser.parse_args()
    cfg = load_cfg(args.cfg)

    run_mi_attack(cfg, args.exptype, args.t, args.runs, args.output)
