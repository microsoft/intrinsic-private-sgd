#!/usr/bin/env ipython
# This is the script which runs the experiment! (trains a model!)

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from time import time
from tensorflow.keras.backend import clear_session
import statistics
from scipy.stats import ttest_rel
import random
import csv
import os.path

from cfg_utils import load_cfg
from attacks import get_threshold, get_mi_attack_accuracy, get_epsilon 
from test_private_model import get_orig_loss_for_mi_attack
from results_utils import get_available_results
from experiment_metadata import lr_convergence_points, nn_convergence_points


def run_mi_attack(cfg, exptype, t, runs, outputfile):
    cfg_name=cfg['cfg_name']
    model = cfg['model']['architecture']
    noise_options = ['noiseless']# 'bolton'] #,'augment_sgd','augment_sgd_diffinit']
    sensitivity_bound = False
    
    if cfg_name == 'mnist_square_mlp':
        loss_metric = 'ce'
    elif cfg_name == 'cifar10_cnn':
        loss_metric = 'ce'
    else:
        loss_metric = 'binary_crossentropy'



    # Get the convergence point from experiment_metadata.py
    if t == None:
        if 'lr' in cfg_name:
            t = lr_convergence_points[cfg_name] 
        elif 'mlp' in cfg_name:
            t = nn_convergence_points[cfg_name]
        elif 'nn' in cfg_name:
            t = nn_convergence_points[cfg_name]
        else:
            print('ERROR: Config file is neither a LR nor an MLP')


    if 'vi' in exptype:
        diffinit = True
    else:
        diffinit = False

    df = get_available_results(cfg_name, model, replace_index=None, seed=None, diffinit=diffinit)
    # print(df)
    available_seeds = df['seed'].value_counts().loc[lambda x : x>1].index
    available_seeds = available_seeds.tolist()
    # print(available_seeds)
    print("***No. of available seeds", len(available_seeds))

    for i in range(runs):
        threshold_seed = random.choice(available_seeds)
        available_seeds.remove(threshold_seed)
        new_seed = threshold_seed

        threshold_replace_indices = df[df['seed'] == threshold_seed]['replace'].unique().tolist() 
        print(threshold_replace_indices)
        th_replace_index = random.choice(threshold_replace_indices)

        threshold_replace_indices.remove(th_replace_index)
        new_replace_index = random.choice(threshold_replace_indices)

        for setting in noise_options:
            results = {}
            results['cfg_name'] = cfg_name
            results['model'] = model
            results['exptype'] = exptype
            results['diffinit'] = diffinit
            results['t'] = t
            results['setting'] = setting
            results['th_seed'] = threshold_seed
            results['th_ri'] = th_replace_index

            loss_train, loss_test = get_orig_loss_for_mi_attack(cfg_name=cfg_name, replace_index=th_replace_index,
                                                                        seed=threshold_seed, t=t, epsilon=None,
                                                                        delta=None,
                                                                        sens_from_bound=sensitivity_bound,
                                                                        metric_to_report=loss_metric,
                                                                        verbose=False,
                                                                        num_deltas='max',
                                                                        multivariate=False, diffinit = diffinit)

            threshold = get_threshold(loss_train, loss_test)
            print("Threshold is ", threshold)

            if 'vs' in exptype:
                new_available_seeds_with_threshold_ri = df[df['replace'] == th_replace_index]['seed'].unique().tolist()
                new_available_seeds_with_threshold_ri.remove(threshold_seed)
                new_seed = random.choice(new_available_seeds_with_threshold_ri)

                new_available_replace_indices = df[df['seed'] == new_seed]['replace'].unique().tolist()
                new_available_replace_indices.remove(th_replace_index)
                new_replace_index = random.choice(new_available_replace_indices)

                # Evaluate with different seed and same replace index
                loss_train, loss_test = get_orig_loss_for_mi_attack(cfg_name=cfg_name, replace_index=th_replace_index,
                                                                        seed=new_seed, t=t, epsilon=None,
                                                                        delta=None,
                                                                        sens_from_bound=sensitivity_bound,
                                                                        metric_to_report=loss_metric,
                                                                        verbose=False,
                                                                        num_deltas='max',
                                                                        multivariate=False, diffinit = diffinit)



            
            
            # Evaluate attack with the same replace index 
            attack_acc_with_same_ri = get_mi_attack_accuracy(loss_train, loss_test, threshold)
            epsilon_with_same_ri = get_epsilon(attack_acc_with_same_ri)           

            results['attack_acc_with_same_ri'] = attack_acc_with_same_ri
            results['epsilon_with_same_ri'] = epsilon_with_same_ri  
            
            results['attack_seed'] = new_seed
            results['attack_ri'] = new_replace_index

            # Evaluate attack with a different replace index 
            new_loss_train, new_loss_test = get_orig_loss_for_mi_attack(cfg_name=cfg_name, replace_index=new_replace_index,
                                                                        seed=new_seed, t=t, epsilon=None,
                                                                        delta=None,
                                                                        sens_from_bound=sensitivity_bound,
                                                                        metric_to_report=loss_metric,
                                                                        verbose=False,
                                                                        num_deltas='max',
                                                                        multivariate=False, diffinit = diffinit)
                
            attack_acc_with_diff_ri = get_mi_attack_accuracy(new_loss_train, new_loss_test, threshold)
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


def analyse_mi_results(df: pd.DataFrame):
    """
    For every attack_seed, compare accuracy when th_seed is either
    1) the same
    2) different
    for fixed and variable initialisation scenarios
    """
    # this is a bit gross but it'll do
    # want to create a dataframe where each row is an attack_seed (model we are attacking)
    # columns:
    # -- attack accuracy with same seed and fixed init
    accuracy_same_fix = []
    # -- attack accuracy with diff seed and fixed init
    accuracy_diff_fix = []
    # -- attack accuracy with same seed and diff init
    accuracy_same_diff = []
    # -- attack accuracy with diff seed and diff init
    accuracy_diff_diff = []
    # -- epsilon with same seed and fixed init
    epsilon_same_fix = []
    # -- attack epsilon with diff seed and fixed init
    epsilon_diff_fix = []
    # -- epsilon with same seed and diff init
    epsilon_same_diff = []
    # -- epsilon with diff seed and diff init
    epsilon_diff_diff = []

    attack_seeds = df['attack_seed'].unique()
    for a_seed in attack_seeds:
        sub_df = df.loc[df['attack_seed'] == a_seed]
        # Same Fix
        same_fix = sub_df.loc[(sub_df['th_seed'] == a_seed) & (~sub_df['diffinit'])]
        try:
            accuracy_same_fix.append(same_fix['attack_acc_with_same_ri'].iloc[0])
            epsilon_same_fix.append(same_fix['epsilon_with_same_ri'].iloc[0])
        except IndexError:
            accuracy_same_fix.append(np.nan)
            epsilon_same_fix.append(np.nan)
        # Diff Fix
        diff_fix = sub_df.loc[(sub_df['th_seed'] != a_seed) & (~sub_df['diffinit'])]
        try:
            accuracy_diff_fix.append(diff_fix['attack_acc_with_same_ri'].iloc[0])
            epsilon_diff_fix.append(diff_fix['epsilon_with_same_ri'].iloc[0])
        except IndexError:
            accuracy_diff_fix.append(np.nan)
            epsilon_diff_fix.append(np.nan)
        # Same Diff
        same_diff = sub_df.loc[(sub_df['th_seed'] == a_seed) & (sub_df['diffinit'])]
        try:
            accuracy_same_diff.append(same_diff['attack_acc_with_same_ri'].iloc[0])
            epsilon_same_diff.append(same_diff['epsilon_with_same_ri'].iloc[0])
        except IndexError:
            accuracy_same_diff.append(np.nan)
            epsilon_same_diff.append(np.nan)
        # Diff Diff
        diff_diff = sub_df.loc[(sub_df['th_seed'] != a_seed) & (sub_df['diffinit'])]
        try:
            accuracy_diff_diff.append(diff_diff['attack_acc_with_same_ri'].iloc[0])
            epsilon_diff_diff.append(diff_diff['epsilon_with_same_ri'].iloc[0])
        except IndexError:
            accuracy_diff_diff.append(np.nan)
            epsilon_diff_diff.append(np.nan)

    results = pd.DataFrame({'accuracy_same_fix': accuracy_same_fix,
                            'accuracy_diff_fix': accuracy_diff_fix,
                            'accuracy_same_diff': accuracy_same_diff,
                            'accuracy_diff_diff': accuracy_diff_diff,
                            'epsilon_same_fix': epsilon_same_fix,
                            'epsilon_diff_fix': epsilon_diff_fix,
                            'epsilon_same_diff': epsilon_same_diff,
                            'epsilon_diff_diff': epsilon_diff_diff}, index=attack_seeds)

    # Now test: either fixinit OR diffinit:
    # -- Is accuracy with same seed different to accuracy with diff seed?
    print('Analysis with paired t-test:')
    print('\tAccuracy:')
    fix = ttest_rel(results['accuracy_same_fix'], results['accuracy_diff_fix'], nan_policy='omit')
    delta_fix = (results['accuracy_same_fix'] - results['accuracy_diff_fix']).mean()
    diff = ttest_rel(results['accuracy_same_diff'], results['accuracy_diff_diff'], nan_policy='omit')
    delta_diff = (results['accuracy_same_diff'] - results['accuracy_diff_diff']).mean()
    print(f'\t\tFixinit: \tStat: {fix.statistic:.4f} \tp: {fix.pvalue:.4f} \tmean delta: {delta_fix:.4f}')
    print(f'\t\tDiffinit: \tStat: {diff.statistic:.4f} \tp: {diff.pvalue:.4f} \tmean delta: {delta_diff:.4f}')
    print('\tEpsilon:')
    fix = ttest_rel(results['epsilon_same_fix'], results['epsilon_diff_fix'], nan_policy='omit')
    delta_fix = (results['epsilon_same_fix'] - results['epsilon_diff_fix']).mean()
    diff = ttest_rel(results['epsilon_same_diff'], results['epsilon_diff_diff'], nan_policy='omit')
    delta_diff = (results['epsilon_same_diff'] - results['epsilon_diff_diff']).mean()
    print(f'\t\tFixinit: \tStat: {fix.statistic:.4f} \tp: {fix.pvalue:.4f} \tmean delta: {delta_fix:.4f}')
    print(f'\t\tDiffinit: \tStat: {diff.statistic:.4f} \tp: {diff.pvalue:.4f} \tmean delta: {delta_diff:.4f}')
    print('\nNote: Delta is SAME SEED - DIFF SEED')
    print('...so delta > 0 means attacking is "easier" (-> less privacy) with the same seed')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='Name of yaml cfg of experiment')
    parser.add_argument('--exptype', type=str, help='fsfi, vsfi, fsvi, vsvi')
    parser.add_argument('--runs', type=int, default=5, help='Number of times to repeat an experiment')
    parser.add_argument('--output', type=str, default='all_cifar10_cnn_results.csv', help='Log file to store all the results')
    parser.add_argument('--t', type=int, default=None, help='Numer of iterations')
    args = parser.parse_args()
    cfg = load_cfg(args.cfg)

    run_mi_attack(cfg, args.exptype, args.t, args.runs, args.output)
