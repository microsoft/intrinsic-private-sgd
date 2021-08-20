#!/usr/bin/env ipython
# This is the script which runs the experiment! (trains a model!)

import argparse
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
import random
import csv
import os.path

from cfg_utils import load_cfg
from attacks import get_threshold, get_mi_attack_accuracy, get_epsilon, get_classifier
from test_private_model import get_orig_loss_for_mi_attack, get_activations_for_mi_attack
from results_utils import get_available_results
from experiment_metadata import lr_convergence_points, nn_convergence_points


def run_mi_attack_steph(cfg, exptype: str, t: int, runs: int, outputfile: str,
                        use_loss: bool = True,
                        use_activations: bool = False):
    cfg_name = cfg['cfg_name']
    model = cfg['model']['architecture']

    if use_loss:
        print('Using loss!')
        if cfg_name == 'mnist_square_mlp':
            loss_metric = 'ce'
        elif cfg_name == 'cifar10_cnn':
            loss_metric = 'ce'
        else:
            loss_metric = 'binary_crossentropy'

    if use_activations:
        layer = -1
        print(f'\t\tRunning attack based on activations after layer {layer}')

    # Get the convergence point from experiment_metadata.py
    if t is None:
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
    available_seeds = df['seed'].value_counts().loc[lambda x: x > 1].index
    available_seeds = available_seeds.tolist()
    # print(available_seeds)
    print("[run_mi_attack] No. of available seeds", len(available_seeds))

    # Each model we attack is a seed + replace index
    run_counter = 0
    for attack_seed in np.random.permutation(available_seeds):
        available_replace_indices = df[df['seed'] == attack_seed]['replace'].unique().tolist()
        for attack_index in available_replace_indices:
            print(f'[run_mi_attack] Attacking model with seed {attack_seed} and replace index {attack_index}')

            if use_loss:
                # First we attack it with itself
                train_loss_self, test_loss_self = get_orig_loss_for_mi_attack(cfg_name=cfg_name,
                                                                              replace_index=attack_index,
                                                                              seed=attack_seed,
                                                                              t=t,
                                                                              metric_to_report=loss_metric,
                                                                              verbose=False,
                                                                              diffinit=diffinit)
            else:
                train_loss_self = None
                test_loss_self = None
            if use_activations:
                train_act_self, test_act_self = get_activations_for_mi_attack(cfg_name,
                                                                              replace_index=attack_index,
                                                                              seed=attack_seed,
                                                                              t=t,
                                                                              layer=layer,
                                                                              verbose=False,
                                                                              diffinit=diffinit)
            else:
                train_act_self = None
                test_act_self = None

            # If we have ONLY the loss, we use threshold
            if use_activations:
                # If we have the loss, we combine it
                if train_loss_self is not None:
                    assert test_loss_self is not None
                    train_self = np.hstack([train_act_self, np.array(train_loss_self).reshape(-1, 1)])
                    test_self = np.hstack([test_act_self, np.array(test_loss_self).reshape(-1, 1)])
                else:
                    train_self = train_act_self
                    test_self = test_act_self
                classifier_self = get_classifier(train_self, test_self)
            else:
                assert use_loss
                classifier_self = get_threshold(train_loss_self)
                print("[run_mi_attack] Threshold is ", classifier_self)
                train_self = train_loss_self
                test_self = test_loss_self

            print('*** SELF CLASSIFIER ON SELF TEST DATA: ***')
            attack_accuracy_self = get_mi_attack_accuracy(train_self, test_self, classifier_self)
            epsilon_self = get_epsilon(attack_accuracy_self)

            # Now we attack with a threshold taken from another model
            # TODO we could atack with multiple seeds now if we wanted
            other_seed = np.random.choice([x for x in available_seeds if not x == attack_seed])
            assert not other_seed == attack_seed
            print(f'[run_mi_attack] attacking with DIFFERENT seed {other_seed}')

            if use_loss:
                # First we attack it with itself
                train_loss_other, test_loss_other = get_orig_loss_for_mi_attack(cfg_name=cfg_name,
                                                                                replace_index=attack_index,
                                                                                seed=other_seed,
                                                                                t=t,
                                                                                metric_to_report=loss_metric,
                                                                                verbose=False,
                                                                                diffinit=diffinit)
            else:
                train_loss_other = None
                test_loss_other = None
            if use_activations:
                train_act_other, test_act_other = get_activations_for_mi_attack(cfg_name,
                                                                                replace_index=attack_index,
                                                                                seed=other_seed,
                                                                                t=t,
                                                                                layer=layer,
                                                                                verbose=False,
                                                                                diffinit=diffinit)
            else:
                train_act_other = None
                test_act_other = None

            # If we have ONLY the loss, we use threshold
            if use_activations:
                # If we have the loss, we combine it
                if train_loss_other is not None:
                    assert test_loss_other is not None
                    train_other = np.hstack([train_act_other, np.array(train_loss_other).reshape(-1, 1)])
                    test_other = np.hstack([test_act_other, np.array(test_loss_other).reshape(-1, 1)])
                else:
                    train_other = train_act_other
                    test_other = test_act_other
                classifier_other = get_classifier(train_other, test_other)
            else:
                assert use_loss
                classifier_other = get_threshold(train_loss_other)
                print("Threshold is ", classifier_other)
                train_other = train_loss_other
                test_other = test_loss_other

            print('*** OTHER CLASSIFIER ON SELF TEST DATA: ***')
            attack_accuracy_other = get_mi_attack_accuracy(train_self, test_self, classifier_other)
            epsilon_other = get_epsilon(attack_accuracy_other)

            results = {}
            results['cfg_name'] = cfg_name
            results['model'] = model
            results['exptype'] = exptype
            results['diffinit'] = diffinit
            results['t'] = t
            results['use_loss'] = use_loss
            results['use_activations'] = use_activations
            results['attack_seed'] = attack_seed
            results['other_seed'] = other_seed
            results['attack_index'] = attack_index
            results['accuracy_self'] = attack_accuracy_self
            results['epsilon_self'] = epsilon_self
            results['accuracy_other'] = attack_accuracy_other
            results['epsilon_other'] = epsilon_other

            run_counter += 1

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

        if run_counter > runs:
            print(f'Hit max runs - stopping after {run_counter}')
            break


def run_mi_attack(cfg, exptype, t, runs, outputfile):
    cfg_name = cfg['cfg_name']
    model = cfg['model']['architecture']

    if cfg_name == 'mnist_square_mlp':
        loss_metric = 'ce'
    elif cfg_name == 'cifar10_cnn':
        loss_metric = 'ce'
    else:
        loss_metric = 'binary_crossentropy'

    # Get the convergence point from experiment_metadata.py
    if t is None:
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
    available_seeds = df['seed'].value_counts().loc[lambda x: x > 1].index
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

        results = {}
        results['cfg_name'] = cfg_name
        results['model'] = model
        results['exptype'] = exptype
        results['diffinit'] = diffinit
        results['t'] = t
        results['th_seed'] = threshold_seed
        results['th_ri'] = th_replace_index

        loss_train, loss_test = get_orig_loss_for_mi_attack(cfg_name=cfg_name,
                                                            replace_index=th_replace_index,
                                                            seed=threshold_seed,
                                                            t=t,
                                                            metric_to_report=loss_metric,
                                                            verbose=False,
                                                            diffinit=diffinit)

        threshold = get_threshold(loss_train)
        print("Threshold is ", threshold)

        if 'vs' in exptype:
            new_available_seeds_with_threshold_ri = df[df['replace'] == th_replace_index]['seed'].unique().tolist()
            new_available_seeds_with_threshold_ri.remove(threshold_seed)
            new_seed = random.choice(new_available_seeds_with_threshold_ri)

            new_available_replace_indices = df[df['seed'] == new_seed]['replace'].unique().tolist()
            new_available_replace_indices.remove(th_replace_index)
            new_replace_index = random.choice(new_available_replace_indices)

            # Evaluate with different seed and same replace index
            loss_train, loss_test = get_orig_loss_for_mi_attack(cfg_name=cfg_name,
                                                                replace_index=th_replace_index,
                                                                seed=new_seed,
                                                                t=t,
                                                                metric_to_report=loss_metric,
                                                                verbose=False,
                                                                diffinit=diffinit)

        # Evaluate attack with the same replace index
        attack_acc_with_same_ri = get_mi_attack_accuracy(loss_train, loss_test, threshold)
        epsilon_with_same_ri = get_epsilon(attack_acc_with_same_ri)

        results['attack_acc_with_same_ri'] = attack_acc_with_same_ri
        results['epsilon_with_same_ri'] = epsilon_with_same_ri

        results['attack_seed'] = new_seed
        results['attack_ri'] = new_replace_index

        # Evaluate attack with a different replace index
        new_loss_train, new_loss_test = get_orig_loss_for_mi_attack(cfg_name=cfg_name,
                                                                    replace_index=new_replace_index,
                                                                    seed=new_seed,
                                                                    t=t,
                                                                    metric_to_report=loss_metric,
                                                                    verbose=False,
                                                                    diffinit=diffinit)

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
    a_seeds = []
    rs = []
    for a_seed in attack_seeds:
        sub_df = df.loc[df['attack_seed'] == a_seed]
        # Need the same th_ri as well
        rs_for_this_seed = sub_df['attack_ri'].unique()
        for r in rs_for_this_seed:
            a_seeds.append(a_seed)
            rs.append(r)
            sub_sub_df = sub_df.loc[sub_df['attack_ri'] == r]
            # Same Fix
            same_fix = sub_sub_df.loc[(sub_sub_df['th_seed'] == a_seed) & (~sub_sub_df['diffinit'])]
            try:
                accuracy_same_fix.append(same_fix['attack_acc_with_same_ri'].iloc[0])
                epsilon_same_fix.append(same_fix['epsilon_with_same_ri'].iloc[0])
            except IndexError:
                accuracy_same_fix.append(np.nan)
                epsilon_same_fix.append(np.nan)
            # Diff Fix
            diff_fix = sub_sub_df.loc[(sub_sub_df['th_seed'] != a_seed) & (~sub_sub_df['diffinit'])]
            try:
                accuracy_diff_fix.append(diff_fix['attack_acc_with_same_ri'].iloc[0])
                epsilon_diff_fix.append(diff_fix['epsilon_with_same_ri'].iloc[0])
            except IndexError:
                accuracy_diff_fix.append(np.nan)
                epsilon_diff_fix.append(np.nan)
            # Same Diff
            same_diff = sub_sub_df.loc[(sub_sub_df['th_seed'] == a_seed) & (sub_sub_df['diffinit'])]
            try:
                accuracy_same_diff.append(same_diff['attack_acc_with_same_ri'].iloc[0])
                epsilon_same_diff.append(same_diff['epsilon_with_same_ri'].iloc[0])
            except IndexError:
                accuracy_same_diff.append(np.nan)
                epsilon_same_diff.append(np.nan)
            # Diff Diff
            diff_diff = sub_sub_df.loc[(sub_sub_df['th_seed'] != a_seed) & (sub_sub_df['diffinit'])]
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
                            'epsilon_diff_diff': epsilon_diff_diff,
                            'replace_index': rs}, index=a_seeds)

    # Questions:

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
    parser.add_argument('--cfg', type=str, help='Name of yaml cfg of experiment', default='cifar10_cnn')
    parser.add_argument('--exptype', type=str, choices=['fsfi', 'vsfi', 'fsvi', 'vsvi'])
    parser.add_argument('--runs', type=int, default=5, help='Number of times to repeat an experiment')
    parser.add_argument('--output', type=str, default='all_cifar10_cnn_results_8000.csv',
                        help='Log file to store all the results')
    parser.add_argument('--mi_type', type=str, default='loss', choices=['loss', 'intermediate', 'both'],
                        help='MI attack on loss, or intermediate activations?')
    parser.add_argument('--t', type=int, default=None, help='Numer of iterations')
    args = parser.parse_args()
    cfg = load_cfg(args.cfg)

    if args.mi_type == 'both':
        use_loss = True
        use_activations = True
    elif args.mi_type == 'loss':
        use_loss = True
        use_activations = False
    elif args.mi_type == 'intermediate':
        use_loss = False
        use_activations = True
    run_mi_attack_steph(cfg, args.exptype, args.t, args.runs, args.output,
                        use_loss=use_loss, use_activations=use_activations)
