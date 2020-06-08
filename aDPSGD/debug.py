#!/usr/bin/env ipython

import numpy as np
import pandas as pd
import glob
import ipdb

import eval_utils

def run_checks(dataset, model, diffinit, data_privacy='all'):
    """

    """
    fail = False
    # look for missing experiments
    # get the convergence point
    if dataset in ['mnist', 'cifar10', 'mnist_square']:
        metric = 'ce'
    else:
        metric = 'binary_crossentropy'
    print('Computing convergence point...')
    convergence_point, _ = eval_utils.find_convergence_point(dataset, model, diffinit, tolerance=3, metric=metric, data_privacy=data_privacy)
    print('convergence point:', convergence_point)
    print('Checking for incomplete experiments...')
    incomp = check_for_incomplete_experiments(dataset, model, t=convergence_point, diffinit=diffinit, data_privacy=data_privacy)
    if incomp is True:
        print('[debug] Passed check for incomplete expeirments')
    else:
        empty, unsure, incomplete = incomp
        print('[debug] Failed check for incomplete expeirments')
        fail = True
        print(incomplete)
    # make sure the same seed always has the same initialisation
    print('Checking for initialisation violations...')
    init_violations = check_for_different_initialisations_with_same_seed(dataset, model, diffinit=diffinit)
    if init_violations is True:
        print('[debug] Passed check for different initialisations')
    else:
        print('[debug] Failed check for different initialisations')
        fail = True
    if fail:
        result = 'Fail'
    else:
        result = 'Pass'
    return result

def check_for_incomplete_experiments(dataset, model, t=5, diffinit=True, data_privacy='all'):
    """
    find experiments where data does not reach time t
    if t is None, we're just looking for experiments where the file is empty
    """
    exp_df = eval_utils.get_available_results(dataset, model, diffinit=diffinit, data_privacy=data_privacy)
    print('Found', exp_df.shape[0], 'experiments!')
    empty = []
    unsure = []
    incomplete = []
    for i, row in exp_df.iterrows():
        drop_index = 'NA'
        replace_index = row['replace']
        seed = row['seed']
        try:
            loss = eval_utils.load_loss(dataset, model, drop_index, replace_index, seed, iter_range=(None, None), diffinit=diffinit, verbose=False, data_privacy=data_privacy)
            if np.nanmax(loss['t']) < t:
                incomplete.append((replace_index, seed))
        except:
            print('issue with', row)
            path = eval_utils.trace_path_stub(dataset, model, drop_index, replace_index, seed, diffinit=diffinit, data_privacy=data_privacy)
            ll = open(path + '.weights.csv').readlines()
            if len(ll) <= 3:
                empty.append((replace_index, seed))
            else:
                print('Cant load loss but file is not empty?...')
                unsure.append((replace_index, seed))
    if len(empty) == 0 and len(unsure) == 0 and len(incomplete) == 0:
        print('Found no issues')
        return True
    else:
        print('Found', len(empty), 'empty experiments')
        print('Found', len(unsure), 'unsure experiments')
        print('Found', len(incomplete), 'incomplete experiments')
        return empty, unsure, incomplete

def find_mismatch_loss_test(dataset, model, diffinit=False, t=2000, metric='accuracy'):
    """
    find models where the apparent performance disagrees with what the trace file says...
    """
    exp_df = eval_utils.get_available_results(dataset, model, diffinit=diffinit)
    print('Found', exp_df.shape[0], 'experiments to test...!')
    good_settings = []
    for i, row in exp_df.iterrows():
        replace = row['replace']
        seed = row['seed']
        performance = eval_utils.debug_just_test(dataset, model, drop_index='NA', replace_index=replace, seed=seed, t=t, diffinit=diffinit, use_vali=True)
        retest_performance = performance[metric]
        loss = eval_utils.load_loss(dataset, model, drop_index='NA', replace_index=replace, seed=seed, iter_range=(t, t+1), diffinit=diffinit)
        orig_performance = loss.loc[loss['minibatch_id'] == 'VALI', metric].values[0]
        discrepancy = np.abs(retest_performance - orig_performance)
        print('\t\t\t',discrepancy)
        if discrepancy < 0.1:
            good_settings.append((replace, seed))
        else:
            print('\t\tBAD EXPERIMENT FOUND!')
            ipdb.set_trace()
        if i % 100 == 0:
            print('\t[DEBUG] We have gone through', i, 'experiments and found', len(good_settings), 'good ones!')
    print('Found', len(good_settings), 'good settings!')
    print('that\'s', 100*len(good_settings)/df.shape[0], 'percent!')
    return good_settings

def check_for_different_initialisations_with_same_seed(dataset, model, diffinit=True):
    """
    same seed should always imply same initialisation
    """
    files = glob.glob('./traces/' + dataset + '/all/' + model + '/' + model + '_DIFFINIT'*diffinit + '.*.weights.csv')
    files = np.random.permutation(files)
    if diffinit:
        seeds = [f.split('/')[-1].split('.')[3].split('_')[1] for f in files]
        seeds = set(seeds)
        print('Found seeds:', seeds)
    else:
        seeds = set(['any'])
    seed_weights = dict()
    violations = []
    for f in files:
        identifier = f.split('/')[-1]
        if 'decoy' in identifier:
            print('DECOY FOUND')
        seed_identifier = identifier.split('.')[3].split('_')[1]
        if diffinit:
            seed = seed_identifier
        else:
            seed = 'any'
        assert seed in seeds
        try:
            #starting weights
            weights = pd.read_csv(f, skiprows=1, nrows=1).values[0, 1:]
        except:
            print('WARNING: no data for identifier', identifier, '- skipping')
            continue
        if seed in seed_weights:
            known_weights = seed_weights[seed][1]
            for i, w in enumerate(known_weights):
                if np.array_equal(weights, w):
                    seed_weights[seed][0][i].add(identifier)
                    # stop iterating
                    break
            else:
                print('WARNING! Found new initial setting in experiment', identifier, 'for seed', seed_identifier)
                violations.append(set([identifier]))
                seed_weights[seed][0].append(set([identifier]))
                seed_weights[seed][1].append(weights)
        else:
            print('First instance of weights for seed', seed_identifier, 'in experiment', identifier)
            seed_weights[seed] = ([set([identifier])], [weights])
    if len(violations) > 0:
        print('Violations found')
        return violations
    else:
        print('all experiments with the same seed have the same initialisation')
        return True
