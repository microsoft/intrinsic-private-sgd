#!/usr/bin/env ipython
import numpy as np
import ipdb

import derived_results
import results_utils
from results_utils import ExperimentIdentifier


def run_checks(cfg_name, model, diffinit, data_privacy='all', convergence_point=None):
    fail = False
    if convergence_point is None:
        print('Computing convergence point...')
        metric = 'binary_crossentropy'          # TODO make work for multiclass
        convergence_point, _ = derived_results.find_convergence_point(cfg_name, model, diffinit, tolerance=3, metric=metric, data_privacy=data_privacy)
    print('convergence point:', convergence_point)
    print('Checking for incomplete experiments...')
    incomplete = check_for_incomplete_experiments(cfg_name, model, t=convergence_point, diffinit=diffinit, data_privacy=data_privacy)
    if incomplete is True:
        print('[debug] Passed check for incomplete expeirments')
    else:
        print('[debug] Failed check for incomplete expeirments')
        fail = True
        print(incomplete)
    # make sure the same seed always has the same initialisation
    print('Checking for initialisation violations...')
    init_violations = check_for_different_initialisations_with_same_seed(cfg_name, model, diffinit=diffinit)
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

def check_for_incomplete_experiments(cfg_name, model, t=5, diffinit=True, data_privacy='all'):
    """
    find experiments where data does not reach time t
    if t is None, we're just looking for experiments where the file is empty
    """
    exp_df = results_utils.get_available_results(cfg_name, model, diffinit=diffinit, data_privacy=data_privacy)
    print('Found', exp_df.shape[0], 'experiments!')
    incomplete = []
    for i, row in exp_df.iterrows():
        replace_index = row['replace']
        seed = row['seed']
        exp =  ExperimentIdentifier(cfg_name, model, replace_index=replace_index, seed=seed, diffinit=diffinit, data_privacy=data_privacy)
        if not exp.exists():
            print(f'WARNING: Experiment {exp.path_stub()} doesn\'t exist?')
            continue
        loss = exp.load_loss(verbose=False)
        if np.nanmax(loss['t']) < t:
            incomplete.append((replace_index, seed))
    if len(incomplete) == 0:
        print('Found no issues')
        return True
    else:
        print('Found', len(incomplete), 'incomplete experiments')
        return incomplete


def check_for_different_initialisations_with_same_seed(cfg_name, model, diffinit=True):
    """
    same seed should always imply same initialisation
    """
    exp_df = results_utils.get_available_results(cfg_name, model, diffinit=diffinit, data_privacy='all')
    exp_df = exp_df.iloc[np.random.permutation(exp_df.shape[0]), :]
    if diffinit:
        seeds = set(exp_df['seed'].unique())
        print('Found seeds:', seeds)
    else:
        seeds = set(['any'])
    seed_weights = dict()
    violations = []
    for i, row in exp_df.iterrows():
        replace_index = row['replace']
        seed_identifier = row['seed']
        exp = ExperimentIdentifier(cfg_name, model, replace_index=replace_index,
                                   seed=seed_identifier, diffinit=diffinit,
                                   data_privacy='all')
        identifier = exp.path_stub()
        if not exp.exists():
            print(f'WARNING: Experiment {identifier} doesn\'t exist - skipping')
            continue
        if diffinit:
            seed = seed_identifier
        else:
            seed = 'any'
        try:
            assert seed in seeds
        except AssertionError:
            ipdb.set_trace()
        # only care about starting weights
        weights = exp.load_weights(iter_range=(0, 0), verbose=False)
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
