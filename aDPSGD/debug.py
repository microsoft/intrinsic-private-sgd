#!/usr/bin/env ipython
import numpy as np
import ipdb

import matplotlib.pyplot as plt
import seaborn as sns

import derived_results
import results_utils
from results_utils import ExperimentIdentifier

plt.style.use('ggplot')


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
        exp = ExperimentIdentifier(cfg_name, model, replace_index=replace_index, seed=seed, diffinit=diffinit, data_privacy=data_privacy)
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
        weights = exp.load_weights(iter_range=(0, 0), verbose=False, sort=False)
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


def compare_loss_with_without_diffinit(cfg_name: str, model: str, t: int = 2000):
    df_diffinit = results_utils.get_available_results(cfg_name, model, diffinit=True, data_privacy='all')
    df_fixinit = results_utils.get_available_results(cfg_name, model, diffinit=False, data_privacy='all')

    loss_diff = []
    loss_fix = []

    if 'cifar10' in cfg_name:
        metric = 'ce'
    else:
        metric = 'binary_crossentropy'
    for idx, row in df_diffinit.iterrows():
        replace_index = row['replace']
        seed = row['seed']
        diffinit = True
        exp = ExperimentIdentifier(cfg_name, model, replace_index=replace_index,
                                   seed=seed, diffinit=diffinit,
                                   data_privacy='all')
        loss = exp.load_loss(iter_range=(t, t+1), verbose=False)
        try:
            loss = float(loss.loc[loss['minibatch_id'] == 'VALI'][metric])
            loss_diff.append(loss)
        except ValueError:
            print(f'skipping {row} due to can\'t convert to float')
    for idx, row in df_fixinit.iterrows():
        replace_index = row['replace']
        seed = row['seed']
        diffinit = False
        exp = ExperimentIdentifier(cfg_name, model, replace_index=replace_index,
                                   seed=seed, diffinit=diffinit,
                                   data_privacy='all')
        loss = exp.load_loss(iter_range=(t, t+1), verbose=False)
        try:
            loss = float(loss.loc[loss['minibatch_id'] == 'VALI'][metric])
            loss_fix.append(loss)
        except ValueError:
            print(f'skipping {row} due to can\'t convert to float')

    fig, axarr = plt.subplots(nrows=1, ncols=1)
    sns.distplot(loss_fix, label='fixed init', ax=axarr)
    sns.distplot(loss_diff, label='diff init', ax=axarr)
    axarr.set_xlabel('loss')
    axarr.legend()
    plt.savefig(f'{cfg_name}_losses.png')
    plt.clf()
    plt.close()


def compare_learning_curves(cfg_name: str, model: str):
    agg_diff = derived_results.AggregatedLoss(cfg_name, model, 'all').load(diffinit=True, generate_if_needed=True)
    agg_fix = derived_results.AggregatedLoss(cfg_name, model, 'all').load(diffinit=False, generate_if_needed=True)

    if 'cifar10' in cfg_name:
        metric = 'ce'
    else:
        metric = 'binary_crossentropy'
    print(agg_diff.head())
    fig, axarr = plt.subplots(nrows=2, ncols=1)
    axarr[0].plot(agg_diff.index, agg_diff[f'{metric}_mean_train'], label='diff init', color='blue')
    axarr[0].fill_between(agg_diff.index, agg_diff[f'{metric}_mean_train'] - agg_diff[f'{metric}_std_train'], agg_diff[f'{metric}_mean_train'] + agg_diff[f'{metric}_std_train'], alpha=0.1, color='blue', label='_nolegend_')
    axarr[0].plot(agg_fix.index, agg_fix[f'{metric}_mean_train'], label='fixed init', color='green')
    axarr[0].fill_between(agg_fix.index, agg_fix[f'{metric}_mean_train'] - agg_fix[f'{metric}_std_train'], agg_fix[f'{metric}_mean_train'] + agg_fix[f'{metric}_std_train'], alpha=0.1, color='lightgreen', label='_nolegend_')


    axarr[1].plot(agg_diff.index, agg_diff[f'{metric}_mean_vali'], label='diff init', color='blue', linestyle='--')
    axarr[1].fill_between(agg_diff.index, agg_diff[f'{metric}_mean_vali'] - agg_diff[f'{metric}_std_vali'], agg_diff[f'{metric}_mean_vali'] + agg_diff[f'{metric}_std_vali'], alpha=0.1, color='blue', label='_nolegend_')
    axarr[1].plot(agg_fix.index, agg_fix[f'{metric}_mean_vali'], label='fix init', color='green', linestyle='--')
    axarr[1].fill_between(agg_fix.index, agg_fix[f'{metric}_mean_vali'] - agg_fix[f'{metric}_std_vali'], agg_fix[f'{metric}_mean_vali'] + agg_fix[f'{metric}_std_vali'], alpha=0.1, color='green', label='_nolegend_')

    axarr[0].set_ylabel('ce train')
    axarr[1].set_ylabel('ce vali')
    for ax in axarr:
        ax.legend()

    plt.savefig(f'{cfg_name}_learning_curves.png')
    plt.clf()
    plt.close()
