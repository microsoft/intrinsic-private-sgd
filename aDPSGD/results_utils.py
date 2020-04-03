#!/usr/bin/env ipython
## author: stephanie hyland
## purpose: Scripts for manipulating experimental results (e.g. mostly loading!)

import numpy as np
import pandas as pd
import ipdb
import glob
import os

def trace_path_stub(dataset, model, replace_index=None, seed=None, diffinit=False, data_privacy='all'):
    path_stub = './traces/' + dataset + '/' + data_privacy + '/' + model + '/' + model
    if diffinit:
        path_stub = path_stub + '_DIFFINIT'
    if not replace_index is None:
        path_stub = path_stub + '.replace_' + str(replace_index)
    if not seed is None:
        path_stub = path_stub + '.seed_' + str(seed)
    return path_stub

def get_list_of_params(dataset, identifier):
    model, replace_index, seed = identifier
    weights = load_weights(dataset, model, replace_index, seed, iter_range=(None, 5))
    params = weights.columns[1:]
    return params

def get_available_results(dataset, model, replace_index=None, seed=None, diffinit=False, data_privacy='all'):
    available_results = ['.'.join(x.split('/')[-1].split('.')[:-2]) for x in glob.glob(trace_path_stub(dataset, model, diffinit=diffinit, data_privacy=data_privacy) + '.*.weights.csv')]
    # remove replace_NA!
    available_results = [x for x in available_results if not 'replace_NA' in x]
    # collect all the results 
    drop_and_replace_and_seeds = [(x.split('.')[1].split('_')[1], x.split('.')[2].split('_')[1], x.split('.')[3].split('_')[1]) for x in available_results]
    drop, replace, seeds = zip(*drop_and_replace_and_seeds)
    df = pd.DataFrame({'drop': drop, 'replace': replace, 'seed': seeds})
    if not replace_index is None:
        df = df.loc[df['replace'] == replace_index, :]
    if not seed is None:
        df = df.loc[df['seed'] == seed, :]
    return df

def check_if_experiment_exists(dataset, model, replace_index, seed, diffinit, data_privacy='all'):
    path = trace_path_stub(dataset, model, replace_index, seed, diffinit, data_privacy) + '.weights.csv'
    exists = os.path.exists(path)
    if not exists:
        logpath = 'missing_experiments.' + dataset + '.' + data_privacy + '.' + model + '.csv'
        if not os.path.exists(logpath):
            logfile = open(logpath, 'w')
            logfile.write('replace,seed,diffinit\n')
        else:
            logfile = open(logpath, 'a')
        logfile.write(str(replace_index) + ',' + str(seed) + ',' + str(diffinit) + '\n')
        logfile.close()
    return exists

def get_posterior_samples(dataset, iter_range, model='linear', replace_index=None, params=None, seeds='all', n_seeds=None, verbose=True, diffinit=False, data_privacy='all'):
    """
    grab the values of the weights of [params] at [at_time] for all the available seeds from identifier_stub
    might want to re-integrate this with sacred at some point
    """
    if seeds == 'all':
        df = get_available_results(dataset, model, replace_index=replace_index, diffinit=diffinit, data_privacy=data_privacy)
        available_seeds = df['seed'].unique().tolist()
    else:
        assert type(seeds) == list
        available_seeds = seeds
    if not n_seeds is None:
        available_seeds = np.random.choice(available_seeds, n_seeds, replace=False)
    S = len(available_seeds) 
    if verbose:
        print('Loading samples from seeds:', available_seeds, 'in range', iter_range)
    samples = []
    for i, s in enumerate(available_seeds):
        weights_from_s = load_weights(dataset, model, replace_index=replace_index, seed=s, iter_range=iter_range, params=params, verbose=False, diffinit=diffinit, data_privacy=data_privacy)
        try:
            if weights_from_s.shape[0] == 0:
                print('WARNING: No data from seed', s, 'in range', iter_range, ' - skipping')
            else:
                # insert the seed (the format should be similar to when we load gradient noise)
                weights_from_s.insert(loc=1, column='seed', value=s)
                samples.append(weights_from_s)
        except AttributeError:
            print('WARNING: No data from seed', s, 'in range', iter_range, 'or something, not sure why this error happened? - skipping')
            ipdb.set_trace()
    if len(samples) > 1:
        samples = pd.concat(samples)
    else:
        print('[get_posterior_samples] WARNING: No actual samples acquired for replace', replace_index, '!')
        samples = False
    return samples

def load_gradients(dataset, model, replace_index, seed, noise=False, iter_range=(None, None), params=None, verbose=False, diffinit=False):
    path_stub = trace_path_stub(dataset, model, replace_index, seed, diffinit=diffinit)
    path = path_stub + '.all_gradients.csv' 
    if not params is None:
        assert type(params) == list
        usecols = ['t', 'minibatch_id'] + params
        if verbose:
            print('Loading gradients with columns:', usecols)
    else:
        if verbose:
            print('WARNING: Loading all columns can be slow!')
        usecols = None
    df = pd.read_csv(path, skiprows=1, usecols=usecols, dtype={'t': np.int64, 'minibatch_id': str})
    
    # remove validation data by default
    df = df.loc[~(df['minibatch_id'] == 'VALI'), :]
    
    if iter_range[0] is not None:
        df = df.loc[df['t'] >= iter_range[0], :]
    if iter_range[1] is not None:
        df = df.loc[df['t'] <= iter_range[1], :]
    
    if noise:
        # separate minibatches from aggregate
        df_minibatch = df.loc[~(df['minibatch_id'] == 'ALL'), :]
        if df_minibatch.shape[0] == 0:
            print('[load_gradients] WARNING: No minibatch information. Try turning off calculation of gradient noise')
        df_all = df.loc[df['minibatch_id'] == 'ALL', :]
        df_minibatch.set_index(['t', 'minibatch_id'], inplace=True)
        df_all = df_all.set_index('t').drop('minibatch_id', axis=1)
        df = df_minibatch - df_all
        df.reset_index(inplace=True)
    return df

def load_weights(dataset, model, replace_index, seed, diffinit=False, 
        iter_range=(None, None), params=None, verbose=True, data_privacy='all'):
    path_stub = trace_path_stub(dataset, model, replace_index, seed, diffinit=diffinit, data_privacy=data_privacy)
    path = path_stub + '.weights.csv' 
    if not params is None:
        assert type(params) == list
        usecols = ['t'] + params
    else:
        if verbose: print('WARNING: Loading all columns can be slow!')
        usecols = None
    
    df = pd.read_csv(path, skiprows=1, usecols=usecols)
    
    if iter_range[0] is not None:
        df = df.loc[df['t'] >= iter_range[0], :]
    if iter_range[1] is not None:
        df = df.loc[df['t'] <= iter_range[1], :]
    
    if verbose: print('Loaded weights from', path)
    return df

def load_loss(dataset, model, replace_index, seed, iter_range=(None, None), diffinit=False, verbose=False, data_privacy='all'):
    path_stub = trace_path_stub(dataset, model, replace_index, seed, diffinit=diffinit, data_privacy=data_privacy)
    path = path_stub + '.loss.csv' 
    df = pd.read_csv(path, skiprows=1)
    
    if iter_range[0] is not None:
        df = df.loc[df['t'] >= iter_range[0], :]
    if iter_range[1] is not None:
        df = df.loc[df['t'] <= iter_range[1], :]
    return df
