#!/usr/bin/env ipython
# author: stephanie hyland
# purpose: Scripts for manipulating experimental results (e.g. mostly loading!)

import numpy as np
import pandas as pd
from pathlib import Path
from experiment_metadata import get_dataset_size, get_input_hidden_size
import ipdb

TRACES_DIR = './traces/'


class ExperimentIdentifier(object):
    def __init__(self, cfg_name=None, model=None, replace_index=None, seed=None,
                 diffinit=True, data_privacy='all', traces_dir=TRACES_DIR):
        self.cfg_name = cfg_name
        self.model = model
        self.replace_index = replace_index
        self.seed = seed
        self.diffinit = diffinit
        self.data_privacy = data_privacy
        self.traces_dir = Path(traces_dir)

    def init_from_cfg(self, cfg):
        self.cfg_name = cfg['cfg_name']
        self.model = cfg['model']['architecture']

    def ensure_directory_exists(self, verbose=True):
        path_stub = self.path_stub()
        folder = path_stub.parent
        if folder.exists():
            if verbose:
                print(f'Path {folder} already exists')
        else:
            folder.mkdir(parents=True)
            if verbose:
                print(f'Created {folder}')

    def path_stub(self):
        path_stub = self.traces_dir / self.cfg_name

        identifier = self.model

        if self.diffinit:
            identifier = f'{identifier}_DIFFINIT'

        if self.replace_index is None:
            identifier = f'{identifier}.replace_NA'
        else:
            identifier = f'{identifier}.replace_{self.replace_index}'

        if self.seed is None:
            identifier = f'{identifier}.seed_NA'
        else:
            identifier = f'{identifier}.seed_{self.seed}'

        path_stub = path_stub / identifier

        return path_stub.resolve()

    def derived_path_stub(self):
        """ This is where derived results go """
        derived_path = self.path_stub().parent / 'derived'
        if not derived_path.exists():
            print(f'Creating path {derived_path}')
            derived_path.mkdir()
        return derived_path

    def exists(self, log_missing=False):
        path = self.path_stub().with_name(self.path_stub().name + '.weights.csv')
        results_exist = path.exists()

        if log_missing and not results_exist:
            logpath = Path(f'missing_experiments_{self.cfg_name}_{self.data_privacy}_{self.model}.csv')

            if not logpath.exists():
                logfile = open(logpath, 'w')
                logfile.write('replace,seed,diffinit\n')
            else:
                logfile = open(logpath, 'a')
            logfile.write(f'{self.replace_index},{self.seed},{self.diffinit}\n')
            logfile.close()

        return results_exist

    def load_gradients(self, noise=False, iter_range=(None, None), params=None, verbose=False) -> pd.DataFrame:
        path = self.path_stub().with_name(self.path_stub().name + '.grads.csv')

        if params is not None:
            assert type(params) == list
            usecols = ['t', 'minibatch_id'] + params

            if verbose:
                print('Loading gradients with columns:', usecols)
        else:
            if verbose:
                print('WARNING: Loading all columns can be slow!')
            usecols = None
        df = pd.read_csv(path, usecols=usecols, dtype={'t': np.int64, 'minibatch_id': str})

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

    def load_weights(self, iter_range=(None, None), params=None,
                     verbose=True, sort=False) -> pd.DataFrame:
        path = self.path_stub().with_name(self.path_stub().name + '.weights.csv')

        df = pd.read_csv(path, usecols=None)

        if iter_range[0] is not None:
            df = df.loc[df['t'] >= iter_range[0], :]

        if iter_range[1] is not None:
            df = df.loc[df['t'] <= iter_range[1], :]

        if sort:
            df = self.sort_weights(df)

        if params is not None:
            assert type(params) == list
            params_cols = ['t'] + params
            df = df[params_cols]

        if verbose:
            if sort:
                print(f'Loaded and sorted weights from {path}')
            else:
                print(f'Loaded weights from {path}')

        return df

    def sort_weights(self, df) -> pd.DataFrame:
        """ Sorting all the weights """
        if not self.model == 'mlp':
            print('WARNING: Weight sorting is only meaningful for MLP - doing nothing')
            return df
        weights = df.iloc[:, 1:].values
        N = df.shape[0]
        # Set up the weights
        input_size, hidden_size = get_input_hidden_size(self.cfg_name)
        dense_layer_shape = (input_size, hidden_size)
        shape_of_weights = [dense_layer_shape, [hidden_size], [hidden_size], [1]]
        try:
            assert np.sum([np.product(x) for x in shape_of_weights]) == weights.shape[1]
        except AssertionError:
            ipdb.set_trace()
        # Split it up (copied from unflattening the weights)
        indicator = 0
        list_of_weights = []
        for shape_size in shape_of_weights:
            weight_size = np.product(shape_size)
            weight_values = weights[:, indicator:(indicator+weight_size)]
            list_of_weights.append(weight_values)
            indicator = indicator + weight_size
        dense_layer = list_of_weights[0].reshape(N,
                                                 dense_layer_shape[0],
                                                 dense_layer_shape[1])
        dense_bias = list_of_weights[1]
        final_layer = list_of_weights[2]
        # Check things
        assert dense_bias.shape[1] == dense_layer.shape[2]
        assert dense_bias.shape[1] == final_layer.shape[1]
        # Sort by the final layer
        sort_idx = np.argsort(final_layer, axis=1)
        dense_layer_sorted = np.take_along_axis(dense_layer,
                                                sort_idx.reshape(N, 1, hidden_size),
                                                axis=2)
        dense_bias_sorted = np.take_along_axis(dense_bias, sort_idx, axis=1)
        final_layer_sorted = np.take_along_axis(final_layer, sort_idx, axis=1)
        # Flatten dense layer again
        dense_layer_sorted = dense_layer_sorted.reshape(N, -1)
        # Reinsert into the weights
        weights = np.hstack([dense_layer_sorted, dense_bias_sorted,
                             final_layer_sorted, list_of_weights[-1]])
        df.iloc[:, 1:] = weights
        return df

    def load_loss(self, iter_range=(None, None), verbose=False):
        path = self.path_stub().with_name(self.path_stub().name + '.loss.csv')

        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            raise ValueError(path, f'{path} appears to be empty?')

        if iter_range[0] is not None:
            df = df.loc[df['t'] >= iter_range[0], :]

        if iter_range[1] is not None:
            df = df.loc[df['t'] <= iter_range[1], :]

        return df


def get_grid_to_run(cfg, num_seeds, num_replaces):
    exp = ExperimentIdentifier()
    exp.init_from_cfg(cfg)
    grid_path = exp.derived_path_stub() / 'grid.csv'
    try:
        grid = pd.read_csv(grid_path)
        print(f'Loaded grid seeds and replace indices from {grid_path}')
        seeds = grid[grid['what'] == 'seed']['value']
        replaces = grid[grid['what'] == 'replace_index']['value']
    except FileNotFoundError:
        seeds = []
        replaces = []
    if len(seeds) < num_seeds:
        candidate_seeds = [x for x in range(99999) if x not in seeds]
        new_seeds = np.random.choice(candidate_seeds, num_seeds - len(seeds), replace=False)
        seeds = np.concatenate([seeds, new_seeds])
    if len(replaces) < num_replaces:
        N = get_dataset_size(cfg['data'])
        candidate_replaces = [x for x in range(N) if x not in replaces]
        new_replaces = np.random.choice(candidate_replaces, num_replaces - len(replaces))
        replaces = np.concatenate([replaces, new_replaces])
    seeds = np.int32(seeds)
    replaces = np.int32(replaces)
    what = ['seed']*len(seeds) + ['replace_index']*len(replaces)
    values = np.concatenate([seeds, replaces])
    grid = pd.DataFrame({'value': values, 'what': what})
    grid.to_csv(grid_path, index=False)
    print(f'Saved grid seeds and replace indices to {grid_path}')
    return seeds, replaces


def get_replace_index_with_most_seeds(cfg_name: str, model: str, diffinit: bool = False) -> int:
    df = get_available_results(cfg_name, model, replace_index=None, seed=None, diffinit=diffinit)
    seeds_per_replace = df['replace'].value_counts()
    replace_with_most_seeds = seeds_per_replace.idxmax()
    print(f'Selecting replace index {replace_with_most_seeds}, which has {seeds_per_replace[replace_with_most_seeds]} seeds')
    return replace_with_most_seeds


def get_available_results(cfg_name: str, model: str, replace_index: int = None, seed: int = None,
                          diffinit: bool = False, data_privacy: str = 'all') -> pd.DataFrame:

    sample_experiment = ExperimentIdentifier(cfg_name=cfg_name, model=model, replace_index=1,
                                             seed=1, data_privacy=data_privacy, diffinit=diffinit)
    directory_path = Path(sample_experiment.path_stub()).parent
    files_in_directory = directory_path.glob('*.weights.csv')
    replaces = []
    seeds = []

    for f in files_in_directory:
        split_name = f.name.split('.')
        assert model in split_name[0], 'Inconsistency detected in file path'

        # which replace index?
        replace_piece = split_name[1]
        replaces.append(int(replace_piece.split('_')[1]))

        # which seed?
        seed_piece = split_name[2]
        seeds.append(int(seed_piece.split('_')[1]))
    df = pd.DataFrame({'replace': replaces, 'seed': seeds})

    if replace_index is not None:
        df = df.loc[df['replace'] == replace_index, :]

    if seed is not None:
        df = df.loc[df['seed'] == seed, :]

    return df


def get_posterior_samples(cfg_name, iter_range, model='logistic', replace_index=None,
                          params=None, seeds='all', num_seeds='max', verbose=True,
                          diffinit=False, data_privacy='all', what='weights', sort=False):
    """
    grab the values of the weights of [params] at [at_time] for all the available seeds from identifier_stub
    """

    if seeds == 'all':
        df = get_available_results(cfg_name, model, replace_index=replace_index,
                                   diffinit=diffinit, data_privacy=data_privacy)
        available_seeds = df['seed'].unique().tolist()
    else:
        assert type(seeds) == list
        available_seeds = seeds
    if not num_seeds == 'max' and num_seeds < len(available_seeds):
        available_seeds = np.random.choice(available_seeds, num_seeds, replace=False)

    if replace_index is None:
        replace_index = get_replace_index_with_most_seeds(cfg_name=cfg_name, model=model, diffinit=diffinit)

    if verbose:
        print(f'Loading {what} from seeds: {available_seeds} in range {iter_range}')
    samples = []

    base_experiment = ExperimentIdentifier(cfg_name, model, replace_index, diffinit=diffinit, data_privacy=data_privacy)

    for i, s in enumerate(available_seeds):
        base_experiment.seed = s
        if not base_experiment.exists():
            print(f'WARNIG: File {base_experiment.path_stub()} doesn\'t seem to exist -skipping')
            continue
        if what == 'weights':
            data_from_s = base_experiment.load_weights(iter_range=iter_range,
                                                       params=params, verbose=False,
                                                       sort=sort)
        elif what == 'gradients':
            data_from_s = base_experiment.load_gradients(iter_range=iter_range, params=params, verbose=False)
        else:
            raise ValueError
        try:
            if data_from_s.shape[0] == 0:
                print('WARNING: No data from seed', s, 'in range', iter_range, ' - skipping')
            else:
                # insert the seed (the format should be similar to when we load gradient noise)
                data_from_s.insert(loc=1, column='seed', value=s)
                samples.append(data_from_s)
        except AttributeError:
            print(f'WARNING: No data from seed {s} in range {iter_range} or something, not sure why this error happened? - skipping')

    if len(samples) > 1:
        samples = pd.concat(samples)
    else:
        print(f'[get_posterior_samples] WARNING: No actual samples acquired for replace {replace_index}!')
        samples = False

    if what == 'gradients':
        # remove reference to minibatchs samples
        samples = samples[samples['minibatch_id'].str.contains('minibatch_sample')].drop(columns='minibatch_id')
    return samples


def get_posterior_from_all_datasets(cfg_name, iter_range, model='logistic',
                                    params=None, seeds='all', num_seeds='max', verbose=True,
                                    diffinit=False, data_privacy='all',
                                    what='weights', sort=False) -> pd.DataFrame:
    """
    This loops over get_posterior, for all replace indices
    """
    results = get_available_results(cfg_name, model, replace_index=None, seed=None, diffinit=diffinit)
    replace_indices = results['replace'].unique()
    all_dfs = []
    for replace_index in replace_indices:
        print(f'Replace index: {replace_index}')
        replace_samples = get_posterior_samples(cfg_name, iter_range, model=model,
                                                replace_index=replace_index,
                                                params=params, seeds=seeds,
                                                num_seeds=num_seeds, verbose=verbose,
                                                diffinit=diffinit,
                                                data_privacy=data_privacy, what=what, sort=sort)
        if replace_samples is not False:
            replace_samples['replace_index'] = replace_index
            all_dfs.append(replace_samples)
    df = pd.concat(all_dfs)
    return df
