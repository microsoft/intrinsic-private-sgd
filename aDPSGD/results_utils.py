#!/usr/bin/env ipython
# author: stephanie hyland
# purpose: Scripts for manipulating experimental results (e.g. mostly loading!) import numpy as np import pandas as pd
from pathlib import Path
from experiment_metadata import get_input_hidden_size, lr_convergence_points, get_experiment_details
from noise_utils import compute_wu_bound
from typing import Tuple
import numpy as np
import pandas as pd
# from stats_utils import estimate_statistics_through_training
import ipdb

TRACES_DIR = './traces/'


def define_output_perturbation_scale(cfg_name: str, target_epsilon=1) -> float:
    """ hack!!! """
    t = lr_convergence_points[cfg_name]
    # Must be LR
    assert '_lr' in cfg_name
    _, batch_size, lr, _, N = get_experiment_details(cfg_name, 'logistic')
    lipschitz_constant = np.sqrt(2)
    sensitivity = compute_wu_bound(lipschitz_constant, t=t, N=N, batch_size=batch_size, eta=lr, verbose=False)
    delta = 1 / (N ** 2)
    c = np.sqrt(2 * np.log(1.25 / delta)) + 1e-6
    sigma = c * (sensitivity / target_epsilon)
    return sigma


class ExperimentIdentifier(object):
    def __init__(self, cfg_name=None, model=None, replace_index=None, seed=None,
                 diffinit=True, data_privacy='all', traces_dir=TRACES_DIR,
                 do_output_perturbation: bool = False):
        self.cfg_name = cfg_name
        self.model = model
        self.replace_index = replace_index
        self.seed = seed
        self.diffinit = diffinit
        self.data_privacy = data_privacy
        self.traces_dir = Path(traces_dir)
        self.do_output_perturbation = do_output_perturbation
        if self.do_output_perturbation:
            self.output_perturbation_scale = define_output_perturbation_scale(self.cfg_name)
            print(f'Using output perturbation scale of {self.output_perturbation_scale}')

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

        if self.do_output_perturbation:
            # Set the seed
            np.random.seed(self.seed)
            # The weights columns are t, then the rest of the weights
            n_weights = df.shape[1] - 1
            # Note we add the same noise at every point during training
            # This is technically incorrect as the output perturbation scale should scale
            # With the number of iterations,
            # However so long as the output_perturbation_scale corresponds to
            # the timestep we are analysing, the output perturbation will work as required
            noise = np.random.normal(size=n_weights, scale=self.output_perturbation_scale)
            # Add it on - pandas/numpy should handle the broadcasting here
            df.iloc[:, 1:] += noise

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
                          diffinit=False, data_privacy='all', what='weights', sort=False,
                          do_output_perturbation: bool = False):
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

    base_experiment = ExperimentIdentifier(cfg_name, model, replace_index,
                                           diffinit=diffinit, data_privacy=data_privacy,
                                           do_output_perturbation=do_output_perturbation)

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


def get_pvals(what, cfg_name, model, t, n_experiments=3, diffinit=False) -> Tuple[np.ndarray, int]:
    """
    load weights/gradients and compute p-vals for them, then return them
    """
    assert what in ['weights', 'gradients']
    # set some stuff up
    iter_range = (t, t + 1)
    # sample experiments
    df = get_available_results(cfg_name, model, diffinit=diffinit)
    replace_indices = df['replace'].unique()
    replace_indices = np.random.choice(replace_indices, n_experiments, replace=False)
    print('Looking at replace indices...', replace_indices)
    all_pvals = []

    for i, replace_index in enumerate(replace_indices):
        experiment = ExperimentIdentifier(cfg_name, model, replace_index, seed=1, diffinit=diffinit)

        if what == 'gradients':
            print('Loading gradients...')
            df = experiment.load_gradients(noise=True, iter_range=iter_range, params=None)
            second_col = df.columns[1]
        elif what == 'weights':
            df = get_posterior_samples(cfg_name, iter_range=iter_range,
                                       model=model, replace_index=replace_index,
                                       params=None, seeds='all')
            second_col = df.columns[1]
        params = df.columns[2:]
        n_params = len(params)
        print(n_params)

        if n_params < 50:
            print('ERROR: Insufficient parameters for this kind of visualisation, please try something else')

            return False
        print('Identified', n_params, 'parameters, proceeding with analysis')
        p_vals = np.zeros(shape=(n_params))

        for j, p in enumerate(params):
            print('getting fit for parameter', p)
            df_fit = estimate_statistics_through_training(what=what, cfg_name=None,
                                                          model=None, replace_index=None,
                                                          seed=None,
                                                          df=df.loc[:, ['t', second_col, p]],
                                                          params=None, iter_range=None)
            p_vals[j] = df_fit.loc[t, 'norm_p']
            del df_fit
        log_pvals = np.log(p_vals)
        all_pvals.append(log_pvals)
    log_pvals = np.concatenate(all_pvals)
    return log_pvals, n_params
