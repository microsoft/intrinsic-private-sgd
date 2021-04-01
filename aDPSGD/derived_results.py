#!/usr/bin/env ipython
# Consume experiment results, produce higher-level statistics and such
# Some functions create "amortised" data
###

import ipdb
import abc
import numpy as np
import pandas as pd
from typing import Tuple
from scipy.stats import ttest_rel
#from test_private_model import test_model_with_noise, compute_wu_bound
from noise_utils import compute_wu_bound
import results_utils
import stats_utils
import experiment_metadata as em
# from visualisations import mvn_covariance


class DerivedResult(object):
    def __init__(self, cfg_name, model, data_privacy):
        self.cfg_name = cfg_name
        self.model = model
        self.data_privacy = data_privacy
        sample_experiment = results_utils.ExperimentIdentifier(cfg_name=cfg_name, model=model,
                                                               data_privacy=data_privacy)
        self.derived_directory = sample_experiment.derived_path_stub()
        self.suffix = None

    @abc.abstractmethod
    def identifier(self, diffinit: bool) -> str:
        pass

    @abc.abstractmethod
    def generate(self, diffinit: bool) -> None:
        pass

    def path_string(self, diffinit: bool = False):
        identifier = self.identifier(diffinit)
        path_string = (self.derived_directory / identifier).with_suffix(self.suffix)

        return path_string

    def load(self, diffinit: bool = True, generate_if_needed: bool = False, verbose=True):
        path = self.path_string(diffinit=diffinit)

        try:
            if self.suffix == '.npy':
                data = np.load(path, allow_pickle=True).item()
                if verbose:
                    print(f'Loaded derived data from {path}')
            elif self.suffix == '.csv':
                data = pd.read_csv(path)
            else:
                raise ValueError(f'Unknown suffix {self.suffix}')
        except FileNotFoundError:
            if generate_if_needed:
                self.generate(diffinit=diffinit)
                data = self.load(diffinit=diffinit, generate_if_needed=False)
            else:
                raise FileNotFoundError(f'{path} not found!')

        return data


class DeltaHistogram(DerivedResult):
    """
    Distribution of etc
    """
    def __init__(self, cfg_name, model, num_deltas='max', t=500,
                 data_privacy='all', multivariate=False, sort=False,
                 do_output_perturbation: bool = False):
        super(DeltaHistogram, self).__init__(cfg_name, model, data_privacy)
        self.num_deltas = num_deltas
        self.t = t
        self.multivariate = multivariate
        self.sort = sort
        self.suffix = '.npy'
        self.do_output_perturbation = do_output_perturbation

    def identifier(self, diffinit: bool) -> str:
        identifier = f'delta_histogram_nd{self.num_deltas}_t{self.t}{"_diffinit"*diffinit}{"_multivar"*self.multivariate}{"_PERTURBED" * self.do_output_perturbation}'

        if self.sort:
            identifier = f'{identifier}_sorted'

        print(f'Identifier: {identifier}')
        return identifier

    def generate(self, diffinit: bool = True) -> None:
        """ Note that diffinit isn't actually used """

        for diffinit in False, True:
            path_string = self.path_string(diffinit)

            if path_string.exists():
                print(f'WARNING: Delta histogram has already been generated, file {path_string} exists!')

                continue

            path_string.parent.mkdir(exist_ok=True)
            print('Couldn\'t find', path_string)
            # vary-both
            vary_both, identifiers_both = get_deltas(self.cfg_name,
                                                     iter_range=(self.t, self.t+1),
                                                     model=self.model,
                                                     vary_seed=True,
                                                     vary_data=True,
                                                     num_deltas=self.num_deltas,
                                                     diffinit=diffinit,
                                                     data_privacy=self.data_privacy,
                                                     multivariate=self.multivariate,
                                                     sort=self.sort,
                                                     do_output_perturbation=self.do_output_perturbation)
            # vary-S
            vary_S, identifiers_S = get_deltas(self.cfg_name, iter_range=(self.t, self.t+1),
                                               model=self.model,
                                               vary_seed=False, vary_data=True,
                                               num_deltas=self.num_deltas,
                                               diffinit=diffinit,
                                               data_privacy=self.data_privacy,
                                               multivariate=self.multivariate,
                                               sort=self.sort,
                                               do_output_perturbation=self.do_output_perturbation)
            # vary-r
            vary_r, identifiers_r = get_deltas(self.cfg_name, iter_range=(self.t, self.t+1),
                                               model=self.model,
                                               vary_seed=True, vary_data=False,
                                               num_deltas=self.num_deltas, diffinit=diffinit,
                                               data_privacy=self.data_privacy,
                                               multivariate=self.multivariate,
                                               sort=self.sort,
                                               do_output_perturbation=self.do_output_perturbation)

            # now save
            delta_histogram_data = {'vary_both': vary_both,
                                    'both_identifiers': identifiers_both,
                                    'vary_S': vary_S,
                                    'S_identifiers': identifiers_S,
                                    'vary_r': vary_r,
                                    'r_identifiers': identifiers_r}
            np.save(path_string, delta_histogram_data)
            print(f'[DeltaHistogram] Saved to {path_string}')


class UtilityCurve(DerivedResult):
    def __init__(self, cfg_name, model, num_deltas, t, data_privacy='all', metric_to_report='binary_accuracy',
                 verbose=True, num_experiments=5000, multivariate: bool = False):
        super(UtilityCurve, self).__init__(cfg_name, model, data_privacy)
        self.num_deltas = num_deltas
        self.num_experiments = num_experiments
        self.t = t
        self.metric_to_report = metric_to_report
        self.multivariate = multivariate
        self.suffix = '.csv'

    def identifier(self, diffinit: bool) -> str:
        identifier = f'utility_nd{self.num_deltas}_t{self.t}_ne{self.num_experiments}{"_multivar"*self.multivariate}'

        return identifier

    def generate(self, diffinit) -> None:
        path_string = self.path_string(diffinit)

        if path_string.exists():
            print(f'[UtilityCurve] WARNING: Utility curve has already been generated, file {path_string} exists!')

            return
        epsilons = np.array([0.5, 1.0])
        # prepare columns of dataframe
        seed = []
        replace = []
        eps_array = []
        noiseless = []
        bolton = []
        augment = []
        augment_diffinit = []
        sens_from = []
        # select a set of experiments
        df = results_utils.get_available_results(self.cfg_name, self.model, diffinit=True)
        random_experiments = df.iloc[np.random.choice(df.shape[0], self.num_experiments), :]

        for i, exp in random_experiments.iterrows():
            exp_seed = exp['seed']
            exp_replace = exp['replace']

            for sens_from_bound in [True, False]:
                if sens_from_bound and not self.model == 'logistic':
                    print(f'Skipping because model is {self.model} cant get sensitivity from bound')
                    # bound isnt meaningful for this model

                    continue

                for eps in epsilons:
                    results = test_model_with_noise(cfg_name=self.cfg_name,
                                                                       replace_index=exp_replace,
                                                                       seed=exp_seed, t=self.t, epsilon=eps,
                                                                       delta=None,
                                                                       sens_from_bound=sens_from_bound,
                                                                       metric_to_report=self.metric_to_report,
                                                                       verbose=False,
                                                                       num_deltas=self.num_deltas,
                                                                       multivariate=self.multivariate)
                    noiseless_at_eps, bolton_at_eps, augment_at_eps, augment_with_diffinit_at_eps = results
                    seed.append(exp_seed)
                    replace.append(exp_replace)
                    eps_array.append(eps)
                    noiseless.append(noiseless_at_eps)
                    bolton.append(bolton_at_eps)
                    augment.append(augment_at_eps)
                    augment_diffinit.append(augment_with_diffinit_at_eps)
                    sens_from.append(sens_from_bound)
        utility_data = pd.DataFrame({'seed': seed, 'replace': replace,
                                     'epsilon': eps_array, 'noiseless': noiseless,
                                     'bolton': bolton, 'augment': augment,
                                     'augment_diffinit': augment_diffinit,
                                     'sensitivity_from_bound': sens_from})

        utility_data.to_csv(path_string, header=True, index=False, mode='a')
        print(f'[UtilityCurve] Saved to {path_string}')


class AggregatedLoss(DerivedResult):
    """
    Collect loss from multiple experiments and compute mean, std etc.
    """
    def __init__(self, cfg_name, model, data_privacy='all', iter_range=(None, None)):
        super(AggregatedLoss, self).__init__(cfg_name, model, data_privacy)
        self.iter_range = iter_range
        self.suffix = '.csv'

    def identifier(self, diffinit: bool) -> str:
        identifier = f'aggregated_loss{"_diffinit"*diffinit}'

        return identifier

    def generate(self, diffinit):
        path_string = self.path_string(diffinit)

        if path_string.exists():
            print(f'[AggregatedLoss] {path_string} already exists, not recomputing!')

            return

        df = results_utils.get_available_results(self.cfg_name, self.model)
        train_list = []
        vali_list = []

        for i, row in df.iterrows():
            experiment = results_utils.ExperimentIdentifier(self.cfg_name, self.model, replace_index=row['replace'],
                                                            seed=row['seed'], diffinit=diffinit,
                                                            data_privacy=self.data_privacy)
            try:
                loss = experiment.load_loss(iter_range=self.iter_range, verbose=False)
            except FileNotFoundError:
                print(f'WARNING: Could not find loss for {experiment.path_stub()}')
                continue
            loss_train = loss.loc[loss['minibatch_id'] == 'ALL', :].set_index('t')
            loss_vali = loss.loc[loss['minibatch_id'] == 'VALI', :].set_index('t')
            train_list.append(loss_train)
            vali_list.append(loss_vali)
        print('All traces collected')
        # dataframe
        train = pd.concat(train_list)
        vali = pd.concat(vali_list)
        # aggregate: mean and std
        train_mean = train.groupby('t').mean()
        train_std = train.groupby('t').std()
        vali_mean = vali.groupby('t').mean()
        vali_std = vali.groupby('t').std()
        # recombine
        train = train_mean.join(train_std, rsuffix='_std', lsuffix='_mean')
        vali = vali_mean.join(vali_std, rsuffix='_std', lsuffix='_mean')
        df = train.join(vali, lsuffix='_train', rsuffix='_vali')

        self.suffix = '.csv'
        df.to_csv(path_string, header=True, index=True)
        print(f'[AggregatedLoss] Saved to {path_string}')

        return

    def load(self, diffinit: bool, generate_if_needed=False) -> pd.DataFrame:
        try:
            data = super(AggregatedLoss, self).load(diffinit)
        except FileNotFoundError:
            if generate_if_needed:
                self.generate(diffinit=diffinit)
                data = self.load(diffinit=diffinit, generate_if_needed=False)
            else:
                raise FileNotFoundError
        data.set_index('t', inplace=True)
        return data


class SensVar(DerivedResult):
    def __init__(self, cfg_name, model,  t, num_pairs='max', data_privacy='all'):
        # Note this doesn't support multivar
        super(SensVar, self).__init__(cfg_name, model, data_privacy)
        self.t = t
        self.num_pairs = num_pairs
        self.suffix = '.csv'

    def identifier(self, diffinit: bool) -> str:
        identifier = f'sens_var_dist_np{self.num_pairs}_t{self.t}{"_diffinit"*diffinit}'

        return identifier

    def generate(self, diffinit=True) -> None:
        for diffinit in [False, True]:
            path_string = self.path_string(diffinit)
            print(path_string)

            if path_string.exists():
                print(f'[SensVar] File {path_string} already exists - not recomputing!')

            else:
                # TODO check if we need to do both
                df = results_utils.get_available_results(self.cfg_name, self.model, diffinit=diffinit)
                replace_counts = df['replace'].value_counts()
                replaces = replace_counts[replace_counts > 10].index.values
                print('Found', len(replaces), 'datasets with at least 10 seeds')
                # For each pair of drops...
                num_replaces = len(replaces)
                sens_array = []
                var_array = []
                overlap_array = []
                pairs_array = []

                for i, di in enumerate(replaces):
                    for j in range(i + 1, num_replaces):
                        dj = replaces[j]
                        pairs_array.append((di, dj))

                if not self.num_pairs == 'max':
                    total_pairs = len(pairs_array)
                    print(total_pairs)
                    pair_picks = np.random.choice(total_pairs, self.num_pairs, replace=False)
                    pairs_array = [pairs_array[i] for i in pair_picks]
                print('Computing "local" epsilon for', len(pairs_array), 'pairs of datasets!')

                for di, dj in pairs_array:
                    pair_sens, pair_var, num_seeds = compute_pairwise_sens_and_var(self.cfg_name, self.model,
                                                                                   self.t, replace_indices=[di, dj],
                                                                                   multivariate=False,
                                                                                   verbose=False,
                                                                                   diffinit=diffinit)
                    sens_array.append(pair_sens)
                    var_array.append(pair_var)
                    overlap_array.append(num_seeds)
                df = pd.DataFrame({'pair': pairs_array,
                                   'sensitivity': sens_array,
                                   'variability': var_array,
                                   'overlapping_seeds': overlap_array})
                print(f'[SensVar] Saved to {path_string}')
                df.to_csv(path_string, header=True, index=False)

        return


class Sigmas(DerivedResult):
    """
    As for estimating the sensitivity, we want to grab a bunch of posteriors and estimate the variability """
    def __init__(self, cfg_name, model, t, num_replaces='max', num_seeds='max',
                 data_privacy='all', sort=False,
                 do_output_perturbation: bool = False):
        super(Sigmas, self).__init__(cfg_name, model, data_privacy)
        self.num_replaces = num_replaces
        self.num_seeds = num_seeds
        self.t = t
        self.sort = sort
        self.suffix = '.npy'
        self.do_output_perturbation = do_output_perturbation

    def identifier(self, diffinit: bool) -> str:
        identifier = f'sigmas_t{self.t}_ns{self.num_seeds}{"_diffinit"*diffinit}{"_PERTURBED" * self.do_output_perturbation}'

        if self.sort:
            identifier = f'{identifier}_sorted'

        return identifier

    def generate(self, diffinit, verbose=True, ephemeral=False):
        """ ephemeral allows us generate it and return without saving """

        if not ephemeral:
            path_string = self.path_string(diffinit)

            if path_string.exists():
                print(f'[Sigmas] File {path_string} already exists, not computing again!')

                return
        # now compute
        df = results_utils.get_available_results(self.cfg_name, self.model,
                                                 data_privacy=self.data_privacy,
                                                 diffinit=diffinit)
        replace_counts = df['replace'].value_counts()
        replaces = replace_counts[replace_counts > 2].index.values

        if verbose:
            print(f'[Sigmas] Estimating variability across {len(replaces)} datasets!')
            print('Warning: this can be slow...')
        sigmas = []
        used_replaces = []

        if not self.num_replaces == 'max' and self.num_replaces < len(replaces):
            replaces = np.random.choice(replaces, self.num_replaces, replace=False)

        for replace_index in replaces:
            if verbose:
                print('replace index:', replace_index)
            samples = results_utils.get_posterior_samples(self.cfg_name, (self.t, self.t+1),
                                                          self.model,
                                                          replace_index=replace_index,
                                                          params=None, seeds='all',
                                                          verbose=verbose,
                                                          diffinit=diffinit,
                                                          data_privacy=self.data_privacy,
                                                          num_seeds=self.num_seeds,
                                                          sort=self.sort,
                                                          do_output_perturbation=self.do_output_perturbation)
            try:
                params = samples.columns[2:]

                this_sigma = samples.std(axis=0)
                this_sigma = this_sigma[params]
                #else:
                #    params_vals = samples[params].values
                #    params_norm = params_vals - params_vals.mean(axis=0)
                #    params_flat = params_norm.flatten()
                #    this_sigma = np.std(params_flat)
                sigmas.append(this_sigma)
                used_replaces.append(replace_index)
            except AttributeError:
                print(f'WARNING: data from {replace_index} is bad - skipping')
                assert samples is False
                # Don\'t append anything
                #this_sigma = np.nan
            #sigmas.append(this_sigma)
        sigmas = np.array(sigmas)
        sigmas_data = {'sigmas': sigmas, 'replaces': used_replaces}

        if not ephemeral:
            np.save(path_string, sigmas_data)
        else:
            return sigmas_data


class VersusTime(DerivedResult):
    """
    Estimate the empirical (and theoretical I guess) sensitivity and variability v. "convergence point" (time)
    The objective is to create a CSV with columns:
    - convergence point
    - train loss
    - vali loss
    - theoretical sensitivity
    - empirical sensitivity
    - for weights w/ different seed:
        - average distance
        - min + max
        - std
    - variability w/out diffinit
    - variability with diffinit
    """
    def __init__(self, cfg_name, model, data_privacy='all',
                 iter_range=(0, 1000), num_deltas='max', cadence=200, sort=False,
                 multivariate: bool = False):
        super(VersusTime, self).__init__(cfg_name, model, data_privacy)
        self.iter_range = iter_range
        self.num_deltas = num_deltas
        assert None not in self.iter_range
        self.cadence = cadence
        self.multivariate = multivariate
        self.sort = sort
        self.suffix = '.csv'

    def identifier(self, diffinit: bool = False) -> str:
        identifier = f'versus_time_nd{self.num_deltas}'

        if self.sort:
            identifier = f'{identifier}_sorted'

        if self.multivariate:
            identifier = f'{identifier}_multivar'

        return identifier

    def generate(self, diffinit=True) -> None:
        path_string = self.path_string()

        if path_string.exists():
            print(f'[VersusTime] WARNING: Versus time has already been generated, file {path_string} exists!')

            return

        if self.model == 'logistic':
            # pre-fetch lr and N for logistic models, not used otherwise
            _, batch_size, lr, _, N = em.get_experiment_details(self.cfg_name, self.model,
                                                                data_privacy=self.data_privacy)
            L = np.sqrt(2)

        t_range = np.arange(self.iter_range[0], self.iter_range[1], self.cadence)
        n_T = len(t_range)
        theoretical_sensitivity_list = [np.nan]*n_T
        empirical_sensitivity_list = [np.nan]*n_T
        variability_fixinit_list = [np.nan]*n_T
        variability_diffinit_list = [np.nan]*n_T
        min_fixinit_distance_list = [np.nan]*n_T
        mean_fixinit_distance_list = [np.nan]*n_T
        max_fixinit_distance_list = [np.nan]*n_T
        std_fixinit_distance_list = [np.nan]*n_T
        min_diffinit_distance_list = [np.nan]*n_T
        mean_diffinit_distance_list = [np.nan]*n_T
        max_diffinit_distance_list = [np.nan]*n_T
        std_diffinit_distance_list = [np.nan]*n_T

        for i, t in enumerate(t_range):

            if self.model == 'logistic':
                theoretical_sensitivity = compute_wu_bound(L, t=t, N=N, batch_size=batch_size, eta=lr)
            else:
                theoretical_sensitivity = np.nan

            # empirical sensitivity computed for all models
            empirical_sensitivity = estimate_sensitivity_empirically(self.cfg_name, self.model, t,
                                                                     num_deltas=self.num_deltas,
                                                                     diffinit=True,
                                                                     data_privacy=self.data_privacy,
                                                                     sort=self.sort,
                                                                     multivariate=self.multivariate)

            assert empirical_sensitivity is not None

            # variability
            variability_fixinit = estimate_variability(self.cfg_name, self.model, t,
                                                       multivariate=self.multivariate,
                                                       diffinit=False,
                                                       data_privacy=self.data_privacy,
                                                       sort=self.sort)
            variability_diffinit = estimate_variability(self.cfg_name, self.model, t,
                                                        multivariate=self.multivariate,
                                                        diffinit=True,
                                                        data_privacy=self.data_privacy,
                                                        sort=self.sort)

            # distance statistics
            statistics_fixinit = compute_distance_statistics(self.cfg_name, self.model, t,
                                                             multivariate=self.multivariate,
                                                             diffinit=False,
                                                             data_privacy=self.data_privacy,
                                                             sort=self.sort)
            statistics_diffinit = compute_distance_statistics(self.cfg_name, self.model, t,
                                                              multivariate=self.multivariate,
                                                              diffinit=True,
                                                              data_privacy=self.data_privacy,
                                                              sort=self.sort)

            # If multivariate, flatten ?
            if self.multivariate:
                theoretical_sensitivity = np.mean(theoretical_sensitivity)
                empirical_sensitivity = np.mean(empirical_sensitivity)
                variability_fixinit = np.mean(variability_fixinit)
                variability_diffinit = np.mean(variability_diffinit)

                statistics_fixinit['min_distance'] = np.mean(statistics_fixinit['min_distance'])
                statistics_fixinit['mean_distance'] = np.mean(statistics_fixinit['mean_distance'])
                statistics_fixinit['max_distance'] = np.mean(statistics_fixinit['max_distance'])
                statistics_fixinit['std_distance'] = np.mean(statistics_fixinit['std_distance'])
                statistics_diffinit['min_distance'] = np.mean(statistics_diffinit['min_distance'])
                statistics_diffinit['mean_distance'] = np.mean(statistics_diffinit['mean_distance'])
                statistics_diffinit['max_distance'] = np.mean(statistics_diffinit['max_distance'])
                statistics_diffinit['std_distance'] = np.mean(statistics_diffinit['std_distance'])

            # now record everything
            theoretical_sensitivity_list[i] = theoretical_sensitivity
            empirical_sensitivity_list[i] = empirical_sensitivity
            variability_fixinit_list[i] = variability_fixinit
            variability_diffinit_list[i] = variability_diffinit
            # All the distance statistics
            min_fixinit_distance_list[i] = statistics_fixinit['min_distance']
            mean_fixinit_distance_list[i] = statistics_fixinit['mean_distance']
            max_fixinit_distance_list[i] = statistics_fixinit['max_distance']
            std_fixinit_distance_list[i] = statistics_fixinit['std_distance']
            min_diffinit_distance_list[i] = statistics_diffinit['min_distance']
            mean_diffinit_distance_list[i] = statistics_diffinit['mean_distance']
            max_diffinit_distance_list[i] = statistics_diffinit['max_distance']
            std_diffinit_distance_list[i] = statistics_diffinit['std_distance']

        # combine everything into a dataframe
        df = pd.DataFrame({'t': t_range,
                           'theoretical_sensitivity': theoretical_sensitivity_list,
                           'empirical_sensitivity': empirical_sensitivity_list,
                           'variability_fixinit': variability_fixinit_list,
                           'variability_diffinit': variability_diffinit_list,
                           'min_fixinit_distance': min_fixinit_distance_list,
                           'mean_fixinit_distance': mean_fixinit_distance_list,
                           'max_fixinit_distance': max_fixinit_distance_list,
                           'std_fixinit_distance': std_fixinit_distance_list,
                           'min_diffinit_distance': min_diffinit_distance_list,
                           'mean_diffinit_distance': mean_diffinit_distance_list,
                           'max_diffinit_distance': max_diffinit_distance_list,
                           'std_diffinit_distance': std_diffinit_distance_list})

        df.set_index('t', inplace=True)
        # now join the losses...
        losses = AggregatedLoss(self.cfg_name, self.model, iter_range=self.iter_range,
                                data_privacy=self.data_privacy).load(diffinit=True, generate_if_needed=True)
        df = df.join(losses)
        df.to_csv(path_string)
        print(f'[VersusTime] Saved to {path_string}')

        return


class Stability(DerivedResult):
    def __init__(self, cfg_name, model, t, data_privacy='all', sort=False):
        super(Stability, self).__init__(cfg_name, model, data_privacy)
        self.t = t
        self.suffix = '.npy'
        self.sort = sort

    def identifier(self, diffinit: bool = False) -> str:
        identifier = f'stability_t{self.t}'

        return identifier

    def generate(self, diffinit=True) -> None:
        path_string = self.path_string()

        if path_string.exists():
            print(f'[Stability] File {path_string} already exists, not computing again!')

            return

        sigma_df = compute_sigma_v_num_seeds(self.cfg_name, self.model, self.t)
        sens_df = compute_sens_v_num_deltas(self.cfg_name, self.model, self.t, sort=self.sort)
        stability_dict = {'sigma': sigma_df,
                          'sens': sens_df}
        print(f'[Stability] Saved to {path_string}')
        np.save(path_string, stability_dict)

        return


def generate_derived_results(cfg_name: str, model: str = 'logistic', t: int = None,
                             multivariate: bool = False,
                             do_output_perturbation: bool = False) -> None:
    if t is None:
        t, valid_frac = find_convergence_point(cfg_name, model, diffinit=True,
                                               tolerance=3, metric='binary_crossentropy', data_privacy='all')

        if valid_frac < 0.5:
            raise ValueError(f'Convergence point not good, valid fraction: {valid_frac}')
        else:
            print(f'Selecting t as convergence point {t}, valid fraction {valid_frac}')

    if model == 'mlp':
        assert not do_output_perturbation
        DeltaHistogram(cfg_name, model, t=t, sort=False, multivariate=multivariate).generate()
        Sigmas(cfg_name, model, t=t, sort=False).generate(diffinit=True)
        Stability(cfg_name, model, t=t, sort=False).generate()
        VersusTime(cfg_name, model, iter_range=(0, t+200), sort=False, multivariate=multivariate).generate()
        SensVar(cfg_name, model, t=t).generate()
    else:
        if do_output_perturbation:
            print('WARNING: Output perturbation is only implemented for DeltaHistogram and Sigmas!')
        DeltaHistogram(cfg_name, model, t=t, multivariate=multivariate, do_output_perturbation=do_output_perturbation).generate()
        AggregatedLoss(cfg_name, model).generate(diffinit=True)
        AggregatedLoss(cfg_name, model).generate(diffinit=False)
        Sigmas(cfg_name, model, t=t, do_output_perturbation=do_output_perturbation).generate(diffinit=True)
        Stability(cfg_name, model, t=t).generate()
        SensVar(cfg_name, model, t=t).generate()
        VersusTime(cfg_name, model, iter_range=(0, t+200), multivariate=multivariate).generate()
        # UtilityCurve(cfg_name, model, num_deltas='max', t=t, multivariate=multivariate).generate(diffinit=True)

    return


def calculate_epsilon(cfg_name, model, t, use_bound=False, diffinit=True,
                      num_deltas='max', multivariate=False, verbose=True,
                      take_sigma_as_min: bool = False,
                      take_sens_as_fixed: bool = False,
                      do_output_perturbation: bool = False):
    """
    just get the intrinsic epsilon
    """
    task, batch_size, lr, n_weights, N = em.get_experiment_details(cfg_name, model)
    delta = 1.0/(N**2)
    if take_sigma_as_min:
        variability = estimate_variability(cfg_name, model, t,
                                           multivariate=True,
                                           diffinit=diffinit, verbose=verbose,
                                           do_output_perturbation=do_output_perturbation)
        variability = np.min(variability)
    else:
        variability = estimate_variability(cfg_name, model, t,
                                           multivariate=multivariate,
                                           diffinit=diffinit, verbose=verbose,
                                           do_output_perturbation=do_output_perturbation)

    if use_bound:
        if model == 'logistic':
            sensitivity = compute_wu_bound(lipschitz_constant=np.sqrt(2), t=t, N=N,
                                           batch_size=batch_size, eta=lr, verbose=verbose)
        else:
            sensitivity = np.nan

        if multivariate and not take_sens_as_fixed:
            # The overall L2 norm is "sensitivity", so giving each dimension equal contribution (!!!), we get
            # Each dimension = sens/sqrt(d)
            assert n_weights == len(variability)
            sensitivity = np.array([sensitivity/np.sqrt(n_weights)]*n_weights)
    else:
        if take_sens_as_fixed:
            sensitivity = estimate_sensitivity_empirically(cfg_name, model, t, num_deltas=num_deltas,
                                                           diffinit=diffinit, multivariate=False,
                                                           verbose=verbose,
                                                           do_output_perturbation=do_output_perturbation)
        else:
            sensitivity = estimate_sensitivity_empirically(cfg_name, model, t, num_deltas=num_deltas,
                                                           diffinit=diffinit, multivariate=multivariate,
                                                           verbose=verbose,
                                                           do_output_perturbation=do_output_perturbation)
    if verbose:
        print('sensitivity:', sensitivity)
        print('variability:', variability)
        print('delta:', delta)
    c = np.sqrt(2 * np.log(1.25/delta))
    if multivariate:
        if take_sens_as_fixed:
            # we are not doing the multivariate thing and don't need the factor of root n
            epsilon = c * sensitivity / variability
        if not take_sens_as_fixed:
            sensitivity = sensitivity.flatten()
            # We have epsilon ~ sqrt(d) sens / var
            assert n_weights == len(sensitivity)
            if not take_sigma_as_min:
                assert len(variability) == len(sensitivity)
            epsilon = c * np.sqrt(n_weights) * sensitivity / variability
        # Now we take the largest
        print(epsilon)
        print(min(variability), max(variability))
        print(min(epsilon), max(epsilon))
        epsilon = max(epsilon)
    else:
        epsilon = c * sensitivity / variability

    return epsilon


def accuracy_at_eps(cfg_name, model, t, use_bound=False, num_experiments=500,
                    num_deltas='max', epsilon=1, do_test=False) -> dict:
    """
    """
    utility_data = UtilityCurve(cfg_name, model, num_deltas, t, num_experiments=num_experiments).load()

    if utility_data is None:
        print('No utility data available, please run UtilityCurve.generate')

        return {}

    if use_bound:
        utility_data = utility_data.loc[utility_data['sensitivity_from_bound'] == True, :]
    else:
        utility_data = utility_data.loc[utility_data['sensitivity_from_bound'] == False, :]
    df_eps = utility_data.loc[utility_data['epsilon'] == epsilon, :]
    mean_accuracy = df_eps['augment'].mean()
    std_accuracy = df_eps['augment'].std()
    mean_accuracy_diffinit = df_eps['augment_diffinit'].mean()
    std_accuracy_diffinit = df_eps['augment_diffinit'].std()
    mean_noiseless = df_eps['noiseless'].mean()
    std_noiseless = df_eps['noiseless'].std()
    mean_bolton = df_eps['bolton'].mean()
    std_bolton = df_eps['bolton'].std()

    if do_test:
        # do a paired (dependent) t-test
        print('\tAcross all epsilon...')
        statistic, pval = ttest_rel(utility_data['augment'], utility_data['bolton'])
        print('Pval of ttest between AUGMENT and BOLTON:', pval)
        print('Average difference:', np.mean(utility_data['augment'] - utility_data['bolton']))
        statistic, pval = ttest_rel(utility_data['augment_diffinit'], utility_data['bolton'])
        print('Pval of ttest between AUGMENT_DIFFINIT and BOLTON:', pval)
        print('Average difference:', np.mean(utility_data['augment_diffinit'] - utility_data['bolton']))
        diff = utility_data['augment_diffinit'] - utility_data['bolton']
        gap = utility_data['noiseless'] - utility_data['bolton']
        frac_improvement = diff/gap
        frac_improvement[~np.isfinite(frac_improvement)] = 0
        print('Average percent difference of gap:', np.mean(100*frac_improvement))

        print('\tAt epsilon = ' + str(epsilon) + '...')
        statistic, pval = ttest_rel(df_eps['augment'], df_eps['bolton'])
        print('Pval of ttest between AUGMENT and BOLTON:', pval)
        print('Average difference:', np.mean(df_eps['augment'] - df_eps['bolton']))
        statistic, pval = ttest_rel(df_eps['augment_diffinit'], df_eps['bolton'])
        print('Pval of ttest between AUGMENT_DIFFINIT and BOLTON:', pval)
        diff = df_eps['augment_diffinit'] - df_eps['bolton']
        print('Average difference:', np.mean(diff))
        gap = df_eps['noiseless'] - df_eps['bolton']
        frac_improvement = diff/gap
        frac_improvement[~np.isfinite(frac_improvement)] = 0
        print('Average percent difference of gap:', np.mean(100*frac_improvement))

    results = {'acc': [mean_accuracy, std_accuracy],
               'acc_diffinit': [mean_accuracy_diffinit, std_accuracy_diffinit],
               'bolton': [mean_bolton, std_bolton],
               'noiseless': [mean_noiseless, std_noiseless]}

    return results


def estimate_sensitivity_empirically(cfg_name, model, t, num_deltas, diffinit=False,
                                     data_privacy='all', multivariate=False,
                                     verbose=True, sort=False,
                                     do_output_perturbation: bool = False):
    """ pull up the histogram
    """
    delta_histogram_data = DeltaHistogram(cfg_name, model, num_deltas, t,
                                          data_privacy, multivariate, sort=sort,
                                          do_output_perturbation=do_output_perturbation).load(diffinit, generate_if_needed=True, verbose=verbose)
    vary_data_deltas = delta_histogram_data['vary_S']
    sensitivity = np.nanmax(vary_data_deltas, axis=0)

    return sensitivity


def compute_distance_statistics(cfg_name, model, t, num_deltas='max', diffinit=False,
                                data_privacy='all', multivariate=False,
                                verbose=True, sort=False):
    """ pull up the histogram
    """
    dh = DeltaHistogram(cfg_name, model, num_deltas, t, data_privacy,
                        multivariate, sort=sort)
    dh_data = dh.load(diffinit, generate_if_needed=True, verbose=verbose)
    vary_seed_deltas = dh_data['vary_r']
    statistics = dict()
    statistics['mean_distance'] = np.nanmean(vary_seed_deltas, axis=0)
    statistics['min_distance'] = np.nanmin(vary_seed_deltas, axis=0)
    statistics['max_distance'] = np.nanmax(vary_seed_deltas, axis=0)
    statistics['std_distance'] = np.nanstd(vary_seed_deltas, axis=0)

    return statistics


def get_deltas(cfg_name, iter_range, model,
               vary_seed=True, vary_data=True, params=None, num_deltas=100,
               include_identifiers=False, diffinit=False, data_privacy='all',
               multivariate=False, verbose=False, sort=False,
               do_output_perturbation: bool = False):
    """
    collect samples of weights from experiments on cfg_name+model, varying:
    - seed (vary_seed)
    - data (vary_data)

    to clarify, we want to estimate |w(S, r) - w(S', r')|,
    with potentially S' = S (vary_data = False), or r' = r (vary_seed = False)

    we need to make sure that we only compare like-with-like!

    we want to get num_deltas values of delta in the end
    """
    df = results_utils.get_available_results(cfg_name, model, diffinit=diffinit, data_privacy=data_privacy)
    # filter out replaces with only a small number of seeds
    seeds_per_replace = df['replace'].value_counts()
    good_replaces = seeds_per_replace[seeds_per_replace > 2].index
    replace_per_seed = df['seed'].value_counts()
    good_seeds = replace_per_seed[replace_per_seed > 2].index
    df = df[df['replace'].isin(good_replaces)]
    df = df[df['seed'].isin(good_seeds)]

    if num_deltas == 'max':
        num_deltas = int(df.shape[0]/2)
        print('Using num_deltas:', num_deltas)

    if df.shape[0] < 2*num_deltas:
        print('ERROR: Run more experiments, or set num_deltas to be at most', int(df.shape[0]/2))

        return None, None
    w_rows = np.random.choice(df.shape[0], num_deltas, replace=False)
    remaining_rows = [x for x in range(df.shape[0]) if x not in w_rows]
    df_remaining = df.iloc[remaining_rows]
    seed_options = df_remaining['seed'].unique()

    if len(seed_options) < 2:
        print('ERROR: Insufficient seeds!')

        return None, None
    data_options = df_remaining['replace'].unique()

    if len(data_options) == 1:
        print('ERROR: Insufficient data!')

        return None, None

    w = df.iloc[w_rows]
    w.reset_index(inplace=True)
    # now let's get comparators for each row of w!
    wp_data_vals = [np.nan]*w.shape[0]
    wp_seed_vals = [np.nan]*w.shape[0]

    for i, row in w.iterrows():
        row_data = row['replace']
        row_seed = row['seed']

        if not vary_seed:
            wp_seed = row_seed
        else:
            # get a new seed
            new_seed = np.random.choice(seed_options)

            while new_seed == row_seed:
                new_seed = np.random.choice(seed_options)
            wp_seed = new_seed

        if not vary_data:
            wp_data = row_data
        else:
            # get a new data
            new_data = np.random.choice(data_options)

            while new_data == row_data:
                new_data = np.random.choice(data_options)
            wp_data = new_data
        wp_data_vals[i] = wp_data
        wp_seed_vals[i] = wp_seed
    wp = pd.DataFrame({'replace': wp_data_vals, 'seed': wp_seed_vals})

    if vary_seed:
        # make sure the seed is always different
        assert ((wp['seed'].astype(int).values - w['seed'].astype(int).values) == 0).sum() == 0
    else:
        # make sure it's alwys the same
        assert ((wp['seed'].astype(int).values - w['seed'].astype(int).values) == 0).mean() == 1

    if vary_data:
        # make sure the data is always different
        assert ((wp['replace'].astype(int).values - w['replace'].astype(int).values) == 0).sum() == 0
    else:
        assert ((wp['replace'].astype(int).values - w['replace'].astype(int).values) == 0).mean() == 1

    deltas = [0]*num_deltas
    _, _, _, n_weights, _ = em.get_experiment_details(cfg_name, model)

    for i in range(num_deltas):
        replace_index = w.iloc[i]['replace']
        seed = w.iloc[i]['seed']

        exp = results_utils.ExperimentIdentifier(cfg_name, model, replace_index, seed,
                                                 diffinit, data_privacy,
                                                 do_output_perturbation=do_output_perturbation)

        if exp.exists():
            w_weights = exp.load_weights(iter_range=iter_range, params=params,
                                         verbose=False, sort=sort).values[:, 1:]
            # the first column is the time-step
        else:
            print('WARNING: Missing data for (seed, replace) = (', seed, replace_index, ')')
            if multivariate:
                w_weights = np.array([np.nan] * n_weights)
            else:
                w_weights = np.array([np.nan])
        replace_index_p = wp.iloc[i]['replace']
        seed_p = wp.iloc[i]['seed']

        exp_p = results_utils.ExperimentIdentifier(cfg_name, model, replace_index_p, seed_p,
                                                   diffinit, data_privacy,
                                                   do_output_perturbation=do_output_perturbation)

        if exp_p.exists():
            wp_weights = exp_p.load_weights(iter_range=iter_range, params=params,
                                            verbose=False, sort=sort).values[:, 1:]
        else:
            print('WARNING: Missing data for (seed, replace) = (', seed_p, replace_index_p, ')')
            if multivariate:
                wp_weights = np.array([np.nan] * n_weights)
            else:
                wp_weights = np.array([np.nan])

        if multivariate:
            delta = np.abs(w_weights - wp_weights)
        else:
            delta = np.linalg.norm(w_weights - wp_weights)
        deltas[i] = delta
    w_identifiers = list(zip(w['replace'], w['seed']))
    wp_identifiers = list(zip(wp['replace'], wp['seed']))
    identifiers = np.array(list(zip(w_identifiers, wp_identifiers)))

    deltas = np.array(deltas)

    return deltas, identifiers


def find_convergence_point_for_single_experiment(cfg_name, model, replace_index,
                                                 seed, diffinit=False, tolerance=3,
                                                 metric='ce', verbose=False,
                                                 data_privacy='all'):
    experiment = results_utils.ExperimentIdentifier(cfg_name, model, replace_index,
                                                    seed, diffinit=diffinit, data_privacy=data_privacy)
    loss = experiment.load_loss(iter_range=(None, None))
    try:
        assert metric in loss.columns
    except AssertionError:
        print('ERROR:', metric, 'is not in columns...', loss.columns)

        return np.nan
    loss = loss.loc[:, ['t', 'minibatch_id', metric]]
    loss = loss.pivot(index='t', columns='minibatch_id', values=metric)
    vali_loss = loss['VALI']
    delta_vali = vali_loss - vali_loss.shift()
    if 'accuracy' in metric:
        # we check for INCREASES
        change = (delta_vali > 0)
        change_type = 'increase'
    else:
        # was there a decrease at that time point? (1 if yes --> good)
        change = (delta_vali < 0)
        change_type = 'decrease'
    counter = 0

    for t, dec in change.items():
        if not dec:
            counter += 1
        else:
            counter = 0

        if counter >= tolerance:
            convergence_point = t

            break
    else:
        if verbose:
            print(f'Did not find instance of validation loss failing to {change_type} for {tolerance} steps - returning nan')
        convergence_point = np.nan

    return convergence_point


def find_convergence_point(cfg_name, model, diffinit, tolerance, metric, data_privacy='all'):
    """ wrapper for the whole experiment """
    results = results_utils.get_available_results(cfg_name, model, diffinit=diffinit, data_privacy=data_privacy)
    n_results = results.shape[0]
    points = np.zeros(n_results)

    for index, row in results.iterrows():
        replace_index = row['replace']
        seed = row['seed']
        point = find_convergence_point_for_single_experiment(cfg_name, model, replace_index,
                                                             seed, diffinit=diffinit,
                                                             tolerance=tolerance,
                                                             metric=metric,
                                                             data_privacy=data_privacy)
        points[index] = point
    print('For cfg_name', cfg_name, 'and model', model, 'with diffinit', diffinit, 'we have:')
    print('STDEV:', np.nanstd(points))
    print('MEDIAN:', np.nanmedian(points))
    print('MEAN:', np.nanmean(points))
    print('FRACTION INVALID:', np.mean(np.isnan(points)))
    convergence_point = np.nanmedian(points)
    valid_frac = np.mean(np.isfinite(points))
    print('Selecting median as convergence point:', convergence_point)

    return convergence_point, valid_frac


def compute_pairwise_sens_and_var(cfg_name, model, t, replace_indices,
                                  multivariate=False, verbose=True, diffinit=False):
    """
    for a pair of experiments...
    estimate sensitivity (distance between means)
    estimate variability (variance about means .. both?)
    given delta ... return this epsilon!
    optionally, by parameter (returns an array!)
    """

    if multivariate:
        raise NotImplementedError
    samples_1 = results_utils.get_posterior_samples(cfg_name, (t, t+1), model,
                                                    replace_index=replace_indices[0],
                                                    params=None, seeds='all',
                                                    verbose=verbose, diffinit=diffinit)
    samples_2 = results_utils.get_posterior_samples(cfg_name, (t, t+1), model,
                                                    replace_index=replace_indices[1],
                                                    params=None, seeds='all',
                                                    verbose=verbose, diffinit=diffinit)
    try:
        samples_1.set_index('seed', inplace=True)
        samples_2.set_index('seed', inplace=True)
    except AttributeError:
        print('ERROR: Issue loading samples from', replace_indices)

        return np.nan, np.nan, np.nan
    params = [x for x in samples_1.columns if not x == 't']
    samples_1 = samples_1[params]
    samples_2 = samples_2[params]
    # get intersection of seeds
    intersection = list(set(samples_1.index).intersection(set(samples_2.index)))
    num_seeds = len(intersection)

    if len(intersection) < 10:
        print(f'WARNING: Experiments with replace indices {replace_indices} only have {num_seeds} overlapping seeds: {intersection}')

        return np.nan, np.nan, num_seeds
    samples_1_intersection = samples_1.loc[intersection, :]
    samples_2_intersection = samples_2.loc[intersection, :]
    # compute the distances on the same seed
    distances = np.linalg.norm(samples_1_intersection - samples_2_intersection, axis=1)
    sensitivity = np.max(distances)

    if verbose:
        print('Max sensitivity from same seed diff data:', sensitivity)
    # compute distance by getting average value and comparing
    mean_1 = samples_1.mean(axis=0)
    mean_2 = samples_2.mean(axis=0)
    sensitivity_bymean = np.linalg.norm(mean_1 - mean_2)

    if verbose:
        print('Sensitivity from averaging posteriors and comparing:', sensitivity_bymean)
    variability_1 = (samples_1 - mean_1).values.std()
    variability_2 = (samples_2 - mean_2).values.std()
    variability = 0.5*(variability_1 + variability_2)

    if verbose:
        print('Variability:', variability)

    return sensitivity, variability, num_seeds


def estimate_variability(cfg_name, model, t, multivariate=False, diffinit=False,
                         data_privacy='all', num_replaces='max', num_seeds='max',
                         ephemeral=False, verbose=True, sort=False,
                         do_output_perturbation: bool = False):
    """
    This just pulls up the Sigmas result, and potentially subsets
    """
    sigmas_result = Sigmas(cfg_name, model, t, num_replaces, num_seeds,
                           data_privacy, sort=sort, do_output_perturbation=do_output_perturbation)

    if ephemeral:
        sigmas_data = sigmas_result.generate(diffinit, verbose=False, ephemeral=True)
    else:
        sigmas_data = sigmas_result.load(diffinit, generate_if_needed=True, verbose=verbose)

    if sigmas_data is None:
        return None

    sigmas = sigmas_data['sigmas']

    if num_replaces == 'max':
        sigmas = sigmas
    else:
        assert type(num_replaces) == int

        if num_replaces >= len(sigmas):
            if verbose and num_replaces > len(sigmas):
                print(f'WARNING: Can\'t select {num_replaces} sigmas, falling back to max ({len(sigmas)})')
            sigmas = sigmas
        else:
            if verbose:
                print(f'Sampling {num_replaces} random sigmas')
            n_sigmas = len(sigmas)
            sampled_sigmas = np.random.choice(n_sigmas, num_replaces, replace=False)
            sigmas = sigmas[sampled_sigmas]

    if verbose:
        print('Estimated variability using', len(sigmas[~np.isnan(sigmas)]), 'replaces')
    estimated_variability = np.nanmin(sigmas, axis=0)

    if not multivariate:
        estimated_variability = np.min(estimated_variability)

    return estimated_variability


def compute_sigma_v_num_seeds(cfg_name, model, t) -> pd.DataFrame:
    """
    """
    num_seeds_array = []
    num_replaces_array = []
    sigma_array = []

    for num_seeds in [2, 5, 10]*5 + [20, 30]*3 + [40, 50]*2 + [60, 70, 80, 90, 100, 200]:
        for num_replaces in [25]:
            sigma = estimate_variability(cfg_name, model, t=t,
                                         num_seeds=num_seeds, num_replaces=num_replaces,
                                         ephemeral=True, diffinit=True)
            num_seeds_array.append(num_seeds)
            num_replaces_array.append(num_replaces)
            sigma_array.append(sigma)
            print(f'{num_replaces} replaces, {num_seeds} seeds')
            print(f'\tsigma: {sigma}')

    stability_sigma = pd.DataFrame({'num_seeds': num_seeds_array,
                                    'num_replaces': num_replaces_array,
                                    'sigma': sigma_array})

    return stability_sigma


def compute_sens_v_num_deltas(cfg_name, model, t, sort=False):
    """
    compute empirical
    - sens
    - variability
    - epsilon
    with differing numbers of experiments, to test stability of estimates
    """
    num_deltas_array = []
    sens_array = []

    for num_deltas in [5, 10, 25, 50, 75, 100, 125, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000]:
        vary_S, _ = get_deltas(cfg_name, iter_range=(t, t+1),
                               model=model, vary_seed=False, vary_data=True,
                               num_deltas=num_deltas, diffinit=True,
                               data_privacy='all', multivariate=False,
                               sort=sort)

        if vary_S is None:
            sens = None
        else:
            print('should have', num_deltas, 'deltas, actually have:', len(vary_S[~np.isnan(vary_S)]))
            sens = np.nanmax(vary_S)
        print(f'{num_deltas} deltas')
        print(f'\tsens: {sens}')
        num_deltas_array.append(num_deltas)
        sens_array.append(sens)
    stability_sens = pd.DataFrame({'num_deltas': num_deltas_array,
                                   'sens': sens_array})

    return stability_sens


def compute_mvn_laplace_fit_and_alpha(cfg_name, model, t, diffinit=True, sort=False,
                                      just_on_normal_marginals=False,
                                      replace_index=None) -> dict:
    if replace_index is None:
        replace_index = results_utils.get_replace_index_with_most_seeds(cfg_name, model, diffinit=diffinit)

    iter_range = (t, t + 1)
    params = None
    df = results_utils.get_posterior_samples(cfg_name, model=model,
                                             replace_index=replace_index,
                                             iter_range=iter_range,
                                             params=params, diffinit=diffinit,
                                             what='weights',
                                             sort=sort)
    df_t = df.loc[df['t'] == t, :]
    X = df_t.iloc[:, 2:].values
    X = X - X.mean(axis=0)
    d = X.shape[1]
    if just_on_normal_marginals:
        print('Selecting just those parameters with normal marginals')
        normal_marginals = []
        for di in range(d):
            Xd = X[:, di]
            _, _, _, pval = stats_utils.fit_normal(Xd)
            if pval > 0.05:
                normal_marginals.append(di)
        print(f'Found {len(normal_marginals)} parameters with normally-distributed marginals!')
        X = X[:, normal_marginals]
        d = X.shape[1]
    if d > 55:
        print(f'More than 55 features (d = {d}), selecting a random subset')
        n_replicates = 2 * (d // 55 + 1)
        print(f'Using {n_replicates} replicates')
        p_array = []
        for _ in range(n_replicates):
            idx_subset = np.random.choice(d, 55, replace=False)
            X_sub = X[:, idx_subset]
            _, _, _, p = stats_utils.fit_multivariate_normal(X_sub)
            p_array.append(p)
        print(p_array)
        p = np.min(p_array)
        print(np.mean(p_array), np.std(p_array))
    else:
        _, _, _, p = stats_utils.fit_multivariate_normal(X)

    alpha, _ = stats_utils.fit_alpha_stable(X)
    # mvn_covariance(X, identifier=f'{cfg_name}_{t}')

    # now for laplace
    laplace_ps = []
    for di in range(d):
        Xd = X[:, di]
        _, _, _, pval = stats_utils.fit_laplace(Xd)
        laplace_ps.append(pval)
    laplace_ps = np.array(laplace_ps)
    print('without bonferroni...')
    fraction_of_laplace_vars = np.mean(laplace_ps > 0.05)
    print(f'\tfraction of laplace vars: {fraction_of_laplace_vars}')
    sum_of_laplace_vars = np.sum(laplace_ps > 0.05)
    print(f'\tsum of laplace vars: {sum_of_laplace_vars}')
    print('with bonferroni...')
    fraction_of_laplace_vars = np.mean(laplace_ps > 0.05/d)
    print(f'\tfraction of laplace vars: {fraction_of_laplace_vars}')
    sum_of_laplace_vars = np.sum(laplace_ps > 0.05/d)
    print(f'\tsum of laplace vars: {sum_of_laplace_vars}')
    mean_of_laplace_ps = np.mean(laplace_ps)
    print(f'mean of laplace ps: {mean_of_laplace_ps}')
    max_of_laplace_ps = np.max(laplace_ps)
    print(f'max of laplace ps: {max_of_laplace_ps}')

    return {'mvn p': p, 'alpha': alpha}


def get_pvals(what, cfg_name, model, t, n_experiments=3, diffinit=False) -> Tuple[np.ndarray, int]:
    """
    load weights/gradients and compute p-vals for them, then return them
    """
    assert what in ['weights', 'gradients']
    # set some stuff up
    iter_range = (t, t + 1)
    # sample experiments
    df = results_utils.get_available_results(cfg_name, model, diffinit=diffinit)
    replace_indices = df['replace'].unique()
    replace_indices = np.random.choice(replace_indices, n_experiments, replace=False)
    print('Looking at replace indices...', replace_indices)
    all_pvals = []

    for i, replace_index in enumerate(replace_indices):
        print(cfg_name)
        experiment = results_utils.ExperimentIdentifier(cfg_name, model, replace_index,
                                                        seed=1, diffinit=diffinit)

        if what == 'gradients':
            print('Loading gradients...')
            df = experiment.load_gradients(noise=True, iter_range=iter_range, params=None)
            second_col = df.columns[1]
        elif what == 'weights':
            df = results_utils.get_posterior_samples(cfg_name, iter_range=iter_range,
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
            df_fit = stats_utils.estimate_statistics_through_training(what=what, cfg_name=None,
                                                                      model=None, replace_index=None,
                                                                      seed=None,
                                                                      df=df.loc[:, ['t', second_col, p]],
                                                                      params=None, iter_range=None,
                                                                      include_mvn=False)
            p_vals[j] = df_fit.loc[t, f'{what}_norm_p']
            del df_fit
        log_pvals = np.log(p_vals)
        all_pvals.append(log_pvals)
    log_pvals = np.concatenate(all_pvals)
    return log_pvals, n_params


def check_offdiagonal(cfg_name: str, model: str, t: int) -> None:
    diffinit = True
    sort = False
    replace_index = results_utils.get_replace_index_with_most_seeds(cfg_name, model, diffinit=diffinit)

    iter_range = (t, t + 1)
    params = None
    df = results_utils.get_posterior_samples(cfg_name, model=model,
                                             replace_index=replace_index,
                                             iter_range=iter_range,
                                             params=params, diffinit=diffinit,
                                             what='weights',
                                             sort=sort)
    df = df.loc[df['t'] == t, :].drop(columns=['t', 'seed'])
    X = df.values
    d = X.shape[1]

    cov = np.cov(X.T)
    cor = np.corrcoef(X.T)
    assert cov.shape == (d, d)
    assert cor.shape == (d, d)
    offdiag_cov = cov - np.diag(np.diag(cov))
    offdiag_cor = cor - np.diag(np.diag(cor))

    cov_vals = np.abs(offdiag_cov[np.triu_indices(d)])
    cor_vals = np.abs(offdiag_cor[np.triu_indices(d)])

    print(f'cov mean: {np.mean(cov_vals)}')
    print(f'cov median: {np.median(cov_vals)}')

    print(f'cor mean: {np.mean(cor_vals)}')
    print(f'cor median: {np.median(cor_vals)}')
