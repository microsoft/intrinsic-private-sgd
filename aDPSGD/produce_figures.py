#!/usr/bin/env ipython
# This script is for performing specific analyses/generating figures
# It relies on e.g. statistics computed across experiments - "derived results"

import numpy as np
import re
import test_private_model
import derived_results as dr
import experiment_metadata as em
import vis_utils
import data_utils
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_rgba
from pathlib import Path
import seaborn as sns
import yaml


plt.switch_backend('Agg')
params = {'font.family': 'sans-serif',
          'font.size': 10}
plt.rcParams.update(params)

# Change this if necessary
FIGS_DIR = Path('./figures/')


def generate_plots(cfg_name: str, model: str, t=None, sort=False) -> None:
    """
    Wrapper to generate plots for a specific config
    (other plots compare across multiple configs)
    """
    # Load plot options from a yaml file (to make it easier to rerun)
    try:
        plot_options = yaml.load(open(FIGS_DIR / 'plot_options.yaml'))
        plot_options = plot_options[cfg_name]

        if t is None:
            t = plot_options['convergence_point']
#        delta_histogram_xlim = plot_options['delta_histogram']['xlim']
        delta_histogram_ymax = plot_options['delta_histogram']['ymax']
        delta_histogram_ylim = (0, delta_histogram_ymax)
        delta_histogram_xlim = None
#        delta_histogram_ylim = plot_options['delta_histogram']['ylim']
    except FileNotFoundError:
        assert t is not None
        delta_histogram_xlim = None
        delta_histogram_ylim = None
        epsilon_distribution_xlim = None
        epsilon_distribution_ylim = None
        versus_time_acc_lims = None

    plot_delta_histogram(cfg_name, model, t=t, include_bounds=(model == 'logistic'),
                         xlim=delta_histogram_xlim, ylim=delta_histogram_ylim,
                         sort=sort, legend=False)
    plot_stability_of_estimated_values(cfg_name, model, t, sort=sort)
    plot_distance_v_time(cfg_name, model, sort, convergence_point=t,
                         legend=('forest' in cfg_name))

    return


def generate_reports(cfg_name: str, model: str, t=None, num_experiments=500) -> None:
    print('\n')
    print(f'Report for {cfg_name} with {model} at {t}')
    print('\n')
    res = dr.compute_mvn_fit_and_alpha(cfg_name, model, t, diffinit=True)
    p = res['mvn p']
    alpha = res['alpha']
    print(f'Fit of MVN: \t\t\t\t{p}')
    print(f'Alpha from alpha-stable: \t\t{alpha}')

    empirical_sensitivity = dr.estimate_sensitivity_empirically(cfg_name, model, t,
                                                                num_deltas='max',
                                                                diffinit=True,
                                                                verbose=False)
    print(f'Empirical sensitivity: \t\t\t{empirical_sensitivity}')

    _, batch_size, lr, _, N = em.get_experiment_details(cfg_name, model)

    if model == 'logistic':
        theoretical_sensitivity = test_private_model.compute_wu_bound(lipschitz_constant=np.sqrt(2),
                                                                      t=t, N=N, batch_size=batch_size,
                                                                      eta=lr, verbose=False)
        print(f'Theoretical sensitivity from bound: \t{theoretical_sensitivity}')
        print('')

    empirical_variability = dr.estimate_variability(cfg_name, model, t=t, diffinit=False, verbose=False)
    empirical_variability_diffinit = dr.estimate_variability(cfg_name, model, t=t, diffinit=True, verbose=False)
    print(f'Empirical sigma (fixed init): \t\t{empirical_variability}')
    print(f'Empirical sigma (variable init): \t{empirical_variability_diffinit}')
    print('')

    print(f'Delta: {1/(N**2)}')
    print('')

    if model == 'logistic':
        epsilon_theoretical = dr.calculate_epsilon(cfg_name, model, t=t, use_bound=True, verbose=False)
        print(f'Epsilon using theoretical sensitivity: \t{epsilon_theoretical}')
    epsilon_empirical = dr.calculate_epsilon(cfg_name, model, t=t, use_bound=False, verbose=False)
    print(f'Epsilon using empirical sensitivity: \t{epsilon_empirical}')
    print('')

    perf_theoretical_eps1 = dr.accuracy_at_eps(cfg_name, model, t, use_bound=True,
                                               epsilon=1, num_experiments=num_experiments)
    perf_empirical_eps1 = dr.accuracy_at_eps(cfg_name, model, t, use_bound=False,
                                             epsilon=1, num_experiments=num_experiments)
    perf_theoretical_eps05 = dr.accuracy_at_eps(cfg_name, model, t, use_bound=True,
                                                epsilon=0.5, num_experiments=num_experiments)
    perf_empirical_eps05 = dr.accuracy_at_eps(cfg_name, model, t, use_bound=False,
                                              epsilon=0.5, num_experiments=num_experiments)
    print(f'Noiseless performance: \t\t\t{perf_theoretical_eps1["noiseless"]}')
    print('')

    print('Performance at epsilon = 1...')
    print('\tWith theoretical sensitivity:')
    print(f'\t\tBolton: \t\t{perf_theoretical_eps1["bolton"]}')
    print(f'\t\taDPSGD (fixinit): \t{perf_theoretical_eps1["acc"]}')
    print(f'\t\taDPSGD (diffinit): \t{perf_theoretical_eps1["acc_diffinit"]}')
    print('\tWith empirical sensitivity:')
    print(f'\t\tBolton: \t\t{perf_empirical_eps1["bolton"]}')
    print(f'\t\taDPSGD (fixinit): \t{perf_empirical_eps1["acc"]}')
    print(f'\t\taDPSGD (diffinit): \t{perf_empirical_eps1["acc_diffinit"]}')

    print('Performance at epsilon = 0.5...')
    print('\tWith theoretical sensitivity:')
    print(f'\t\tBolton: \t\t{perf_theoretical_eps05["bolton"]}')
    print(f'\t\taDPSGD (fixinit): \t{perf_theoretical_eps05["acc"]}')
    print(f'\t\taDPSGD (diffinit): \t{perf_theoretical_eps05["acc_diffinit"]}')
    print('\tWith empirical sensitivity:')
    print(f'\t\tBolton: \t\t{perf_empirical_eps05["bolton"]}')
    print(f'\t\taDPSGD (fixinit): \t{perf_empirical_eps05["acc"]}')
    print(f'\t\taDPSGD (diffinit): \t{perf_empirical_eps05["acc_diffinit"]}')

    return


def plot_delta_histogram(cfg_name: str, model: str, num_deltas='max', t=500,
                         include_bounds=False, xlim=None, ylim=None,
                         data_privacy='all', multivariate=False,
                         sort=False, legend=True) -> None:

    if multivariate:
        raise NotImplementedError('Multivariate plotting is not implemented')
    delta_histogram = dr.DeltaHistogram(cfg_name, model, num_deltas, t, data_privacy, multivariate, sort=sort)
    plot_data = delta_histogram.load(diffinit=False)
    plot_data_diffinit = delta_histogram.load(diffinit=True)

    vary_both = plot_data['vary_both']
    vary_S = plot_data['vary_S']
    vary_r = plot_data['vary_r']

    vary_both_diffinit = plot_data_diffinit['vary_both']
    vary_S_diffinit = plot_data_diffinit['vary_S']
    vary_r_diffinit = plot_data_diffinit['vary_r']

    # remove NANs
    vary_both = vary_both[~np.isnan(vary_both)]
    vary_S = vary_S[~np.isnan(vary_S)]
    vary_r = vary_r[~np.isnan(vary_r)]
    vary_both_diffinit = vary_both_diffinit[~np.isnan(vary_both_diffinit)]
    vary_S_diffinit = vary_S_diffinit[~np.isnan(vary_S_diffinit)]
    vary_r_diffinit = vary_r_diffinit[~np.isnan(vary_r_diffinit)]
    # merge vary_S for the different initialisations
    vary_S = np.concatenate([vary_S, vary_S_diffinit])

    plt.clf()
    plt.close()
    fig, axarr = plt.subplots(nrows=1, ncols=1, figsize=(4, 2.1))
    print('Plotting varying S... number of deltas:', vary_S.shape[0])
    sns.distplot(vary_S, ax=axarr,
                 color=em.dp_colours['bolton'],
                 label=r'$\Delta_S$',
                 kde=True,
                 hist_kws={'alpha': 0.45},
                 norm_hist=True)
    print('Plotting varying r... number of deltas:', vary_r.shape[0])
    sns.distplot(vary_r, ax=axarr,
                 color=em.dp_colours['augment'],
                 label=r'$\Delta_V^{fix}$',
                 kde=True,
                 hist_kws={'alpha': 0.7},
                 norm_hist=True)
    sns.distplot(vary_r_diffinit, ax=axarr,
                 color=em.dp_colours['augment_diffinit'],
                 label=r'$\Delta_V^{vary}$',
                 kde=True,
                 hist_kws={'alpha': 0.9},
                 norm_hist=True)

    print('Plotting varying both... number of deltas:', vary_both.shape[0])
    sns.distplot(vary_both, ax=axarr,
                 color=em.dp_colours['both'],
                 label=r'$\Delta_{S+V}^{fix}$',
                 kde=True,
                 hist=False,
                 kde_kws={'linestyle': '--'})
    sns.distplot(vary_both_diffinit, ax=axarr,
                 color=em.dp_colours['both_diffinit'],
                 label=r'$\Delta_{S+V}^{vary}$',
                 kde=True,
                 hist=False,
                 kde_kws={'linestyle': ':', 'lw': 2})

    if include_bounds:
        assert model == 'logistic'
        lipschitz_constant = np.sqrt(2.0)
        _, batch_size, lr, _, N = em.get_experiment_details(cfg_name, model, verbose=True)
        wu_bound = test_private_model.compute_wu_bound(lipschitz_constant, t=t, N=N, batch_size=batch_size, eta=lr)
        axarr.axvline(x=wu_bound, ls='--', color=em.dp_colours['bolton'], label=r'$\hat{\Delta}_S$')

    if legend:
        axarr.legend()
    else:
        axarr.get_legend().remove()
    axarr.set_xlabel(r'$\|w - w^\prime\|$')
    axarr.set_ylabel('density')

    if xlim is not None:
        axarr.set_xlim(xlim)

    if ylim is not None:
        axarr.set_ylim(ylim)

    vis_utils.beautify_axes(np.array([axarr]))
    plt.tight_layout()

    figure_identifier = f'delta_histogram_{cfg_name}_{data_privacy}_{model}_t{t}'

    if sort:
        figure_identifier = f'{figure_identifier}_sorted'
    plt.savefig((FIGS_DIR / figure_identifier).with_suffix('.png'))
    plt.savefig((FIGS_DIR / figure_identifier).with_suffix('.pdf'))

    return


def plot_distance_v_time(cfg_name, model, num_pairs='max', sort=False,
                         convergence_point=None, legend=True) -> None:
    """
    This will take precedence over the normal sens_var_over_time one
    """
    df = dr.VersusTime(cfg_name, model, sort=sort).load()

    # Get distance (vary seed)
    distance_columns = [x for x in df.columns if 'distance' in x]
    df_distance = df[['t'] + distance_columns]
    df_distance.dropna(axis=0, inplace=True)

    # Get sensitivity (vary data)
    df_sens = df[['t', 'theoretical_sensitivity', 'empirical_sensitivity']]
    if model in ['mlp', 'cnn']:
        df_sens.drop(columns='theoretical_sensitivity', inplace=True)
    else:
        # discretise the sensitivity
        ds = [np.nan]*df_sens.shape[0]
        for i, ts in enumerate(df_sens['theoretical_sensitivity'].values):
            ds[i] = test_private_model.discretise_theoretical_sensitivity(cfg_name, model, ts)
        df_sens['theoretical_sensitivity_discretised'] = ds
    df_sens.dropna(axis=0, inplace=True)

    # Now plot
    size = 6
    fig, axarr = plt.subplots(nrows=1, ncols=1, figsize=(4, 2.1))

    # First distance (vary seed)
    t = df_distance['t']
    which_colours = {'fixinit': em.dp_colours['augment'],
                     'diffinit': em.dp_colours['augment_diffinit']}
    which_labels = {'fixinit': np.nan,
                    'diffinit': r'$\Delta_V^{vary}$'}

    for which in ['diffinit']:           # not interested in fixinit
        min_dist = df_distance[f'min_{which}_distance']
        mean_dist = df_distance[f'mean_{which}_distance']
        max_dist = df_distance[f'max_{which}_distance']
        std_dist = df_distance[f'std_{which}_distance']
        axarr.plot(t, mean_dist, label=which_labels[which], color=which_colours[which], alpha=0.5)
        axarr.scatter(t, mean_dist, color=which_colours[which], label='_nolegend_', s=size)
        axarr.fill_between(t, mean_dist - std_dist, mean_dist + std_dist,
                           alpha=0.2, label='_nolegend_', color=which_colours[which])
        axarr.fill_between(t, min_dist, max_dist, alpha=0.1,
                           label='_nolegend_', color=which_colours[which])

    # Now sensitivity (vary data)
    t = df_sens['t']
    if 'theoretical_sensitivity_discretised' in df_sens:
        axarr.plot(t, df_sens['theoretical_sensitivity_discretised'],
                   label=r'$\hat{\Delta}_S$', alpha=0.5, c=em.dp_colours['bolton'], ls='--')
    axarr.scatter(t, df_sens['empirical_sensitivity'],
                  label='_nolegend_', s=size, c=em.dp_colours['bolton'])
    axarr.plot(t, df_sens['empirical_sensitivity'],
               label=r'$\hat{\Delta}^*_S$',
               ls=':', alpha=0.5, c=em.dp_colours['bolton'])

    if convergence_point is not None:
        # add a vertical line
        axarr.axvline(x=convergence_point, ls='--', alpha=0.5, color='black')
    # Now save and stuff
    if legend:
        axarr.legend()
    axarr.set_ylabel(r'$\|w - w^\prime\|$')
    axarr.set_xlabel('training steps')
    xmin, _ = axarr.get_xlim()           # this is a hack for mnist
    axarr.set_xlim(xmin, t.max())

    vis_utils.beautify_axes(np.array([axarr]))
    plt.tight_layout()

    figure_identifier = f'distance_v_time_{cfg_name}'
    plt.savefig((FIGS_DIR / figure_identifier).with_suffix('.png'))
    plt.savefig((FIGS_DIR / figure_identifier).with_suffix('.pdf'))

    return


def plot_stability_of_estimated_values(cfg_name, model, t) -> None:
    stability = dr.Stability(cfg_name, model, t)
    stability_dict = stability.load()

    # lets just do 3 separate plots
    figsize = (3.5, 2.8)
    size = 6
    # SIGMA V N SEEDS
    print('Plotting sigma v seeds')
    sigma_df = stability_dict['sigma']
    sigma_v_seed = sigma_df[['num_seeds', 'sigma']]
    fig, axarr = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    axarr.scatter(sigma_v_seed['num_seeds'], sigma_v_seed['sigma'], s=size, c=em.dp_colours['augment_diffinit'])
    sigma_we_use = dr.estimate_variability(cfg_name, model, t, diffinit=True)
    axarr.axhline(y=sigma_we_use, ls='--', c=em.dp_colours['augment_diffinit'], alpha=0.4)
    axarr.set_xlabel('number of random seeds')
    axarr.set_ylabel(r'estimated $\sigma_i(\mathcal{D})$')
    axarr.set_title(em.get_dataset_name(cfg_name) + ' (' + em.model_names[model] + ')')
    upper_y = 1.05*max(np.max(sigma_v_seed['sigma']), sigma_we_use)
    lower_y = 0.95*np.min(sigma_v_seed['sigma'])
    axarr.set_ylim(lower_y, upper_y)
    vis_utils.beautify_axes(np.array([axarr]))

    plt.tight_layout()
    figure_identifier = f'stability_sigma_v_seeds_{cfg_name}_{model}_t{t}'
    plt.savefig((FIGS_DIR / figure_identifier).with_suffix('.png'))
    plt.savefig((FIGS_DIR / figure_identifier).with_suffix('.pdf'))

    plt.clf()
    plt.close()

    # With fixed num_deltas, sensitivity
    print('Plotting sens v num deltas')
    sens_df = stability_dict['sens']
    sens_v_deltas = sens_df[['num_deltas', 'sens']].drop_duplicates()
    fig, axarr = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    axarr.scatter(sens_v_deltas['num_deltas'], sens_v_deltas['sens'], s=size, c=em.dp_colours['bolton'])
    sens_we_use = dr.estimate_sensitivity_empirically(cfg_name, model, t,
                                                      num_deltas='max', diffinit=True,
                                                      data_privacy='all')
    axarr.axhline(y=sens_we_use, ls='--', c=em.dp_colours['bolton'], alpha=0.4)
    axarr.set_xlabel('number of dataset comparisons')
    axarr.set_ylabel('estimated sensitivity')
    axarr.set_ylim(0, None)
    axarr.set_xscale('log')
    axarr.set_title(em.get_dataset_name(cfg_name) + ' (' + em.model_names[model] + ')')
    vis_utils.beautify_axes(np.array([axarr]))
    plt.tight_layout()

    figure_identifier = f'stability_sens_v_deltas_{cfg_name}_{model}_t{t}'
    plt.savefig((FIGS_DIR / figure_identifier).with_suffix('.png'))
    plt.savefig((FIGS_DIR / figure_identifier).with_suffix('.pdf'))

    plt.clf()
    plt.close()

    return


def overlay_pval_plot(model='logistic', xlim=None, n_experiments=50,
                      cfg_names=None, ylim=None) -> None:
    """
    want to overlay pvals from the four datasets in one plot
    """
    what = 'weights'
    figsize = (3.7, 3.05)

    if model == 'logistic':
        convergence_points = em.lr_convergence_points
        title = 'Logistic regression'
    else:
        convergence_points = em.nn_convergence_points
        title = 'Neural network'

    if cfg_names is None:
        cfg_names = em.dataset_colours.keys()
        plot_label = '_'
    else:
        plot_label = '_'.join(cfg_names) + '_'
    fig, axarr = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    vertical_lines_we_already_have = set()

    for ds in cfg_names:
        print(ds)
        log_pvals, n_params = vis_utils.fit_pval_histogram(what=what, dataset=ds, model=model,
                                                           t=convergence_points[ds],
                                                           n_experiments=n_experiments,
                                                           plot=False)
        sns.distplot(log_pvals, kde=True, bins=min(100, int(len(log_pvals)*0.25)),
                     ax=axarr, color=em.dataset_colours[ds], norm_hist=True,
                     label=em.get_dataset_name(ds),
                     kde_kws={'alpha': 0.6})

        if n_params not in vertical_lines_we_already_have:
            axarr.axvline(x=np.log(0.05/(n_params*n_experiments)), ls='--',
                          label='p = 0.05/' + str(n_params*n_experiments),
                          color=em.dataset_colours[ds], alpha=0.75)
            vertical_lines_we_already_have.add(n_params)
    axarr.axvline(x=np.log(0.05), ls=':', label='p = 0.05', color='black', alpha=0.75)
    axarr.legend()
    axarr.set_xlabel(r'$\log(p)$')
    axarr.set_ylabel('density')
    axarr.set_title(title)

    if ylim is not None:
        axarr.set_ylim(ylim)

    if xlim is not None:
        axarr.set_xlim(xlim)
    else:
        axarr.set_xlim((None, 0.01))
    vis_utils.beautify_axes(np.array([axarr]))
    plt.tight_layout()
    figure_identifier = f'pval_histogram_{plot_label}_{model}'
    plt.savefig((FIGS_DIR / figure_identifier).with_suffix('.png'))
    plt.savefig((FIGS_DIR / figure_identifier).with_suffix('.pdf'))

    return
