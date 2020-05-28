#!/usr/bin/env ipython
# This script is for performing specific analyses/generating figures
# It relies on e.g. statistics computed across experiments - "derived results"

# may need this for fonts
# sudo apt-get install ttf-mscorefonts-installer
# sudo apt-get install texlive-full

import numpy as np
import pandas as pd
import re
import ipdb
import results_utils
import test_private_model
import derived_results as dr
import experiment_metadata as em
import vis_utils
import data_utils
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_rgba
import seaborn as sns


plt.switch_backend('Agg')
params = {'font.family': 'sans-serif',
          'font.size': 10}
plt.rcParams.update(params)


# DEFINE SOME CONSTANTS !
augment_colour = '#14894e'
both_colour = 'black'
augment_diffinit_colour = '#441e85'
both_diffinit_colour = 'black'
bolton_colour = '#c3871c'


def weight_evolution(dataset, model, n_seeds=50, replace_indices=None,
                     iter_range=(None, None), params=['#4', '#2'],
                     diffinit=False, aggregate=False):
    """
    """
    plt.clf()
    plt.close()
    fig, axarr = plt.subplots(nrows=len(params), ncols=1, sharex=True, figsize=(4, 3))

    if aggregate:
        colours = cm.get_cmap('Set1')(np.linspace(0.2, 0.8, len(replace_indices)))
        assert n_seeds > 1

        for i, replace_index in enumerate(replace_indices):
            vary_S = results_utils.get_posterior_samples(dataset, iter_range, model,
                                                         replace_index=replace_index,
                                                         params=params, seeds='all',
                                                         n_seeds=n_seeds, diffinit=diffinit)
            vary_S_min = vary_S.groupby('t').min()
            vary_S_std = vary_S.groupby('t').std()
            vary_S_max = vary_S.groupby('t').max()
            vary_S_mean = vary_S.groupby('t').mean()

            for j, p in enumerate(params):
                axarr[j].fill_between(vary_S_min.index, vary_S_min[p], vary_S_max[p],
                                      alpha=0.1, color=colours[i], label='_legend_')
                axarr[j].fill_between(vary_S_mean.index,
                                      vary_S_mean[p] - vary_S_std[p],
                                      vary_S_mean[p] + vary_S_std[p],
                                      alpha=0.1, color=colours[i],
                                      label='_nolegend_', linestyle='--')
                axarr[j].plot(vary_S_min.index, vary_S_mean[p],
                              color=colours[i], alpha=0.7,
                              label='D -' + str(replace_index))
                axarr[j].set_ylabel('weight ' + p)
    else:
        colours = cm.get_cmap('plasma')(np.linspace(0.2, 0.8, n_seeds))
        assert len(replace_indices) == 1
        replace_index = replace_indices[0]
        vary_S = results_utils.get_posterior_samples(dataset, iter_range, model,
                                                     replace_index=replace_index,
                                                     params=params, seeds='all',
                                                     n_seeds=n_seeds, diffinit=diffinit)
        seeds = vary_S['seed'].unique()

        for i, s in enumerate(seeds):
            vary_Ss = vary_S.loc[vary_S['seed'] == s, :]

            for j, p in enumerate(params):
                axarr[j].plot(vary_Ss['t'], vary_Ss[p], color=colours[i],
                              label='seed ' + str(s), alpha=0.8)

                if i == 0:
                    axarr[j].set_ylabel(r'$\mathbf{w}^{' + p[1:] + '}$')

    axarr[-1].set_xlabel('training steps')
    vis_utils.beautify_axes(np.array([axarr]))
    plt.tight_layout()
    plt.savefig(f'./plots/analyses/weight_trajectory_{dataset}.{model}.png')
    plt.savefig(f'./plots/analyses/weight_trajectory_{dataset}.{model}.pdf')

    return True


def weight_posterior(dataset, model, replace_indices=None, t=500, param='#0', overlay_normal=False):
    """
    show posterior of weight for two datasets, get all the samples
    """
    iter_range = (t, t+1)

    if replace_indices == 'random':
        print('Picking two *random* replace indices for this setting...')
        df = results_utils.get_available_results(dataset, model)
        replace_counts = df['replace'].value_counts()
        replaces = replace_counts[replace_counts > 2].index.values
        replace_indices = np.random.choice(replaces, 2, replace=False)
    assert len(replace_indices) == 2
    # now load the data!
    df_1 = results_utils.get_posterior_samples(dataset, iter_range, model,
                                               replace_index=replace_indices[0],
                                               params=[param], seeds='all')
    df_2 = results_utils.get_posterior_samples(dataset, iter_range, model,
                                               replace_index=replace_indices[1],
                                               params=[param], seeds='all')
    print(f'Loaded {df_1.shape[0]} and {df_2.shape[0]} samples respectively')
    fig, axarr = plt.subplots(nrows=1, ncols=1)
    n_bins = 25
    sns.distplot(df_1[param], ax=axarr, label='D - ' + str(replace_indices[0]),
                 kde=True, color='blue', bins=n_bins, norm_hist=True)
    sns.distplot(df_2[param], ax=axarr, label='D - ' + str(replace_indices[1]),
                 kde=True, color='red', bins=n_bins, norm_hist=True)
    # show the empirical means
    # this si sort of just debugging the sensitivity...

    if overlay_normal:
        mean_1 = df_1[param].mean()
        mean_2 = df_2[param].mean()
        std_1 = df_1[param].std()
        std_2 = df_2[param].std()
        normal_1 = np.random.normal(loc=mean_1, scale=std_1, size=1000)
        normal_2 = np.random.normal(loc=mean_2, scale=std_2, size=1000)
        sns.kdeplot(normal_1, color='blue', ax=axarr)
        sns.kdeplot(normal_2, color='red', ax=axarr)
    axarr.set_xlabel('weight ' + param)
    axarr.legend()
    axarr.set_ylabel('# runs')
    vis_utils.beautify_axes(np.array([axarr]))
    plt.savefig(f'./plots/analyses/weight_posterior.{dataset}.{model}.{param}.png')
    plt.savefig(f'./plots/analyses/weight_posterior.{dataset}.{model}.{param}.pdf')

    return True


def plot_delta_histogram(dataset, model, num_deltas='max', t=500,
                         include_bounds=False, xlim=None, ylim=None,
                         data_privacy='all', plot=True, multivariate=False):
    """
    num_deltas is the number of examples we're using to estimate the histograms
    """

    if multivariate:
        raise NotImplementedError('Multivariate plotting is not implemented')
    delta_histogram = dr.DeltaHistogram(dataset, model, num_deltas, t, data_privacy, multivariate)
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
    
    # plot
    plt.clf()
    plt.close()
    fig, axarr = plt.subplots(nrows=1, ncols=1, figsize=(4, 2.1))
    print('Plotting varying S... number of deltas:', vary_S.shape[0])
    sns.distplot(vary_S, ax=axarr, color=bolton_colour, label=r'$\Delta_S$', kde=True, norm_hist=True)
    print('Plotting varying r... number of deltas:', vary_r.shape[0])
    print('Plotting varying both... number of deltas:', vary_both.shape[0])
    sns.distplot(vary_r, ax=axarr, color=augment_colour, label=r'$\Delta_V^{fix}$', kde=True, norm_hist=True)
    sns.distplot(vary_both, ax=axarr,
                 color=both_colour,
                 label=r'$\Delta_{S+V}^{fix}$',
                 kde=True,
                 hist=False,
                 kde_kws={'linestyle': '--'})
    sns.distplot(vary_r_diffinit, ax=axarr,
                 color=augment_diffinit_colour,
                 label=r'$\Delta_V^{vary}$',
                 kde=True,
                 norm_hist=True)
    sns.distplot(vary_both_diffinit, ax=axarr,
                 color=both_diffinit_colour,
                 label=r'$\Delta_{S+V}^{vary}$',
                 kde=True,
                 hist=False,
                 kde_kws={'linestyle': ':', 'lw': 2})

    if include_bounds:
        assert model == 'logistic'
        lipschitz_constant = np.sqrt(2.0)
        _, batch_size, lr, _, N = em.get_experiment_details(dataset, model, verbose=True)
        wu_bound = test_private_model.compute_wu_bound(lipschitz_constant, t=t, N=N, batch_size=batch_size, eta=lr)
        axarr.axvline(x=wu_bound, ls='--', color=bolton_colour, label=r'$\hat{\Delta}_S$')
    axarr.legend()
    axarr.set_xlabel(r'$\|w - w^\prime\|$')
    axarr.set_ylabel('density')

    if xlim is not None:
        axarr.set_xlim(xlim)

    if ylim is not None:
        axarr.set_ylim(ylim)

    vis_utils.beautify_axes(np.array([axarr]))
    plt.tight_layout()

    plt.savefig('./plots/analyses/delta_histogram_' + dataset + '_' + data_privacy + '_' + model + '_t' + str(t) + '.png')
    plt.savefig('./plots/analyses/delta_histogram_' + dataset + '_' + data_privacy + '_' + model + '_t' + str(t) + '.pdf')

    return True


def plot_epsilon_distribution(dataset, model, t, delta, num_pairs,
                              which='both',
                              sensitivity_from='local', sharex=False,
                              variability_from='empirical', xlim=None, ylim=None,
                              data_privacy='all'):
    """
    overlay epsilon dist with and without diffinit
    which  takes values both, vary, fix
    """
    sens_var = dr.SensVar(dataset, model, data_privacy, t, num_pairs)
    df = sens_var.load(diffinit=False)
    df_diffinit = sens_var.load(diffinit=True)

    # now set it all up
    _, batch_size, eta, _, N = em.get_experiment_details(dataset, model)

    if delta is None:
        delta = 1.0/(N**2)
        print('Delta:', delta)

    if num_pairs is not None:
        if df.shape[0] > num_pairs:
            pick_rows = np.random.choice(df.shape[0], num_pairs, replace=False)
            df = df.iloc[pick_rows, :]

        if df_diffinit.shape[0] > num_pairs:
            pick_rows = np.random.choice(df_diffinit.shape[0], num_pairs, replace=False)
            df_diffinit = df_diffinit.iloc[pick_rows, :]

    if sensitivity_from == 'wu_bound':
        assert model == 'logistic'
        lipschitz_constant = np.sqrt(2)
        sensitivity = test_private_model.compute_wu_bound(lipschitz_constant, t, N, batch_size, eta, verbose=True)
        sensitivity_diffinit = sensitivity
        print('Wu sensitivity bound:', sensitivity)
    elif sensitivity_from == 'empirical':
        sensitivity = dr.estimate_sensitivity_empirically(dataset, model,
                                                          t, num_deltas='max',
                                                          diffinit=False)
        sensitivity_diffinit = dr.estimate_sensitivity_empirically(dataset, model,
                                                                   t, num_deltas='max',
                                                                   diffinit=True)
    else:
        sensitivity = df['sensitivity']
        sensitivity_diffinit = df_diffinit['sensitivity']
    c = np.sqrt(2 * np.log(1.25 / delta))

    if variability_from == 'local':
        variability = df['variability']
        variability_diffinit = df_diffinit['variability']
    else:
        variability = dr.estimate_variability(dataset, model, t, multivariate=False, diffinit=False)
        variability_diffinit = dr.estimate_variability(dataset, model, t, multivariate=False, diffinit=True)
    print('Sens size:', sensitivity.shape)
    print('Var size:', variability.shape)
    epsilon = c * sensitivity / variability
    epsilon_diffinit = c * sensitivity_diffinit / variability_diffinit
    epsilon.dropna(inplace=True)
    epsilon_diffinit.dropna(inplace=True)
    # now plot!
    print('Visualising with', epsilon.shape[0], 'and', epsilon_diffinit.shape[0], 'epsilon values')
    n_bins = 50
    figsize = (4, 2.1)

    if sharex:
        fig, axarr = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=figsize)
        axarr = np.array([axarr])
    else:
        fig, axarr = plt.subplots(nrows=2, ncols=1, sharex=False, figsize=figsize)
    kde = True

    if which in ['both', 'fix']:
        sns.distplot(epsilon, ax=axarr[0], label=r'$\epsilon^{fix}$',
                     color=augment_colour, bins=n_bins, norm_hist=True, kde=kde)

    if which in ['both', 'vary']:
        sns.distplot(epsilon_diffinit, ax=axarr[-1], label=r'$\epsilon^{vary}$',
                     color=augment_diffinit_colour, bins=n_bins, norm_hist=True, kde=kde)
    axarr[0].set_xlabel('')
    axarr[-1].set_xlabel(r'pairwise $\epsilon$')

    for ax in axarr:
        ax.set_ylabel('density')

    if which == 'both':
        for ax in axarr:
            ax.legend()

    if xlim is not None:
        for ax in axarr:
            ax.set_xlim(xlim)

    if ylim is not None:
        for ax in axarr:
            ax.set_ylim(ylim)
    plt.tight_layout()
    vis_utils.beautify_axes(axarr)
    plt.savefig('./plots/analyses/epsilon_distribution_' + str(dataset) + '_' + str(model) + '_' + sensitivity_from + '_' + which + '.png')
    plt.savefig('./plots/analyses/epsilon_distribution_' + str(dataset) + '_' + str(model) + '_' + sensitivity_from + '_' + which + '.pdf')

    return True


def plot_utility_curve(dataset, model, delta, t,
                       metric_to_report='binary_accuracy', verbose=True,
                       num_deltas='max', diffinit=False, num_experiments=50,
                       xlim=None, ylim=None, identifier=None, include_fix=False):
    """
    for a single model (this is how it is right now), plot
    performance v. epsilon at fixed delta
    for
    - noiseless
    - bolt-on
    - augmenting SGD (us) (+ including random initialisation)
    """
    plt.clf()
    plt.close()
    path = './fig_data/utility.' + str(dataset) + '.' + str(model) + '.t' + str(t) + '.nd_' + str(num_deltas) + '.ne_' + str(num_experiments) + '.csv'
    try:
        utility_data = pd.read_csv(path)
        print('Loaded from', path)
    except FileNotFoundError:
        print('Couldn\'t find', path)

        return False

    fig, axarr = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(4, 2.1))

    if metric_to_report == 'binary_accuracy':
        label_stub = 'accuracy (binary)'
    elif metric_to_report == 'accuracy':
        label_stub = 'accuracy'
    else:
        raise ValueError(metric_to_report)
    # NOW FOR PLOTTING...!
    aggregate = False
    scale = False

    if not aggregate:
        if identifier is None:
            identifiers = utility_data.loc[:, ['replace', 'seed']].drop_duplicates()
            identifiers = identifiers.iloc[np.random.permutation(identifiers.shape[0]), :]
            identifier = identifiers.iloc[0]
            print('Picking identifier', identifier)
        else:
            assert 'replace' in identifier
            assert 'seed' in identifier
            print('Using identifier', identifier)
        utility_data = utility_data.loc[(utility_data['replace'] == identifier['replace']) & (utility_data['seed'] == identifier['seed']), :]
    noiseless_performance = utility_data['noiseless'].iloc[0]
    axarr.axhline(y=noiseless_performance, label='Noiseless', c='black', alpha=0.75)

    for j, sensitivity_from_bound in enumerate([False, True]):
        df = utility_data.loc[utility_data['sensitivity_from_bound'] == sensitivity_from_bound, :].drop(columns='sensitivity_from_bound')

        if df.shape[0] == 0:
            print('No data for setting', sensitivity_from_bound, '-skipping')

            continue

        if scale:
            maxx = df['noiseless'].copy()
            minn = df['bolton'].copy()
            df['bolton'] = (df['bolton'] - minn)/(maxx - minn)
            df['augment'] = (df['augment'] - minn)/(maxx - minn)
            df['augment_diffinit'] = (df['augment_diffinit'] - minn)/(maxx - minn)
            df['noiseless'] = (df['noiseless'] - minn)/(maxx - minn)

        if aggregate:
            df_mean = df.groupby('epsilon').mean()
            df_std = df.groupby('epsilon').std()
            df_min = df.groupby('epsilon').min()
            df_max = df.groupby('epsilon').max()
            # diff experiments!
            #noiseless
            axarr[j].plot(df_mean.index, df_mean['noiseless'],
                          label='noiseless', alpha=0.5, c='black')
            axarr[j].fill_between(df_mean.index, df_min['noiseless'], df_max['noiseless'],
                                  label='_nolegend_', alpha=0.1, color='black')
            # bolton
            axarr[j].plot(df_mean.index, df_mean['bolton'],
                          label='bolton', alpha=0.5, c=bolton_colour)
            axarr[j].fill_between(df_mean.index, df_min['bolton'], df_max['bolton'],
                                  label='_nolegend_', alpha=0.1, color=bolton_colour)

            if include_fix:
                # augment
                axarr[j].plot(df_mean.index, df_mean['augment'],
                              label='augment', alpha=0.5, c=augment_colour)
                axarr[j].fill_between(df_mean.index, df_min['augment'], df_max['augment'],
                                      label='_nolegend_', alpha=0.1, color=augment_colour)
            # augment with diffinit
            axarr[j].plot(df_mean.index, df_mean['augment_diffinit'],
                          label='augment_diffinit', alpha=0.5, c=augment_diffinit_colour)
            axarr[j].fill_between(df_mean.index, df_min['augment_diffinit'], df_max['augment_diffinit'],
                                  label='_nolegend_', alpha=0.1, color=augment_diffinit_colour)
        else:
            linestyle = '--' if sensitivity_from_bound is False else '-'
            size = 6
            line_alpha = 0.75
            axarr.scatter(df['epsilon'], df['bolton'], label='_nolegend_',
                          s=size, c=bolton_colour)
            axarr.plot(df['epsilon'], df['bolton'],
                       label=r'$\sigma_{target}$' if j == 1 else '_nolegend_',
                       alpha=line_alpha, c=bolton_colour, ls=linestyle)

            if include_fix:
                axarr.scatter(df['epsilon'], df['augment'],
                              label='_nolegend_', s=size, c=augment_colour)
                axarr.plot(df['epsilon'], df['augment'],
                           label=r'$\sigma_{augment}^{fix}$' if j == 1 else '_nolegend_',
                           alpha=line_alpha, c=augment_colour, ls=linestyle)
            axarr.scatter(df['epsilon'], df['augment_diffinit'],
                          label='_nolegend_', s=size, c=augment_diffinit_colour)
            axarr.plot(df['epsilon'], df['augment_diffinit'],
                       label=r'$\sigma_{augment}$' if j == 1 else '_nolegend_',
                       alpha=line_alpha, c=augment_diffinit_colour, ls=linestyle)
    axarr.legend()
    axarr.set_ylabel(label_stub)
    axarr.set_xlabel(r'$\epsilon$')

    if xlim is not None:
        axarr.set_xlim(xlim)

    if ylim is not None:
        axarr.set_ylim(ylim)
    vis_utils.beautify_axes(np.array([axarr]))
    plt.tight_layout()
    print('Reminder, the identifier was', identifier)
    plt.savefig('./plots/analyses/utility_' + str(dataset) + '_' + str(model) + '_withfix'*include_fix + '.png')
    plt.savefig('./plots/analyses/utility_' + str(dataset) + '_' + str(model) + '_withfix'*include_fix + '.pdf')

    return True


def plot_sens_and_var_over_time(dataset, model, num_deltas=500, iter_range=(0, 1000),
                                data_privacy='all', metric='binary_crossentropy', acc_lims=None):
    """
    Estimate the empirical (and theoretical I guess) sensitivity and variability v. "convergence point" (time)
    The objective is to create a CSV with columns:
    - convergence point
    - train loss
    - vali loss
    - theoretical sensitivity
    - empirical sensitivity
    - variability w/out diffinit
    - variability with diffinit
    ... and then plot that, basically
    """
    path = 'fig_data/v_time.' + dataset + '.' + data_privacy + '.' + model + '.nd_' + str(num_deltas) + '.csv'
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print('Didn\'t find', path)

        return False
    df = df.loc[df['t'] <= iter_range[1], :]
    df = df.loc[df['t'] >= iter_range[0], :]
    fig, axarr = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(3.4, 4))
    # losses
    train_loss = df[metric + '_mean_train']
    vali_loss = df[metric + '_mean_vali']
    train_loss_std = df[metric + '_std_train']
    vali_loss_std = df[metric + '_std_vali']
    axarr[0].scatter(df['t'], train_loss, label='_nolegend_', s=6, c='black')
    axarr[0].plot(df['t'], train_loss, alpha=0.5, label='train', c='black')
    axarr[0].fill_between(df['t'], train_loss - train_loss_std, train_loss + train_loss_std,
                          label='_nolegend_', alpha=0.2, color='black')
    axarr[0].scatter(df['t'], vali_loss, label='_nolegend_', s=6, c='grey')
    axarr[0].plot(df['t'], vali_loss, ls='--', label='validation', alpha=0.5, c='grey')
    axarr[0].fill_between(df['t'], vali_loss - vali_loss_std, vali_loss + vali_loss_std,
                          label='_nolegend_', alpha=0.2, color='grey')
    axarr[0].set_ylabel(re.sub('_', '\n', metric))

    if acc_lims is not None:
        axarr[0].set_ylim(acc_lims)
    # sensitivity (discretise it)
    ds = [np.nan]*df.shape[0]

    for i, ts in enumerate(df['theoretical_sensitivity'].values):
        ds[i] = test_private_model.discretise_theoretical_sensitivity(dataset, model, ts)
    axarr[1].plot(df['t'], ds, label='theoretical', alpha=0.5, c=bolton_colour, ls='--')
    axarr[1].scatter(df['t'], df['empirical_sensitivity'], label='_nolegend_', s=6, c=bolton_colour)
    axarr[1].plot(df['t'], df['empirical_sensitivity'], label='empirical', alpha=0.5, c=bolton_colour)
    axarr[1].set_ylabel('sensitivity')
    # variability
    axarr[2].scatter(df['t'], df['variability_diffinit'], label='variable init', s=6, c=augment_diffinit_colour)
    axarr[2].plot(df['t'], df['variability_diffinit'], label='_nolegend_', alpha=0.5, c=augment_diffinit_colour)
    axarr[2].scatter(df['t'], df['variability_fixinit'], label='fixed init', s=6, c=augment_colour)
    axarr[2].plot(df['t'], df['variability_fixinit'], label='_nolegend_', alpha=0.5, c=augment_colour)
    axarr[2].set_ylabel(r'$\sigma_i(\mathcal{D})$')
    # shared things
    axarr[-1].set_xlabel('T (number of steps)')

    if model == 'logistic':
        convergence_points = em.lr_convergence_points
        title = em.dataset_names[dataset] + ' (logistic regression)'
    else:
        convergence_points = em.nn_convergence_points
        title = em.dataset_names[dataset] + ' (neural network)'
    convergence_point = convergence_points[dataset]

    for ax in axarr:
        ax.axvline(x=convergence_point, ls='--', color='black', alpha=0.5)
        ax.legend()
    axarr[0].set_title(title)

    for ax in axarr:
        ax.legend()
    vis_utils.beautify_axes(np.array([axarr]))
    # save
    plt.tight_layout()
    plt.savefig('plots/analyses/v_time.' + dataset + '.' + data_privacy + '.' + model + '.nd_' + str(num_deltas) + '.png')
    plt.savefig('plots/analyses/v_time.' + dataset + '.' + data_privacy + '.' + model + '.nd_' + str(num_deltas) + '.pdf')
    plt.clf()
    plt.close()

    return True


def find_different_datasets(dataset, model, num_deltas, t):
    delta_histogram = dr.DeltaHistogram(dataset, model, num_deltas, t)
    plot_data = delta_histogram.load()
    if plot_data is None:
        print('[find_different_datasets] ERROR: Run delta_histogram for this setting first')

        return None, None
    vary_data_deltas = plot_data['vary_S']
    vary_data_identifiers = plot_data['S_identifiers']
    # just get the top 10 biggest
    biggest_idx = np.argsort(-vary_data_deltas)[:10]
    biggest_deltas = vary_data_deltas[biggest_idx]
    biggest_identifiers = vary_data_identifiers[biggest_idx]

    return biggest_deltas, biggest_identifiers


def plot_stability_of_estimated_values(dataset, model, t):
    """
    """
    stability = dr.Stability(dataset, model, t)
    stability_dict = stability.load()

    # lets just do 3 separate plots
    figsize = (3.5, 2.8)
    size = 6
    # SIGMA V N SEEDS
    print('Plotting sigma v seeds')
    sigma_df = stability_dict['sigma']
    sigma_v_seed = sigma_df[['n_seeds', 'sigma']]
#    sigma_v_seed = sigma_df[sigma_df['n_replaces'] == sigma_df['n_replaces'].max()][['n_seeds', 'sigma']]
    fig, axarr = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    axarr.scatter(sigma_v_seed['n_seeds'], sigma_v_seed['sigma'], s=size, c=augment_diffinit_colour)
    sigma_we_use = dr.estimate_variability(dataset, model, t, diffinit=True)
    axarr.axhline(y=sigma_we_use, ls='--', c=augment_diffinit_colour, alpha=0.4)
    axarr.set_xlabel('number of random seeds')
    axarr.set_ylabel(r'estimated $\sigma_i(\mathcal{D})$')
    axarr.set_title(em.dataset_names[dataset] + ' (' + em.model_names[model] + ')')
    upper_y = 1.05*max(np.max(sigma_v_seed['sigma']), sigma_we_use)
    lower_y = 0.95*np.min(sigma_v_seed['sigma'])
    axarr.set_ylim(lower_y, upper_y)
    vis_utils.beautify_axes(np.array([axarr]))
    plt.tight_layout()
    plt.savefig('./plots/analyses/stability_sigma_v_seeds_' + dataset + '_' + model + '_t' + str(t) + '.png')
    plt.savefig('./plots/analyses/stability_sigma_v_seeds_' + dataset + '_' + model + '_t' + str(t) + '.pdf')
    plt.clf()
    plt.close()
    # with fixed num_deltas, sensitivity
    print('Plotting sens v num deltas')
    sens_df = stability_dict['sens']
    sens_v_deltas = sens_df[['n_deltas', 'sens']].drop_duplicates()
    fig, axarr = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    axarr.scatter(sens_v_deltas['n_deltas'], sens_v_deltas['sens'], s=size, c=bolton_colour)
    sens_we_use = dr.estimate_sensitivity_empirically(dataset, model, t,
                                                      num_deltas='max', diffinit=True,
                                                      data_privacy='all')
    axarr.axhline(y=sens_we_use, ls='--', c=bolton_colour, alpha=0.4)
    axarr.set_xlabel('number of dataset comparisons')
    axarr.set_ylabel('estimated sensitivity')
    axarr.set_ylim(0, None)
    axarr.set_xscale('log')
    axarr.set_title(em.dataset_names[dataset] + ' (' + em.model_names[model] + ')')
    vis_utils.beautify_axes(np.array([axarr]))
    plt.tight_layout()
    plt.savefig('./plots/analyses/stability_sens_v_deltas_' + dataset + '_' + model + '_t' + str(t) + '.png')
    plt.savefig('./plots/analyses/stability_sens_v_deltas_' + dataset + '_' + model + '_t' + str(t) + '.pdf')
    plt.clf()
    plt.close()

    return True


def plot_sigmas_distribution(model, datasets=None, ylim=None):
    if model == 'logistic':
        convergence_points = em.lr_convergence_points
        title = 'Logistic regression'
    else:
        convergence_points = em.nn_convergence_points
        title = 'Neural network'

    if datasets is None:
        datasets = convergence_points.keys()
    fig, axarr = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))

    for ds in datasets:
        t = convergence_points[ds]
        # now just the sigmas distribution
        all_sigmas = dr.Sigmas(ds, model, t).load(diffinit=True)['sigmas']
        # lose the nans
        all_sigmas = all_sigmas[~np.isnan(all_sigmas)]
        min_sigma = np.nanmin(all_sigmas)
        sns.distplot(all_sigmas - min_sigma, ax=axarr, norm_hist=True,
                     label=em.dataset_names[ds], color=to_rgba(em.dataset_colours[ds]),
                     kde=False, bins=50)
        percentiles = np.percentile(all_sigmas, [0, 0.25, 0.5, 0.75, 1])
        print(ds, len(all_sigmas))
        print(percentiles)

    axarr.set_xlabel('variability estimate')
    axarr.set_ylabel('density')
    axarr.legend()

    if ylim is not None:
        axarr.set_ylim(ylim)
    axarr.set_xlim(0, None)
    vis_utils.beautify_axes(np.array([axarr]))
    plt.tight_layout()
    plt.savefig('./plots/analyses/stability_sigmas_dist__' + str(model) + '.png')
    plt.savefig('./plots/analyses/stability_sigmas_dist__' + str(model) + '.pdf')
    plt.clf()
    plt.close()

    return True


def compute_sigma_v_n_seeds(dataset, model, t):
    """
    """
    n_seeds_array = []
    n_replaces_array = []
    sigma_array = []

    for n_seeds in [2, 5, 10]*5 + [20, 30]*3 + [40, 50]*2 + [60, 70, 80, 90, 100, 200]:
        for n_replaces in [25]:
            # [50]:      # this is what it is for mnist (MLP)
                        # [75]: # this is what it is for adult and forest )LR)
            #        [100]: # this is what it is for the others (LR)
            # setting ephemeral = True will make this very slow but I think it's worth it for my sanity
            # otherwise I need to do even more refactoring
            sigma = dr.estimate_variability(dataset, model, t=t,
                                            n_seeds=n_seeds, n_replaces=n_replaces,
                                            ephemeral=True, diffinit=True)
            n_seeds_array.append(n_seeds)
            n_replaces_array.append(n_replaces)
            sigma_array.append(sigma)
            print(f'{n_replaces} replaces, {n_seeds} seeds')
            print(f'\tsigma: {sigma}')

    stability_sigma = pd.DataFrame({'n_seeds': n_seeds_array,
                                    'n_replaces': n_replaces_array,
                                    'sigma': sigma_array})

    return stability_sigma


def compute_sens_v_n_deltas(dataset, model, t):
    """
    compute empirical
    - sens
    - variability
    - epsilon
    with differing numbers of experiments, to test stability of estimates
    """
    num_deltas_array = []
    sens_array = []

    for n_deltas in [5, 10, 25, 50, 75, 100, 125, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000]:
        vary_S, _ = dr.get_deltas(dataset, iter_range=(t, t+1),
                                  model=model, vary_seed=False, vary_data=True,
                                  num_deltas=n_deltas, diffinit=True,
                                  data_privacy='all', multivariate=False)

        if vary_S is False:
            sens = None
        else:
            print('should have', n_deltas, 'deltas, actually have:', len(vary_S[~np.isnan(vary_S)]))
            sens = np.nanmax(vary_S)
        print(f'{n_deltas} deltas')
        print(f'\tsens: {sens}')
        num_deltas_array.append(n_deltas)
        sens_array.append(sens)
    stability_sens = pd.DataFrame({'n_deltas': num_deltas_array,
                                   'sens': sens_array})

    return stability_sens


def compare_mnist_variants():
    """
    visualise distribution of feature values across the three mnist datasets
    """
    colours = cm.get_cmap('Set1')(np.arange(3))
    # ---- compare the distributions ---- #
    mnist_binary_pca, _, _, _, _, _ = data_utils.load_data('mnist_binary_pca')
    mnist_binary_grp, _, _, _, _, _ = data_utils.load_data('mnist_binary')
    mnist_binary_cropped, _, _, _, _, _ = data_utils.load_data('mnist_binary_cropped')
    print(f'cropped sparsity: {np.mean(mnist_binary_cropped.flatten() == 0)}')
    fig, axarr = plt.subplots(nrows=1, ncols=1, figsize=(3, 2.5))
    sns.distplot(mnist_binary_pca.flatten(), ax=axarr, label='PCA',
                 norm_hist=True, kde=False, color=to_rgba(colours[0]))
    sns.distplot(mnist_binary_grp.flatten(), ax=axarr, label='GRP',
                 norm_hist=True, kde=False, color=to_rgba(colours[1]))
    sns.distplot(mnist_binary_cropped.flatten(), ax=axarr, label='Crop',
                 norm_hist=True, kde=False, color=to_rgba(colours[2]))
    axarr.set_xlabel('feature value')
    axarr.set_ylabel('density')
    axarr.legend()
    axarr.set_ylim(0, 5)
    axarr.set_xlim(-0.6, 0.6)
    vis_utils.beautify_axes(np.array([axarr]))
    plt.tight_layout()
    plt.savefig('./plots/analyses/mnist_preprocessing_comparison.png')
    plt.savefig('./plots/analyses/mnist_preprocessing_comparison.pdf')
    plt.clf()
    plt.close()
    # ------- compare the traces ----- #
    pca_loss = dr.AggregatedLoss('mnist_binary_pca', 'logistic').load(diffinit=True)
    grp_loss = dr.AggregatedLoss('mnist_binary', 'logistic').load(diffinit=True)
    cropped_loss = dr.AggregatedLoss('mnist_binary_cropped', 'logistic').load(diffinit=True)
    fig, axarr = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(3.5, 3.5))
    size = 6
    # first row is crossnetropy
    axarr[0].scatter(pca_loss.index, pca_loss['binary_crossentropy_mean_train'], label='PCA', s=size, color=colours[0])
    axarr[0].scatter(grp_loss.index, grp_loss['binary_crossentropy_mean_train'], label='GRP', s=size, color=colours[1])
    axarr[0].scatter(cropped_loss.index, cropped_loss['binary_crossentropy_mean_train'],
                     label='Crop', s=size, color=colours[2])
    axarr[0].plot(pca_loss.index, pca_loss['binary_crossentropy_mean_train'],
                  label='_nolegend_', color=colours[0], alpha=0.5)
    axarr[0].plot(grp_loss.index, grp_loss['binary_crossentropy_mean_train'],
                  label='_nolegend_', color=colours[1], alpha=0.5)
    axarr[0].plot(cropped_loss.index, cropped_loss['binary_crossentropy_mean_train'],
                  label='_nolegend_', color=colours[2], alpha=0.5)
    axarr[0].fill_between(pca_loss.index,
                          pca_loss['binary_crossentropy_mean_train'] - pca_loss['binary_crossentropy_std_train'],
                          pca_loss['binary_crossentropy_mean_train'] + pca_loss['binary_crossentropy_std_train'],
                          label='_nolegend_', alpha=0.2, color=colours[0])
    axarr[0].fill_between(grp_loss.index,
                          grp_loss['binary_crossentropy_mean_train'] - grp_loss['binary_crossentropy_std_train'],
                          grp_loss['binary_crossentropy_mean_train'] + grp_loss['binary_crossentropy_std_train'],
                          label='_nolegend_', alpha=0.2, color=colours[1])
    axarr[0].fill_between(cropped_loss.index,
                          cropped_loss['binary_crossentropy_mean_train'] - cropped_loss['binary_crossentropy_std_train'],
                          cropped_loss['binary_crossentropy_mean_train'] + cropped_loss['binary_crossentropy_std_train'],
                          label='_nolegend_', alpha=0.2, color=colours[2])
    # second row is accuracy
    axarr[1].scatter(pca_loss.index, pca_loss['binary_accuracy_mean_train'], label='PCA', s=size, color=colours[0])
    axarr[1].scatter(grp_loss.index, grp_loss['binary_accuracy_mean_train'], label='GRP', s=size, color=colours[1])
    axarr[1].scatter(cropped_loss.index, cropped_loss['binary_accuracy_mean_train'],
                     label='Crop', s=size, color=colours[2])
    axarr[1].plot(pca_loss.index, pca_loss['binary_accuracy_mean_train'],
                  label='_nolegend_', color=colours[0], alpha=0.5)
    axarr[1].plot(grp_loss.index, grp_loss['binary_accuracy_mean_train'],
                  label='_nolegend_', color=colours[1], alpha=0.5)
    axarr[1].plot(cropped_loss.index, cropped_loss['binary_accuracy_mean_train'],
                  label='_nolegend_', color=colours[2], alpha=0.5)
    axarr[1].fill_between(pca_loss.index,
                          pca_loss['binary_accuracy_mean_train'] - pca_loss['binary_accuracy_std_train'],
                          pca_loss['binary_accuracy_mean_train'] + pca_loss['binary_accuracy_std_train'],
                          label='_nolegend_', alpha=0.2, color=colours[0])
    axarr[1].fill_between(grp_loss.index,
                          grp_loss['binary_accuracy_mean_train'] - grp_loss['binary_accuracy_std_train'],
                          grp_loss['binary_accuracy_mean_train'] + grp_loss['binary_accuracy_std_train'],
                          label='_nolegend_', alpha=0.2, color=colours[1])
    axarr[1].fill_between(cropped_loss.index,
                          cropped_loss['binary_accuracy_mean_train'] - cropped_loss['binary_accuracy_std_train'],
                          cropped_loss['binary_accuracy_mean_train'] + cropped_loss['binary_accuracy_std_train'],
                          label='_nolegend_', alpha=0.2, color=colours[2])
    # labels and stuff
    axarr[0].legend()
    axarr[0].set_ylabel('crossentropy')
    axarr[1].set_ylabel('accuracy')
    axarr[-1].set_xlabel('T (number of steps)')
    # limits and convergence
    axarr[0].set_xlim(0, 2100)
    axarr[1].set_xlim(0, 2100)
    axarr[0].axvline(x=1850, ls='--', color='black', alpha=0.5)
    axarr[1].axvline(x=1850, ls='--', color='black', alpha=0.5)
    axarr[1].set_ylim(0.75, 0.96)
    # tidy
    vis_utils.beautify_axes(axarr)
    plt.tight_layout()
    plt.savefig('./plots/analyses/mnist_preprocessing_comparison_trace.png')
    plt.savefig('./plots/analyses/mnist_preprocessing_comparison_trace.pdf')
    plt.clf()
    plt.close()

    return True


def overlay_pval_plot(model='logistic', xlim=None, n_experiments=50,
                      datasets=None, ylim=None):
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

    if datasets is None:
        datasets = em.dataset_colours.keys()
        plot_label = '_'
    else:
        plot_label = '_'.join(datasets) + '_'
    fig, axarr = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    vertical_lines_we_already_have = set()

    for ds in datasets:
        print(ds)
        log_pvals, n_params = vis_utils.fit_pval_histogram(what=what, dataset=ds, model=model,
                                                           t=convergence_points[ds],
                                                           n_experiments=n_experiments,
                                                           plot=False)
        sns.distplot(log_pvals, kde=True, bins=min(100, int(len(log_pvals)*0.25)),
                     ax=axarr, color=em.dataset_colours[ds], norm_hist=True,
                     label=em.dataset_names[ds],
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
    plt.savefig('plots/analyses/pval_histogram_' + plot_label + model + '.png')
    plt.savefig('plots/analyses/pval_histogram_' + plot_label + model + '.pdf')

    return True


def overlay_eps_plot(model='logistic', datasets=None, xlim=None, ylim=None, title=None):
    figsize = (3.7, 2.9)
    n_bins = 50

    if model == 'logistic':
        convergence_points = em.lr_convergence_points
    else:
        convergence_points = em.nn_convergence_points

    if datasets is None:
        datasets = em.dataset_colours.keys()
        plot_label = '_'
    else:
        plot_label = '_'.join(datasets) + '_'
    fig, axarr = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    for ds in datasets:
        _, eps_diffinit = dr.epsilon_distribution(ds, model, t=convergence_points[ds],
                                                  delta=None, num_pairs=None, which='vary',
                                                  sensitivity_from='local', sharex=True,
                                                  variability_from='empirical',
                                                  xlim=None, ylim=None,
                                                  data_privacy='all', plot=False)
        sns.distplot(eps_diffinit, ax=axarr, label=em.dataset_names[ds],
                     color=em.dataset_colours[ds], bins=n_bins, norm_hist=True, kde=True)
    axarr.legend()
    axarr.set_ylabel('density')
    axarr.set_xlabel(r'pairwise $\epsilon_i(\mathcal{D})$')

    if title is not None:
        axarr.set_title(title)

    if ylim is not None:
        axarr.set_ylim(ylim)

    if xlim is not None:
        axarr.set_xlim(xlim)
    vis_utils.beautify_axes(np.array([axarr]))
    plt.tight_layout()
    plt.savefig('plots/analyses/epsilon_distribution_' + plot_label + model + '.png')
    plt.savefig('plots/analyses/epsilon_distribution_' + plot_label + model + '.pdf')
    plt.clf()
    plt.close()

    return True
