#!/usr/bin/env ipython # This script is for performing specific analyses/generating figures
# It relies on e.g. statistics computed across experiments - "derived results"

import numpy as np
import pandas as pd
import vis_utils
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_rgba, to_hex
from pathlib import Path
import seaborn as sns
import statsmodels.api as sm
import re
import ipdb
import umap
import test_private_model
import results_utils
import derived_results as dr
import experiment_metadata as em

plt.switch_backend('Agg')
params = {'font.family': 'sans-serif',
          'font.size': 10}
plt.rcParams.update(params)

# Separate to figures, these are more throw-away type plots
PLOTS_DIR = Path('./visualisations/')


def weight_evolution(cfg_name, model, n_seeds=50, replace_indices=None,
                     iter_range=(None, None), params=['#4', '#2'],
                     diffinit=False, aggregate=False):
    plt.clf()
    plt.close()
    fig, axarr = plt.subplots(nrows=len(params), ncols=1, sharex=True, figsize=(4, 3))

    if aggregate:
        colours = cm.get_cmap('Set1')(np.linspace(0.2, 0.8, len(replace_indices)))
        assert n_seeds > 1

        for i, replace_index in enumerate(replace_indices):
            vary_S = results_utils.get_posterior_samples(cfg_name, iter_range, model,
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
        vary_S = results_utils.get_posterior_samples(cfg_name, iter_range, model,
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
    plot_identifier = f'weight_trajectory_{cfg_name}.{model}'
    plt.savefig((PLOTS_DIR / plot_identifier).with_suffix('.png'))
    plt.savefig((PLOTS_DIR / plot_identifier).with_suffix('.pdf'))

    return


def weight_posterior(cfg_name, model, replace_indices='random', t=500,
                     param='#0', sort=False, n_bins=25):
    """
    """
    if sort and not model == 'mlp':
        print(f'WARNING: Sort = True is not implemented/meaningful for model {model}')

    iter_range = (t, t+1)
    nolegend = False

    if replace_indices == 'random':
        print('Picking two *random* replace indices for this setting...')
        df = results_utils.get_available_results(cfg_name, model)
        replace_counts = df['replace'].value_counts()
        replaces = replace_counts[replace_counts > 2].index.values
        replace_indices = np.random.choice(replaces, 2, replace=False).tolist()
    elif type(replace_indices) == int:
        replace_indices = [replace_indices]
        nolegend = True

    assert type(replace_indices) == list
    # Set up the plot
    fig, axarr = plt.subplots(nrows=1, ncols=1, figsize=(4, 2.5))
    # now load the data!
    for replace_index in replace_indices:
        df = results_utils.get_posterior_samples(cfg_name, iter_range, model,
                                                 replace_index=replace_index,
                                                 params=[param], seeds='all',
                                                 sort=sort)
        sns.distplot(df[param], ax=axarr, label=f'{replace_index}',
                     kde=True, bins=n_bins, norm_hist=True)

    axarr.set_xlabel('weight ' + param)
    if not nolegend:
        axarr.legend()
    axarr.set_ylabel('# runs')
    vis_utils.beautify_axes(np.array([axarr]))
    plt.tight_layout()

    plot_identifier = f'weight_posterior_{cfg_name}_{param}'
    if sort:
        plot_identifier = f'{plot_identifier}_sorted'
    plt.savefig((PLOTS_DIR / plot_identifier).with_suffix('.png'))
    plt.savefig((PLOTS_DIR / plot_identifier).with_suffix('.pdf'))

    return


def plot_utility_curve(cfg_name, model, delta, t,
                       metric_to_report='binary_accuracy', verbose=True,
                       num_deltas='max', diffinit=False, num_experiments=50,
                       xlim=None, ylim=None, identifier=None, include_fix=False) -> None:
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
    try:
        utility_data = dr.UtilityCurve(cfg_name, model, num_deltas=num_deltas, t=t).load(diffinit=diffinit)
    except FileNotFoundError:
        return

    fig, axarr = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(4, 2.1))

    if metric_to_report == 'binary_accuracy':
        label_stub = 'accuracy (binary)'
    elif metric_to_report == 'accuracy':
        label_stub = 'accuracy'
    else:
        raise ValueError(metric_to_report)

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

        linestyle = '--' if sensitivity_from_bound is False else '-'
        size = 6
        line_alpha = 0.75
        axarr.scatter(df['epsilon'], df['bolton'], label='_nolegend_',
                      s=size, c=em.dp_colours['bolton'])
        axarr.plot(df['epsilon'], df['bolton'],
                   label=r'$\sigma_{target}$' if j == 1 else '_nolegend_',
                   alpha=line_alpha, c=em.dp_colours['bolton'], ls=linestyle)

        if include_fix:
            axarr.scatter(df['epsilon'], df['augment'],
                          label='_nolegend_', s=size, c=em.dp_colours['augment'])
            axarr.plot(df['epsilon'], df['augment'],
                       label=r'$\sigma_{augment}^{fix}$' if j == 1 else '_nolegend_',
                       alpha=line_alpha, c=em.dp_colours['augment'], ls=linestyle)
        axarr.scatter(df['epsilon'], df['augment_diffinit'],
                      label='_nolegend_', s=size, c=em.dp_colours['augment_diffinit'])
        axarr.plot(df['epsilon'], df['augment_diffinit'],
                   label=r'$\sigma_{augment}$' if j == 1 else '_nolegend_',
                   alpha=line_alpha, c=em.dp_colours['augment_diffinit'], ls=linestyle)
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

    plot_identifier = f'utility_{cfg_name}_{model}_{"_withfix"*include_fix}'
    plt.savefig((PLOTS_DIR / plot_identifier).with_suffix('.png'))
    plt.savefig((PLOTS_DIR / plot_identifier).with_suffix('.pdf'))

    return


def plot_sigmas_distribution(model, cfg_names=None, ylim=None) -> None:
    if model == 'logistic':
        convergence_points = em.lr_convergence_points
        title = 'Logistic regression'
    else:
        convergence_points = em.nn_convergence_points
        title = 'Neural network'

    if cfg_names is None:
        cfg_names = convergence_points.keys()
    fig, axarr = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))

    for ds in cfg_names:
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
    axarr.set_title(title)
    axarr.legend()

    if ylim is not None:
        axarr.set_ylim(ylim)
    axarr.set_xlim(0, None)
    vis_utils.beautify_axes(np.array([axarr]))
    plt.tight_layout()

    plot_identifier = f'stability_sigmas_dist_{model}'
    plt.savefig((PLOTS_DIR / plot_identifier).with_suffix('.png'))
    plt.savefig((PLOTS_DIR / plot_identifier).with_suffix('.pdf'))
    plt.clf()
    plt.close()

    return


def qq_plot(what: str, cfg_name: str, model: str, replace_index: int, seed: int, times=[50], params='random') -> None:
    """
    grab trace file, do qq plot for gradient noise at specified time-point
    """
    plt.clf()
    plt.close()
    assert what in ['gradients', 'weights']

    if what == 'weights':
        print('Looking at weights, this means we consider all seeds!')
    colours = cm.viridis(np.linspace(0.2, 0.8, len(times)))

    experiment = results_utils.ExperimentIdentifier(cfg_name, model, replace_index, seed)

    if params == 'random':
        if what == 'gradients':
            df = experiment.load_gradients(noise=True, params=None, iter_range=(min(times), max(times)+1))
        else:
            df = results_utils.get_posterior_samples(cfg_name, model=model,
                                                     replace_index=replace_index,
                                                     iter_range=(min(times), max(times) + 1),
                                                     params=None)
        params = np.random.choice(df.columns[2:], 1)
        print('picking random parameter', params)
        first_two_cols = df.columns[:2].tolist()
        df = df.loc[:, first_two_cols + list(params)]
    else:
        if what == 'gradients':
            df = experiment.load_gradients(noise=True, params=params, iter_range=(min(times), max(times)+1))
        else:
            df = results_utils.get_posterior_samples(cfg_name, model=model,
                                                     replace_index=replace_index,
                                                     iter_range=(min(times), max(times) + 1),
                                                     params=params)

    if df is False:
        print('ERROR: No data available')

        return False
    fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(7, 3.5))

    for i, t in enumerate(times):
        df_t = df.loc[df['t'] == t, :]
        X = df_t.iloc[:, 2:].values.flatten()
        print('number of samples:', X.shape[0])
        sns.distplot(X, ax=axarr[0], kde=False, color=to_hex(colours[i]), label=str(t))
        sm.qqplot(X, line='45', fit=True, ax=axarr[1], c=colours[i], alpha=0.5, label=str(t))
    plt.suptitle('cfg_name: ' + cfg_name + ', model:' + model + ',' + what)
    axarr[0].legend()
    axarr[1].legend()
    axarr[0].set_xlabel('parameter:' + '.'.join(params))
    vis_utils.beautify_axes(axarr)
    plt.tight_layout()

    plot_identifier = f'qq_{what}_{cfg_name}_{model}_{"_".join(params)}'
    plt.savefig((PLOTS_DIR / plot_identifier).with_suffix('.png'))
    plt.savefig((PLOTS_DIR / plot_identifier).with_suffix('.pdf'))

    return


def bivariate_gradients(cfg_name, model, replace_index, seed, df=None,
                        params=['#3', '#5'], iter_range=(None, None), n_times=2, save=False) -> None:
    print('Comparing gradients for parameters', params, 'at', n_times, 'random time-points')

    if df is None:
        experiment = results_utils.ExperimentIdentifier(cfg_name, model, replace_index, seed)
        df = experiment.load_gradients(noise=True, iter_range=iter_range, params=params)

    if params is None:
        params = np.random.choice(df.columns[2:], 2, replace=False)
    else:
        for p in params:
            assert p in df.columns

    # pick n_times to visualise
    times = df['t'].unique()
    times = sorted(np.random.choice(times, n_times, replace=False))
    print('Visualising at times:', times)

    # check pairwise correlation between gradients
    fig, axarr = plt.subplots(nrows=n_times, ncols=1, sharex='col')

    if n_times == 1:
        axarr = [axarr]

    for i, t in enumerate(times):
        dft = df.loc[df['t'] == t, :]
        axarr[i].set_title('iter:' + str(t))
        sns.kdeplot(dft[params[0]], dft[params[1]], shade=True, ax=axarr[i], shade_lowest=False, cmap='Reds')
        axarr[i].scatter(dft[params[0]], dft[params[1]], s=4, color='black', alpha=0.75)
        axarr[i].set(adjustable='box-forced', aspect='equal')
    vis_utils.beautify_axes(axarr)
    plt.tight_layout()

    if save:
        plot_identifier = f'gradient_pairs_{cfg_name}_{model}_replace{replace_index}_seed{seed}_params{"_".join(params)}'
        plt.savefig((PLOTS_DIR / plot_identifier).with_suffix('.png'))
        plt.savefig((PLOTS_DIR / plot_identifier).with_suffix('.pdf'))
        plt.clf()
        plt.close()

    return


def fit_pval_histogram(what, cfg_name, model, t, n_experiments=3, diffinit=False,
                       xlim=None, seed=1) -> None:
    """
    histogram of p-values (across parameters-?) for a given model etc.
    """
    assert what in ['weights', 'gradients']
    # set some stuff up
    iter_range = (t, t + 1)
    fig, axarr = plt.subplots(nrows=1, ncols=1, figsize=(3.5, 2.1))
    pval_colour = '#b237c4'
    # sample experiments
    df = results_utils.get_available_results(cfg_name, model, diffinit=diffinit)
    replace_indices = df['replace'].unique()
    replace_indices = np.random.choice(replace_indices, n_experiments, replace=False)
    print('Looking at replace indices...', replace_indices)
    all_pvals = []

    for i, replace_index in enumerate(replace_indices):
        experiment = results_utils.ExperimentIdentifier(cfg_name, model, replace_index, seed, diffinit)

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
            df_fit = dr.estimate_statistics_through_training(what=what, cfg_name=None,
                                                             model=None, replace_index=None,
                                                             seed=None,
                                                             df=df.loc[:, ['t', second_col, p]],
                                                             params=None, iter_range=None)
            p_vals[j] = df_fit.loc[t, 'norm_p']
            del df_fit
        log_pvals = np.log(p_vals)
        all_pvals.append(log_pvals)
    log_pvals = np.concatenate(all_pvals)

    if xlim is not None:
        # remove values below the limit
        number_below = (log_pvals < xlim[0]).sum()
        print('There are', number_below, 'p-values below the limit of', xlim[0])
        log_pvals = log_pvals[log_pvals > xlim[0]]
        print('Remaining pvals:', len(log_pvals))
    sns.distplot(log_pvals, kde=True, bins=min(100, int(len(log_pvals)*0.25)),
                 ax=axarr, color=pval_colour, norm_hist=True)
    axarr.axvline(x=np.log(0.05), ls=':', label='p = 0.05', color='black', alpha=0.75)
    axarr.axvline(x=np.log(0.05/n_params), ls='--', label='p = 0.05/' + str(n_params), color='black', alpha=0.75)
    axarr.legend()
    axarr.set_xlabel(r'$\log(p)$')
    axarr.set_ylabel('density')

    if xlim is not None:
        axarr.set_xlim(xlim)
    else:
        axarr.set_xlim((None, 0.01))
#    axarr.set_xscale('log')
    vis_utils.beautify_axes(np.array([axarr]))
    plt.tight_layout()
    plot_identifier = f'pval_histogram_{cfg_name}_{model}_{what}'
    plt.savefig((PLOTS_DIR / plot_identifier).with_suffix('.png'))
    plt.savefig((PLOTS_DIR / plot_identifier).with_suffix('.pdf'))

    return


def visualise_fits(cfg_name, model, replace_index, seed, save=True, params=None) -> None:
    print('Visualising distribution fits through training')
    if params is None:
        params = [None]

    fig, axarr = plt.subplots(nrows=3, ncols=1, sharex='col', figsize=(4, 5))

    n_comparators = len(params)
    colours = cm.viridis(np.linspace(0.2, 0.8, n_comparators))

    for i, p in enumerate(params):
        print('visualising fit for parameter parameter', p)
        df_fit = dr.estimate_statistics_through_training(what='gradients', cfg_name=cfg_name,
                                                         model=model, replace_index=replace_index,
                                                         seed=seed, params=[p])
        if df_fit is False:
            print('No fit data available')

            return
        iterations = df_fit.index
        color = to_hex(colours[i])
        axarr[0].scatter(iterations, df_fit['norm_sigma'], c=color, alpha=1, s=4, zorder=2, label=p)
        axarr[0].plot(iterations, df_fit['norm_sigma'], c=color, alpha=0.75, zorder=2, label='_nolegend_')
        axarr[0].set_ylabel('standard deviation')
        axarr[1].scatter(iterations, df_fit['norm_W'], c=color, alpha=1, s=4, zorder=2, label=p)
        axarr[1].plot(iterations, df_fit['norm_W'], c=color, alpha=0.75, zorder=2, label='_nolegend_')
        axarr[1].set_ylabel('test statistic\nshapiro-wilk')
        axarr[2].scatter(iterations, df_fit['norm_p'], c=color, alpha=1, s=4, zorder=2, label=p)
        axarr[2].plot(iterations, df_fit['norm_p'], c=color, alpha=0.75, zorder=2, label='_nolegend_')
        axarr[2].set_ylabel('log p-value')

    if (len(params) > 1) and (len(params) < 5):
        print(len(params), 'parameters - adding a legend')
        axarr[0].legend()
    axarr[1].set_ylim(0.9, 1)
    axarr[-1].set_yscale('log')
    axarr[-1].axhline(y=0.05, c='red', ls='--', label='p = 0.05')
    axarr[-1].set_xlabel('training iterations')
    vis_utils.beautify_axes(axarr)
    plt.tight_layout()

    if save:
        plot_label = model + '_replace' + str(replace_index) + '_seed' + str(seed) + '_'.join(params)
        plt.savefig('plots/' + cfg_name + '/' + plot_label + '_fits.png')
        plt.clf()
        plt.close()

    return


def visualise_trace(cfg_names, models, replaces, seeds, privacys, save=True,
                    include_batches=False, iter_range=(None, None),
                    include_convergence=True, diffinit=False, convergence_tolerance=3,
                    include_vali=True, labels=None, figsize=None) -> None:
    """
    Show the full training set loss as well as the gradient (at our element) over training
    """
    identifiers = vis_utils.process_identifiers(cfg_names, models, replaces, seeds, privacys)
    print(identifiers)

    if len(identifiers) > 1:
        print('WARNING: When more than one experiment is included, we turn off visualisation of batches to avoid cluttering the plot')
        include_batches = False

    if labels is None:
        labels = [f'{x["cfg_name"]}-{x["model"]}-{x["replace"]}-{x["seed"]}' for x in identifiers]
    else:
        assert len(labels) == len(identifiers)
    loss_list = []

    for identifier in identifiers:
        cfg_name = identifier['cfg_name']
        model = identifier['model']
        replace_index = identifier['replace']
        seed = identifier['seed']
        data_privacy = identifier['data_privacy']
        experiment = results_utils.ExperimentIdentifier(cfg_name, model, replace_index, seed,
                                                        data_privacy=data_privacy, diffinit=diffinit)
        df_loss = experiment.load_loss(iter_range=iter_range)

        if df_loss is False:
            print('No fit data available for identifier:', identifier)
            df_loss = []
        loss_list.append(df_loss)

    if len(loss_list) == 0:
        print('Error: no valid data')

        return False

    if include_batches:
        minibatch_ids = loss_list[0]['minibatch_id'].unique()
        colormap = dict(zip(minibatch_ids, cm.viridis(np.linspace(0, 1, len(minibatch_ids)))))
    colours = cm.viridis(np.linspace(0.2, 0.8, len(loss_list)))

    # what metrics were recorded for this run?
    metrics = loss_list[0].columns[2:]
    print('Visualising trace of', identifiers, 'with metrics', metrics)

    nrows = len(metrics)
    if figsize is None:
        figsize = (4, 3.2)
    fig, axarr = plt.subplots(nrows=nrows, ncols=1, sharex='col', figsize=figsize)

    if nrows == 1:
        axarr = np.array([axarr])

    for j, df in enumerate(loss_list):
        # this is just for the purpose of plotting the overall, not batches
        df_train = df.loc[df['minibatch_id'] == 'ALL', :]
        df_vali = df.loc[df['minibatch_id'] == 'VALI', :]

        # plot all

        for i, metric in enumerate(metrics):
            axarr[i].scatter(df_train['t'], df_train[metric], s=4, color=colours[j], zorder=2,
                             label='_nolegend_', alpha=0.5)
            axarr[i].plot(df_train['t'], df_train[metric], alpha=0.25, color=colours[j], zorder=2,
                          label=labels[j])

            if include_vali:
                axarr[i].plot(df_vali['t'], df_vali[metric], ls='--', color=colours[j], zorder=2,
                              label='_nolegend_', alpha=0.5)
            axarr[i].legend()

            if metric in ['mse']:
                axarr[i].set_yscale('log')
            axarr[i].set_ylabel(re.sub('_', '\n', metric))

            if include_batches:
                axarr[i].scatter(df['t'], df[metric], c=[colormap[x] for x in df['minibatch_id']],
                                 s=4, alpha=0.2, zorder=0)

                for minibatch_idx in df['minibatch_id'].unique():
                    df_temp = df.loc[df['minibatch_id'] == minibatch_idx, :]
                    axarr[i].plot(df_temp['t'], df_temp[metric], c=colormap[minibatch_idx], alpha=0.1, zorder=0)

    if include_convergence:
        for j, identifier in enumerate(identifiers):
            cfg_name = identifier['cfg_name']
            model = identifier['model']
            replace_index = identifier['replace']
            seed = identifier['seed']
            data_privacy = identifier['data_privacy']
            convergence_point = dr.find_convergence_point_for_single_experiment(cfg_name, model,
                                                                                replace_index,
                                                                                seed, diffinit,
                                                                                tolerance=convergence_tolerance,
                                                                                metric=metrics[0],
                                                                                data_privacy=data_privacy)
            print('Convergence point:', convergence_point)

            for ax in axarr:
                ax.axvline(x=convergence_point, ls='--', color=colours[j])
    axarr[-1].set_xlabel('training steps')

    vis_utils.beautify_axes(axarr)
    plt.tight_layout()

    if save:
        plot_label = '__'.join([f'r{x["replace"]}-s{x["seed"]}' for x in identifiers])
        plot_identifier = f'trace_{cfg_name}_{plot_label}'
        plt.savefig((PLOTS_DIR / plot_identifier).with_suffix('.png'))
        plt.savefig((PLOTS_DIR / plot_identifier).with_suffix('.pdf'))
    plt.clf()
    plt.close()

    return


def visualise_autocorrelation(cfg_name, model, replace_index, seed, params, save=True, n_lags=500) -> None:
    experiment = results_utils.ExperimentIdentifer(cfg_name, model, replace_index, seed)
    df = experiment.load_weights(params=params)
    autocorr = np.zeros(n_lags)
    fig, axarr = plt.subplots(nrows=len(params), ncols=1, sharex='col', figsize=(4, 1.5*len(params) + 1))
    axarr[0].set_title('autocorrelation of weight trajectory')

    for i, p in enumerate(params):
        for lag in range(n_lags):
            autocorr[lag] = df[p].autocorr(lag=lag)

        axarr[i].plot(range(n_lags), autocorr, alpha=0.5)
        axarr[i].scatter(range(n_lags), autocorr, s=4, zorder=2)
        axarr[i].set_ylabel(p)
    axarr[-1].set_xlabel('lag')

    plt.tight_layout()
    vis_utils.beautify_axes(axarr)

    if save:
        plot_identifier = f'autocorrelation_{cfg_name}_{model}_replace{replace_index}_seed{seed}'
        plt.savefig((PLOTS_DIR / plot_identifier).with_suffix('.png'))
        plt.savefig((PLOTS_DIR / plot_identifier).with_suffix('.pdf'))
    plt.clf()
    plt.close()

    return


def visually_compare_distributions(cfg_name, model, replace_index, seed, df=None,
                                   times=[10, 25], save=False, iter_range=(None, None), params=None) -> None:
    print('Visually comparing distributions at times', times)

    if df is None:
        experiment = results_utils.ExperimentIdentifier(cfg_name, model, replace_index, seed)
        df = experiment.load_gradients(noise=True, iter_range=iter_range, params=params)
    else:
        if params is not None:
            print('WARNING: Data provided, params argument ignored.')

    if df is False:
        print('[visually_compare_distributions] ERROR: No data available.')

        return False
    # restrict to a specific iteration
    fig, axarr = plt.subplots(nrows=len(times), ncols=3, sharey='row', sharex='col', figsize=(8, 2*len(times)+1))

    for i, t in enumerate(times):
        axrow = axarr[i, :]

        df_t = df.loc[df['t'] == t, :]

        if df_t.shape[0] == 0:
            print('WARNING: No data from iteration', t, ' - skipping!')

            continue
        df_fit = dr.estimate_statistics_through_training(what='gradients', cfg_name=cfg_name,
                                                         model=model,
                                                         replace_index=replace_index,
                                                         seed=seed, df=df_t)

        if params is not None:
            n_params = len(params)
            grad_noise = df_t.iloc[:, -n_params:].values.flatten()
        else:
            grad_noise = df_t.iloc[:, 2:].values.flatten()

        axrow[0].set_title('iteration: ' + str(t))
        sns.distplot(grad_noise, ax=axrow[0], kde=False, norm_hist=True, color='red')
        axrow[0].set_xlabel('')

        try:
            gauss_data = np.random.normal(size=10000, loc=df_fit['norm_mu'], scale=df_fit['norm_sigma'])
        except KeyError:
            ipdb.set_trace()
        sns.distplot(gauss_data, ax=axrow[1], kde=False, norm_hist=True, label='Gaussian')
        sns.distplot(grad_noise, ax=axrow[1], kde=False, norm_hist=True, color='red', label='gradients')
        axrow[1].set_xlabel('')

        lap_data = np.random.laplace(size=10000, loc=df_fit['lap_loc'], scale=df_fit['lap_scale'])
        sns.distplot(lap_data, ax=axrow[2], kde=False, norm_hist=True, label='Laplace')
        sns.distplot(grad_noise, ax=axrow[2], kde=False, norm_hist=True, color='red', label='gradients')
        axrow[2].set_xlabel('')

        vis_utils.beautify_axes(axrow)
    axarr[-1, 0].set_xlabel('Gradient noise')
    axarr[-1, 1].set_xlabel('Gaussian')
    axarr[-1, 1].legend()
    axarr[-1, 2].set_xlabel('Laplace')
    axarr[-1, 2].legend()

    plt.tight_layout()

    if save:
        plot_identifier = f'visual_{cfg_name}_{model}_replace{replace_index}_seed{seed}'

        if params is not None:
            plot_identifier = plot_identifier + f'_{n_params}params'
        else:
            plot_identifier = plot_identifier + '_joint'
        plt.savefig((PLOTS_DIR / plot_identifier).with_suffix('.png'))
        plt.savefig((PLOTS_DIR / plot_identifier).with_suffix('.pdf'))
        plt.clf()
        plt.close()

    return


def visualise_weight_trajectory(cfg_name, identifiers, df=None, save=True,
                                iter_range=(None, None), params=['#4', '#2'],
                                include_autocorrelation=False, diffinit=False) -> None:
    """
    """
    df_list = []

    for identifier in identifiers:
        model = identifier['model']
        replace_index = identifier['replace']
        seed = identifier['seed']
        experiment = results_utils.ExperimentIdentifier(cfg_name, model, replace_index, seed, diffinit)
        df = experiment.load_weights(iter_range=iter_range, params=params)
        df_list.append(df)
    colors = cm.viridis(np.linspace(0.2, 0.8, len(df_list)))
    labels = [':'.join(x) for x in identifiers]

    if params is None:
        if len(df.columns) > 6:
            print('WARNING: No parameters indicated, choosing randomly...')
            params = np.random.choice(df_list[0].columns[1:], 4, replace=False)
        else:
            print('WARNING: No parameters indicated, selecting all')
            params = df_list[0].columns[1:]

    for p in params:
        for df in df_list:
            assert p in df.columns

    if include_autocorrelation:
        ncols = 2
    else:
        ncols = 1
    fig, axarr = plt.subplots(nrows=len(params), ncols=ncols, sharex='col', figsize=(4*ncols, 1.5*len(params) + 1))

    firstcol = axarr[:, 0] if include_autocorrelation else axarr

    for k, df in enumerate(df_list):
        color = to_hex(colors[k])

        for i, p in enumerate(params):
            firstcol[i].scatter(df['t'], df[p], c=color, alpha=1, s=4, label=labels[k])
            firstcol[i].plot(df['t'], df[p], c=color, alpha=0.75, label='_nolegend_')
            firstcol[i].set_ylabel('param: ' + str(p))

        firstcol[0].set_title('weight trajectory')
        firstcol[-1].set_xlabel('training steps')
        firstcol[0].legend()

        if include_autocorrelation:
            n_lags = 100
            autocorr = np.zeros(n_lags)
            axarr[0, 1].set_title('autocorrelation of weight trajectory')

            for i, p in enumerate(params):
                for lag in range(n_lags):
                    autocorr[lag] = df[p].autocorr(lag=lag)

                axarr[i, 1].plot(range(n_lags), autocorr, alpha=0.5, color=color)
                axarr[i, 1].scatter(range(n_lags), autocorr, s=4, zorder=2, color=color)
                axarr[i, 1].set_ylabel(p)
                axarr[i, 1].axhline(y=0, ls='--', alpha=0.5, color='black')
            axarr[-1, 1].set_xlabel('lag')

    vis_utils.beautify_axes(axarr)
    plt.tight_layout()

    if save:
        plot_identifier = f'weights_{cfg_name}_{"_".join(labels)}'
        plt.savefig((PLOTS_DIR / plot_identifier).with_suffix('.png'))
        plt.savefig((PLOTS_DIR / plot_identifier).with_suffix('.pdf'))
    plt.clf()
    plt.close()

    return


def compare_posteriors_with_different_data(cfg_name, model, t, replace_indices, params) -> None:
    plt.clf()
    plt.close()
    fig, axarr = plt.subplots(nrows=1, ncols=len(params))
    colours = cm.viridis(np.linspace(0.2, 0.8, len(replace_indices)))

    for j, replace_index in enumerate(replace_indices):
        for i, p in enumerate(params):
            samples = results_utils.get_posterior_samples(cfg_name, iter_range=(t, t+1), model=model,
                                                          replace_index=replace_index, params=[p])
            sns.distplot(samples, ax=axarr[i], color=to_hex(colours[j]), label=str(replace_index), kde=False)
    # save

    for i, p in enumerate(params):
        axarr[i].set_xlabel('parameter ' + p)

    axarr[0].set_title('iteration ' + str(t))
    axarr[-1].legend()
    vis_utils.beautify_axes(axarr)

    return


def delta_over_time(cfg_name, model, identifier_pair, iter_range, include_bound=False) -> None:
    """
    """
    assert len(identifier_pair) == 2
    replace_1, seed_1 = identifier_pair[0]['replace'], identifier_pair[0]['seed']
    replace_2, seed_2 = identifier_pair[1]['replace'], identifier_pair[1]['seed']
    experiment_1 = results_utils.ExperimentIdentifier(cfg_name, model, replace_1, seed_1)
    experiment_2 = results_utils.ExperimentIdentifier(cfg_name, model, replace_2, seed_2)
    samples_1 = experiment_1.load_weights(iter_range=iter_range)
    samples_2 = experiment_2.load_weights(iter_range=iter_range)
    gradients_1 = experiment_1.load_gradients(noise=False, iter_range=iter_range)
    gradients_2 = experiment_2.load_gradients(noise=False, iter_range=iter_range)
    # align the time-points
    samples_1.set_index('t', inplace=True)
    samples_2.set_index('t', inplace=True)
    gradients_1.set_index('t', inplace=True)
    gradients_2.set_index('t', inplace=True)
    gradients_1 = gradients_1.loc[gradients_1['minibatch_id'] == 'ALL', :]
    gradients_2 = gradients_2.loc[gradients_2['minibatch_id'] == 'ALL', :]
    gradients_1.drop(columns='minibatch_id', inplace=True)
    gradients_2.drop(columns='minibatch_id', inplace=True)
    assert np.all(samples_1.index == samples_2.index)
    assert np.all(gradients_1.index == gradients_2.index)
    delta = np.linalg.norm(samples_1 - samples_2, axis=1)
    gradnorm_1 = np.linalg.norm(gradients_1, axis=1)
    gradnorm_2 = np.linalg.norm(gradients_2, axis=1)
    t = samples_1.index
    # now visualise
    fig, axarr = plt.subplots(nrows=3, ncols=1)
    axarr[0].plot(t, delta, alpha=0.5)
    axarr[0].scatter(t, delta, s=4)
    axarr[0].set_ylabel('|| w - w\' ||')

    if include_bound:
        assert model == 'logistic'
        _, batch_size, eta, _, N = em.get_experiment_details(cfg_name, model)
        L = np.sqrt(2)
        bound = np.zeros(len(t))

        for i, ti in enumerate(t):
            bound[i] = test_private_model.compute_wu_bound(L, ti, N, batch_size, eta, verbose=False)
        axarr[0].plot(t, bound)
    axarr[1].plot(t, gradnorm_1)
    axarr[2].plot(t, gradnorm_2)
    axarr[1].axhline(y=L, ls='--')
    axarr[2].axhline(y=L, ls='--')
    vis_utils.beautify_axes(axarr)

    return


def sensitivity_v_variability(cfg_name, model, t, num_pairs, diffinit=False) -> None:
    try:
        df = dr.SensVar(cfg_name, model, t=t, num_pairs=num_pairs).load(diffinit=diffinit)
    except FileNotFoundError:
        print('ERROR: Compute sens and var dist')
        return

    fig, axarr = plt.subplots(nrows=2, ncols=1)
    axarr[0].scatter(df['sensitivity'], df['variability'], s=4, alpha=0.5)
    sns.kdeplot(df['sensitivity'], df['variability'], ax=axarr[1], shade=True, cbar=False, shade_lowest=False)
    axarr[0].set_xlim(axarr[1].get_xlim())
    axarr[0].set_ylim(axarr[1].get_ylim())

    for ax in axarr:
        ax.set_xlabel('sensitivity')
        ax.set_ylabel('variability')
    axarr[0].set_title('cfg_name: ' + cfg_name + ', model: ' + model + ', t: ' + str(t) + ' (variable init)'*diffinit)
    plt.tight_layout()
    vis_utils.beautify_axes(axarr)

    return


def mvn_covariance(X, identifier: str = '') -> None:
    cov = np.cov(X.T)
    sns.heatmap(cov)
    plt.savefig(PLOTS_DIR / f'{identifier}_cov.png')
    plt.clf()
    plt.close()
    return


def multivariate_normal_test_vis(df, logscale: bool = False) -> None:
    fig, axarr = plt.subplots(nrows=3, ncols=1, sharex=True)
    axarr[-1].set_xlabel('N')
    ds = df['d'].unique()
    colours = cm.viridis(np.linspace(0, 1, len(ds)))
    for i, d in enumerate(ds):
        df_d = df[df['d'] == d]
        for j, label in enumerate(['pval_diagonal_gauss', 'pval_nondiag_gauss', 'pval_laplace']):
            val_mean = df_d[[label, 'n']].groupby('n').mean()
            val_std = df_d[[label, 'n']].groupby('n').std()
            uh = axarr[j].plot(val_mean.index, val_mean.values[:, 0], color=colours[i], label=d)
            axarr[j].fill_between(val_mean.index, (val_mean - val_std).values[:, 0], (val_mean + val_std).values[:, 0], color=colours[i], alpha=0.1)
    fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(vmin=min(ds), vmax=max(ds)), cmap='viridis'),
                 ax=axarr, label='dimension', drawedges=False, ticks=ds)
    axarr[0].set_ylabel('pval\nMVN')
    axarr[1].set_ylabel('pval\nMVNd')
    axarr[2].set_ylabel('pval\nlaplace')

    for ax in axarr:
        #ax.legend()
        ax.axhline(y=0.05, ls='--', color='red', alpha=0.5)
        if logscale:
            ax.set_yscale('log')
            ax.set_ylim(1e-5, 1)
        else:
            ax.set_ylim(0, 1)
    vis_utils.beautify_axes(axarr)

    plt.savefig(PLOTS_DIR / f'multivar_test{"_log"*logscale}.png')
    plt.savefig(PLOTS_DIR / f'multivar_test{"_log"*logscale}.pdf')
    plt.clf()
    plt.close()
    return


def sens_v_var_per_parameter(cfg_name, model, t, diffinit=True) -> None:
    sens = dr.estimate_sensitivity_empirically(cfg_name, model, t, num_deltas='max',
                                               diffinit=diffinit, multivariate=True)
    sens_uni = dr.estimate_sensitivity_empirically(cfg_name, model, t, num_deltas='max',
                                               diffinit=diffinit, multivariate=False)
    var_diffinit = dr.estimate_variability(cfg_name, model, t=t, multivariate=True, diffinit=True)
    var_diffinit_uni = dr.estimate_variability(cfg_name, model, t=t, multivariate=False, diffinit=True)
    var_fixinit = dr.estimate_variability(cfg_name, model, t=t, multivariate=True, diffinit=False)
    var_fixinit_uni = dr.estimate_variability(cfg_name, model, t=t, multivariate=False, diffinit=False)
    
    fig, axarr = plt.subplots(nrows=1, ncols=1)
    size = 4
    axarr.scatter(x=sens, y=var_fixinit, label='fixed init', color='red', s=size)
    axarr.scatter(x=sens, y=var_diffinit, label='diff init', color='blue', s=size)
    axarr.axvline(x=sens_uni, ls='--', label='overall sensitivity', color='black')
    axarr.axhline(y=var_diffinit_uni, ls='--', label='overall sigma (diff init)', color='blue')
    axarr.axhline(y=var_fixinit_uni, ls='--', label='overall sigma (fixed init)', color='red')
    vis_utils.beautify_axes(np.array([axarr]))
    axarr.set_xlabel('sensitivity')
    axarr.legend()
    axarr.set_ylabel('sigma')
    axarr.set_title(f'{cfg_name} at {t}')

    plt.savefig(PLOTS_DIR / f'per_param_sens_v_var_{cfg_name}_t{t}.png')
    return


def plot_umap(embedded, labels, identifier: str, xlim=None, ylim=None) -> None:
    assert 'replace_index' in labels.columns
    assert 'seed' in labels.columns

    fig, axarr = plt.subplots(nrows=2, ncols=1, figsize=(3, 5), sharex=True)
    size = 5
    axarr[0].set_title('Coloured by dataset')
    axarr[0].scatter(embedded[:, 0], embedded[:, 1], c=labels['replace_index'], cmap='Spectral', s=size, alpha=0.5)
    axarr[1].set_title('Coloured by random seed')
    axarr[1].scatter(embedded[:, 0], embedded[:, 1], c=labels['seed'], cmap='Spectral', s=size, alpha=0.5)
    vis_utils.beautify_axes(axarr)

    if xlim is not None:
        for ax in axarr:
            ax.set_xlim(xlim)
    if ylim is not None:
        for ax in axarr:
            ax.set_ylim(ylim)

    plt.savefig(PLOTS_DIR / f'{identifier}.png')
    return


def umap_on_weights(cfg_name, model, t, diffinit=False, xlim=None, ylim=None) -> None:
    print(f'Loading all the weights at time {t}')
    temp_path = PLOTS_DIR / 'temp_data' / f'umap_{cfg_name}_{t}_diffinit{diffinit}.csv'
    try:
        df = pd.read_csv(temp_path)
        print('Loaded a temp file of weights')
    except FileNotFoundError:
        df = results_utils.get_posterior_from_all_datasets(cfg_name, iter_range=(t, t+1),
                                                           model=model, what='weights',
                                                           verbose=False,
                                                           diffinit=diffinit, sort=False)
        df.to_csv(temp_path)

    if cfg_name == 'mnist_binary_lr':
        # subsample replace indices 4350 and 5669 - or filter to "normal" seeds
        normal_seeds = df.loc[df['replace_index'] == 4831]['seed'].unique()
        df_new = df.loc[df['seed'].isin(normal_seeds)]
        print(f'Subsampled down to {df_new.shape[0]} examples, from {df.shape[0]}')
        df = df_new

    weights = df.drop(columns=['t', 'seed', 'replace_index']).values
    labels = df[['seed', 'replace_index']]

    print('checking counts for replace index and seeds..')
    print(labels['replace_index'].value_counts())
    print(labels['seed'].value_counts())

    reducer = umap.UMAP()
    print(f'Data size: {weights.shape}')
    print('UMAPping!')
    embedded = reducer.fit_transform(weights)

    identifier = f'umap_{cfg_name}_t{t}_diffinit{diffinit}'
    plot_umap(embedded, labels, identifier, xlim=xlim, ylim=ylim)
