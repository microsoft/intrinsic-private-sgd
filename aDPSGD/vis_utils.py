#!/usr/bin/env ipython
# visualisation tools!

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_hex
import statsmodels.api as sm
import re
import seaborn as sns
import ipdb

import results_utils
import derived_results
import test_private_model
import data_utils
import experiment_metadata

def beautify_axes(axarr):
    """
    Standard prettification edits I do in matplotlib
    """
    if len(axarr.shape) == 1:
        axarr = [axarr]
    for axrow in axarr:
        for ax in axrow:
            ax.set_facecolor((0.95, 0.95, 0.95)) 
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.tick_params(bottom=False, left=False)
            ax.grid(linestyle='--', alpha=0.5)
    return True

def process_identifiers(datasets, models, replaces, seeds, privacys):
    # figure out length of longest provided list
    n_identifiers = 1
    for identifier_component in datasets, replaces, seeds, privacys:
        if type(identifier_component) == list:
            length = len(identifier_component)
            if length > n_identifiers:
                n_identifiers = length
    # make sure everything is either that length or not a list
    if type(datasets) == list:
        if not len(datasets) == n_identifiers:
            assert len(datasets) == 1
            datasets = n_identifiers*datasets
    else:
        datasets = [datasets]*n_identifiers
    if type(models) == list:
        if not len(models) == n_identifiers:
            assert len(models) == 1
            models = n_identifiers*models
    else:
        models = [models]*n_identifiers
    if type(replaces) == list:
        if not len(replaces) == n_identifiers:
            assert len(replaces) == 1
            replaces = n_identifiers*replaces
    else:
        replaces = [replaces]*n_identifiers
    if type(seeds) == list:
        if not len(seeds) == n_identifiers:
            assert len(seeds) == 1
            seeds = n_identifiers*seeds
    else:
        seeds = [seeds]*n_identifiers
    if type(privacys) == list:
        if not len(privacys) == n_identifiers:
            assert len(privacys) == 1
            privacys = n_identifiers*privacys
    else:
        privacys = [privacys]*n_identifiers
    identifiers = list(zip(datasets, models, replaces, seeds, privacys))
    return identifiers

def qq_plot(what, dataset, identifier, times=[50], params='random'):
    """
    grab trace file, do qq plot for gradient noise at specified time-point
    """
    plt.clf()
    plt.close()
    assert what in ['gradients', 'weights']
    model, replace_index, seed = identifier
    if what == 'weights':
        print('Looking at weights, this means we consider all seeds!')
    colours = cm.viridis(np.linspace(0.2, 0.8, len(times)))
    if params == 'random':
        if what == 'gradients':
            df = results_utils.load_gradients(dataset, model, replace_index, seed, noise=True, params=None, iter_range=(min(times), max(times)+1))
        else:
            df = results_utils.get_posterior_samples(dataset, model=model, replace_index=replace_index, iter_range=(min(times), max(times) + 1), params=None)
        params = np.random.choice(df.columns[2:], 1)
        print('picking random parameter', params)
        first_two_cols = df.columns[:2].tolist()
        df = df.loc[:, first_two_cols + list(params)]
    else:
        if what == 'gradients':
            df = results_utils.load_gradients(dataset, model, replace_index, seed, noise=True, params=params, iter_range=(min(times), max(times)+1))
        else:
            df = results_utils.get_posterior_samples(dataset, model=model, replace_index=replace_index, iter_range=(min(times), max(times) + 1), params=params)
    if df is False:
        print('ERROR: No data available')
        return False
    # now actually visualise 
    fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(7, 3.5))
    for i, t in enumerate(times):
        df_t = df.loc[df['t'] == t, :]
        X = df_t.iloc[:, 2:].values.flatten()
        print('number of samples:', X.shape[0])
        sns.distplot(X, ax=axarr[0], kde=False, color=to_hex(colours[i]), label=str(t))
        sm.qqplot(X, line='45', fit=True, ax=axarr[1], c=colours[i], alpha=0.5, label=str(t))
    plt.suptitle('dataset: ' + dataset + ', model:' + model + ',' + what)
    axarr[0].legend()
    axarr[1].legend()
    axarr[0].set_xlabel('parameter:' + '.'.join(params))
    beautify_axes(axarr)
    plt.tight_layout()
    if what == 'weights':
        plot_label = model + '.' + '.'.join(params)
    else:
        plot_label = '.'.join(identifier) + '.'.join(params)
    plt.savefig('plots/' + dataset + '/' + plot_label + '_qq' + '_' + what + '.png')
    return True

def visualise_gradient_values(dataset, identifiers, save=True, iter_range=(None, None), params=None, full_batch=True, include_max=False, diffinit=False, what='norm'):
    """
    if include_max: plot the max gradient norm (this would be the empirical lipschitz constant)
    """
    fig, axarr = plt.subplots(nrows=1, ncols=1)
    assert len(identifiers) == len(scaling)
    for i, identifier in enumerate(identifiers):
        label = ':'.join(identifier)
        model, replace_index, seed = identifier
        df = results_utils.load_gradients(dataset, model, replace_index, seed, noise=False, iter_range=iter_range, params=params, diffinit=diffinit)
        if full_batch:
            # only use gradients from full dataset
            df = df.loc[df['minibatch_id'] == 'ALL', :]
        else:
            # only use gradients from minibatches
            df = df.loc[~(df['minibatch_id'] == 'ALL'), :]
        times = df['t'].values.astype('float')
        if what == 'norm':
            grad_vals = np.linalg.norm(df.values[:, 2:].astype(float), axis=1)
        elif what == 'max':
            grad_vals = np.max(df.values[:, 2:].astype(float), axis=1)
        axarr.scatter(times, grad_vals, label=label, s=4)
        axarr.plot(times, grad_vals, label=label, alpha=0.5) 
        if include_max:
            axarr.axhline(y=max(grad_vals), ls='--', alpha=0.5, color='black')
    axarr.legend()
    beautify_axes(np.array([axarr]))
    axarr.set_title(dataset + ' ' + model)
    axarr.set_ylabel('gradient ' + what)
    axarr.set_xlabel('training steps')
    if save:
        plot_label = '_'.join([':'.join(x) for x in identifiers])
        plt.savefig('plots/' + dataset + '/grad' + what + '_' + plot_label + '.png')
    return True

def bivariate_gradients(dataset, model, replace_index, seed, df=None, params=['#3', '#5'], iter_range=(None, None), n_times=2, save=False):
    print('Comparing gradients for parameters', params, 'at', n_times, 'random time-points')
    if df is None:
        df = results_utils.load_gradients(dataset, model, replace_index, seed, noise=True, iter_range=iter_range, params=params)
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
    beautify_axes(axarr)
    plt.tight_layout()
    if save:
        assert not identifier is None
        plt.savefig('plots/' + identifier + '_gradient_pairs.png')
        plt.clf()
        plt.close()
    return True

def fit_pval_histogram(what, dataset, model, t, n_experiments=3, diffinit=False, xlim=None):
    """
    histogram of p-values (across parameters-?) for a given model etc.
    """
    # set some stuff up
    iter_range = (t, t +1)
    fig, axarr = plt.subplots(nrows=1, ncols=1, figsize=(3.5, 2.1))
    colours = cm.viridis(np.linspace(0.2, 0.8, n_experiments))
    pval_colour = '#b237c4'
    # sample experiments
    df = results_utils.get_available_results(dataset, model, diffinit=diffinit)
    replace_indices = df['replace'].unique()
    replace_indices = np.random.choice(replace_indices, n_experiments, replace=False)
    print('Looking at replace indices...', replace_indices)
    all_pvals = []
    for i, replace_index in enumerate(replace_indices):
        if what == 'gradients':
            print('Loading gradients...')
            df = results_utils.load_gradients(dataset, model, replace_index,
                    seed, noise=True, params=None, iter_range=iter_range)
            second_col = df.columns[1]
        elif what == 'weights':
            df = results_utils.get_posterior_samples(dataset, iter_range=iter_range, model=model, replace_index=replace_index, params=None, seeds='all')
            second_col = df.columns[1]
        else:
            raise ValueError(what)
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
            df_fit = derived_results.estimate_statistics_through_training(what=what, dataset=None, identifier=None, df=df.loc[:, ['t', second_col, p]], params=None, iter_range=None)
            p_vals[j] = df_fit.loc[t, 'norm_p']
            del df_fit
        log_pvals = np.log(p_vals)
        all_pvals.append(log_pvals)
    log_pvals = np.concatenate(all_pvals)
    if not xlim is None:
        # remove values below the limit
        number_below = (log_pvals < xlim[0]).sum()
        print('There are', number_below, 'p-values below the limit of', xlim[0])
        log_pvals = log_pvals[log_pvals > xlim[0]]
        print('Remaining pvals:', len(log_pvals))
    sns.distplot(log_pvals, kde=True, bins=min(100, int(len(log_pvals)*0.25)), 
            ax=axarr, color=pval_colour, norm_hist=True)
    #axarr.set_title('dataset: ' + dataset + ', model: ' + model)
    axarr.axvline(x=np.log(0.05), ls=':', label='p = 0.05', color='black', alpha=0.75)
    axarr.axvline(x=np.log(0.05/n_params), ls='--', label='p = 0.05/' + str(n_params), color='black', alpha=0.75)
    axarr.legend()
    axarr.set_xlabel(r'$\log(p)$')
    axarr.set_ylabel('density')
    if not xlim is None:
        axarr.set_xlim(xlim)
    else:
        axarr.set_xlim((None, 0.01))
#    axarr.set_xscale('log')
    beautify_axes(np.array([axarr]))
    plt.tight_layout()
    plt.savefig('plots/analyses/' + dataset + '_' + model + '_' + what + '_pval_histogram.png')
    plt.savefig('plots/analyses/' + dataset + '_' + model + '_' + what + '_pval_histogram.pdf')
    return True

def visualise_fits(dataset, identifier, save=True, params=None):
    print('Visualising distribution fits through training')
    # load and fit the data
    if params is None:
        params = [None]
    
    # establish the plot stuff
    fig, axarr = plt.subplots(nrows=3, ncols=1, sharex='col', figsize=(4,5))
      
    n_comparators = len(params)
    colours = cm.viridis(np.linspace(0.2, 0.8, n_comparators))
    for i, p in enumerate(params):
        print('visualising fit for parameter parameter', p)
        df_fit = derived_results.estimate_statistics_through_training(what='gradients', dataset=dataset, identifier=identifier, params=[p])
        if df_fit is False:
            print('No fit data available for identifier:', identifier)
            return False
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
        
        #axarr[1, 0].scatter(iterations, df_fit['lap_scale'], c=color, alpha=1, s=4, zorder=2, label='scale' if i == 0 else '_nolegend_')
        #axarr[1, 0].plot(iterations, df_fit['lap_scale'], c=color, alpha=0.75, zorder=2, label='_nolegend_')
        #axarr[1, 1].scatter(iterations, df_fit['lap_D'], c=color, alpha=1, s=4, zorder=2, label='D (KS)' if i == 0 else '_nolegend_')
        #axarr[1, 1].plot(iterations, df_fit['lap_D'], c=color, alpha=0.75, zorder=2, label='_nolegend_')
        #axarr[1, 2].scatter(iterations, df_fit['lap_p'] + np.spacing(1), c=color, alpha=1, s=4, zorder=2, label='log(p)' if i == 0 else '_nolegend_')
        #axarr[1, 2].plot(iterations, df_fit['lap_p'] + np.spacing(1), c=color, alpha=0.75, zorder=2, label='_nolegend_')

        
    #    axarr[2, 0].scatter(iterations, df_fit['alpha'], c='black', alpha=1, s=4, zorder=2, label='alpha')
    #    axarr[2, 0].plot(iterations, df_fit['alpha'], c='black', alpha=0.75, zorder=2, label='_nolegend_')

       
    if (len(params) > 1) and (len(params) < 5):
        print(len(params), 'parameters - adding a legend')
        axarr[0].legend()
    axarr[1].set_ylim(0.9, 1)
    axarr[-1].set_yscale('log')
    axarr[-1].axhline(y=0.05, c='red', ls='--', label='p = 0.05')
    axarr[-1].set_xlabel('training iterations')
    # fix y-limits of first two rows
    #for col in [0, 1, 2]:
        #    y_max = np.max([axarr[0, col].get_ylim()[1], axarr[1, col].get_ylim()[1]])
        #y_min = np.min([axarr[0, col].get_ylim()[0], axarr[1, col].get_ylim()[0]])
        #axarr[0, col].set_ylim((y_min, y_max))
        #axarr[1, col].set_ylim((y_min, y_max))
    beautify_axes(axarr)
    plt.tight_layout()
   
    if save:
        plot_label = '.'.join(identifier) + '.'.join(params)
        plt.savefig('plots/' + dataset + '/' + plot_label + '_fits.png')
        plt.clf()
        plt.close()
    return True

def visualise_variance(df, times, colormap=None, identifier=None, save=False, value_lim=None):
    """
    At a set of timepoints (one on each row), show histogram of the value of the columns of df
    with variance arising due to minibatch sampling.
    
    Also overlay the "batch" estimate
    """
    print('Visualising variance at times:', times)
    assert df.shape[1] > 2
    what_is_it = df.columns[2:]
    if len(what_is_it) > 4:
        what_is_it = np.random.choice(what_is_it, 4, replace=False)
        print('Subsetting to columns', what_is_it)
    
    nrows = len(times)
    assert nrows > 1
    ncols = len(what_is_it)
   
    fig, axarr = plt.subplots(nrows=nrows, ncols=ncols, sharex='col')
    for i, t in enumerate(times):
        df_t = df.loc[df['t'] == t, :]
        df_minibatch = df_t.loc[~((df_t['minibatch_id'] == 'ALL')|(df_t['minibatch_id'] == 'VALI')), :]
        if not df_minibatch.shape[0] > 1:
            print('WARNING: No minibatch data at time', t)
            continue
        df_all = df_t.loc[df_t['minibatch_id'] == 'ALL', :]
        assert df_all.shape[0] == 1
        axrow = axarr[i]
        if ncols == 1:
            axrow = [axrow]
        for j, what in enumerate(what_is_it):
            ax = axrow[j]
            sns.distplot(df_minibatch.loc[:, what], ax=ax, norm_hist=True, kde=False)
            batch_value = df_all[what].iloc[0]
            ax.axvline(x=batch_value, ls='--', color='blue')
            if i == nrows - 1:
                ax.set_xlabel(what)
            if j == 0:
                ax.set_ylabel('iter:' + str(t))
            if j == ncols - 1 and not value_lim is None:
                ax.set_xlim(value_lim)


#    axarr[0].set_title(label)
#    if not value_lim is None:
#        axarr[-1].set_xlim(value_lim)
   
#    for ax in axarr[:-1]:
#        ax.set_xlabel('')

    beautify_axes(axarr)
    plt.tight_layout()
    if save:
        plt.savefig('plots/' + identifier + '_variance.png')
        plt.clf()
        plt.close()
    return True

def visualise_trace(datasets, models, replaces, seeds, privacys, save=True, 
        include_batches=False, iter_range=(None, None), 
        include_convergence=True, diffinit=False, convergence_tolerance=3, 
        include_vali=True, labels=None):
    """
    Show the full training set loss as well as the gradient (at our element) over training
    """
    identifiers = process_identifiers(datasets, models, replaces, seeds, privacys)

    if len(identifiers) > 1:
        print('WARNING: When more than one experiment is included, we turn off visualisation of batches to avoid cluttering the plot')
        include_batches = False

    if labels is None:
        labels = [':'.join(x) for x in identifiers]
    else:
        assert len(labels) == len(identifiers)

    loss_list = []
    for identifier in identifiers:
        dataset, model, replace_index, seed, data_privacy = identifier
        df_loss = results_utils.load_loss(dataset, model, replace_index, seed, iter_range=iter_range, diffinit=diffinit, data_privacy=data_privacy)
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
    fig, axarr = plt.subplots(nrows=nrows, ncols=1, sharex='col', figsize=(4, 3.2))
    if nrows == 1:
        axarr = np.array([axarr])

    for j, df in enumerate(loss_list):
        # this is just for the purpose of plotting the overall, not batches
        df_train = df.loc[df['minibatch_id'] == 'ALL', :]
        df_vali = df.loc[df['minibatch_id'] == 'VALI', :]

        # plot all
        for i, metric in enumerate(metrics):
            axarr[i].scatter(df_train['t'], df_train[metric], s=4, color=colours[j], zorder=2, label='_nolegend_', alpha=0.5)
            axarr[i].plot(df_train['t'], df_train[metric], alpha=0.25, color=colours[j], zorder=2, label=labels[j] )
            if include_vali:
                axarr[i].plot(df_vali['t'], df_vali[metric], ls='--', color=colours[j], zorder=2, label='_nolegend_', alpha=0.5)
            #axarr[i].plot(df_vali['t'], df_vali[metric], alpha=0.5, color='red', zorder=2, label='_nolegend_')
            axarr[i].legend()
            if metric in ['mse']:
                axarr[i].set_yscale('log')
            axarr[i].set_ylabel(re.sub('_', '\n', metric))
            if include_batches:
                axarr[i].scatter(df['t'], df[metric], c=[colormap[x] for x in df['minibatch_id']], s=4, alpha=0.2, zorder=0)
                for minibatch_idx in df['minibatch_id'].unique():
                    df_temp = df.loc[df['minibatch_id'] == minibatch_idx, :]
                    axarr[i].plot(df_temp['t'], df_temp[metric], c=colormap[minibatch_idx], alpha=0.1, zorder=0)

    if include_convergence:
        for j, identifier in enumerate(identifiers):
            dataset, model, replace_index, seed, data_privacy  = identifier
            convergence_point = derived_results.find_convergence_point_for_single_experiment(dataset, model, replace_index, seed, diffinit, tolerance=convergence_tolerance, metric=metrics[0], data_privacy=data_privacy)
            print('Convergence point:', convergence_point)
            for ax in axarr:
                ax.axvline(x=convergence_point, ls='--', color=colours[j])
    axarr[-1].set_xlabel('training steps')
    
    beautify_axes(axarr)
    plt.tight_layout()
    if save:
        plot_label = '.'.join([':'.join(x) for x in identifiers])
        plt.savefig('plots/' + dataset + '/' + plot_label + '_trace.png')
        plt.savefig('plots/' + dataset + '/' + plot_label + '_trace.pdf')
    plt.clf()
    plt.close()
    return True

def visualise_autocorrelation(dataset, model, replace_index, seed, params, save=True):
    """ what's the autocorrelation of the weights?.... or gradients? """
    df = results_utils.load_weights(dataset, model, replace_index, seed, params=params)
    n_lags = 500
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
    beautify_axes(axarr)
    if save:
        identifier = model + '.' + str(replace_index) + '.' + str(seed)
        plt.savefig('plots/' + dataset + '/' + identifier + '_autocorrelation.png')
    plt.clf()
    plt.close()
    return True

def examine_parameter_level_gradient_noise(dataset, identifier, times=[10, 25], save=True, params=['#1', '#5']):
    print('demonstrating gradient noise distributions for', identifier, 'at times', times, 'for parameters', params)
    iter_range = (min(times) - 1, max(times) + 1)
    assert not params is None
    df = results_utils.load_gradients(dataset, model, replace_index, seed, noise=True, iter_range=iter_range, params=params)

    ncols = len(params)
    param_cols = cm.viridis(np.linspace(0.2, 0.8, ncols))
    fig, axarr = plt.subplots(nrows=len(times), ncols=ncols, sharey='row', sharex='col', figsize=(1.7*len(params) + 1, 2*len(times)+1))

    for i, t in enumerate(times):
        df_t = df.loc[df['t'] == t, :]
        if df_t.shape[0] == 0:
            print('WARNING: No data from iteration', t, ' - skipping!')
            continue
        for j, p in enumerate(params):
            grad_noise = df_t.loc[:, p].values.flatten()
            sns.distplot(grad_noise, ax=axarr[i, j], kde=False, norm_hist=True, color=to_hex(param_cols[j]))

    # now set the axes and stuff
    for i, t in enumerate(times):
        axarr[i, 0].set_ylabel('iteration :' + str(t))
    for j, p in enumerate(params):
        axarr[0, j].set_title('param: ' + p)
        axarr[-1, j].set_xlabel('gradient noise')

    beautify_axes(axarr)
    if save:
        plot_label = identifier + '.gradient_noise.params' + '.'.join(params)
        plt.savefig('plots/' + dataset + '/' + plot_label + '_trace.png')
    plt.clf()
    plt.close()
    return True

def visually_compare_distributions(identifier, df=None, times=[10, 25], save=False, iter_range=(None, None), params=None):
    print('Visually comparing distributions for', identifier, 'at times', times)
    if df is None:
        df = results_utils.load_gradients(dataset, model, replace_index, seed, noise=True, iter_range=iter_range, params=params)
    else:
        if not params is None:
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
        df_fit = derived_results.estimate_statistics_through_training(what='gradinets', dataset=dataset, identifier=identifier, df=df_t)
        if not params is None:
            n_params = len(params)
            grad_noise = df_t.iloc[:, -n_params:].values.flatten()
            #grad_noise = df_t['grad_noise'].values
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

        lap_data= np.random.laplace(size=10000, loc=df_fit['lap_loc'], scale=df_fit['lap_scale'])
        sns.distplot(lap_data, ax=axrow[2], kde=False, norm_hist=True, label='Laplace')
        sns.distplot(grad_noise, ax=axrow[2], kde=False, norm_hist=True, color='red', label='gradients')
        axrow[2].set_xlabel('')

        beautify_axes(axrow)
    axarr[-1, 0].set_xlabel('Gradient noise')
    axarr[-1, 1].set_xlabel('Gaussian')
    axarr[-1, 1].legend()
    axarr[-1, 2].set_xlabel('Laplace')
    axarr[-1, 2].legend()
    
    plt.tight_layout()
    if save:
        if not params is None:
            label = identifier + '_' + str(n_params) + 'params'
        else:
            label = identifier + '_joint'
        plt.savefig('plots/' + label + '_visual.png')
        plt.clf()
        plt.close()
    return True

def visualise_weight_trajectory(dataset, identifiers, df=None, save=True, iter_range=(None, None), params=['#4', '#2'], include_optimum=False,
        include_autocorrelation=False, diffinit=False):
    """
    """
    df_list = []
    for identifier in identifiers:
        model, replace_index, seed = identifier
        df = results_utils.load_weights(dataset, model, replace_index, seed, diffinit=diffinit, iter_range=iter_range, params=params)
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
 
    if include_optimum:
        # hack!
        dataset = identifier.split('/')[0]
        optimum, hessian = data_utils.solve_with_linear_regression(dataset)
   
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
            if include_optimum:
                firstcol[i].axhline(y=optimum[int(p[1:])], ls='--', color='red', alpha=0.5)
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

    beautify_axes(axarr)
    plt.tight_layout()
    if save:
        plot_label = '.'.join(labels)
        plt.savefig('plots/' + dataset + '/' + plot_label + '_weights.png')
    plt.clf()
    plt.close()
    return True


def compare_posteriors_with_different_data(dataset, model, t, replace_indices, params):
    plt.clf()
    plt.close()
    fig, axarr = plt.subplots(nrows=1, ncols=len(params))
    colours = cm.viridis(np.linspace(0.2, 0.8, len(replace_indices)))
    for j, replace_index in enumerate(replace_indices):
        for i, p in enumerate(params):
            samples = results_utils.get_posterior_samples(dataset, iter_range=(t, t+1), model=model, replace_index=replace_index, params=[p])
            sns.distplot(samples, ax=axarr[i], color=to_hex(colours[j]), label=str(replace_index), kde=False)
    # save
    for i, p in enumerate(params):
        axarr[i].set_xlabel('parameter ' + p)
   
    axarr[0].set_title('iteration ' + str(t))
    axarr[-1].legend()
    beautify_axes(axarr)
    return True

def delta_over_time(dataset, model, identifier_pair, iter_range, include_bound=False):
    """
    """
    assert len(identifier_pair) == 2
    replace_1, seed_1 = identifier_pair[0]
    replace_2, seed_2 = identifier_pair[1]
    samples_1 = results_utils.load_weights(dataset, model, replace_1, seed_1, iter_range=iter_range)
    samples_2 = results_utils.load_weights(dataset, model, replace_2, seed_2, iter_range=iter_range)
    gradients_1 = results_utils.load_gradients(dataset, model, replace_1, seed_1, iter_range=iter_range)
    gradients_2 = results_utils.load_gradients(dataset, model, replace_2, seed_2, iter_range=iter_range)
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
        _, batch_size, eta, _, N = experiment_metadata.get_experiment_details(dataset, model)
        L = np.sqrt(2)
        bound = np.zeros(len(t))
        for i, ti in enumerate(t):
            bound[i] = test_private_model.compute_wu_bound(L, ti, N, batch_size, eta, verbose=False)
        axarr[0].plot(t, bound)
    axarr[1].plot(t, gradnorm_1)
    axarr[2].plot(t, gradnorm_2)
    axarr[1].axhline(y=L, ls='--')
    axarr[2].axhline(y=L, ls='--')
    beautify_axes(axarr)
    return True


def sensitivity_v_variability(dataset, model, t, num_pairs, diffinit=False):
    path = './fig_data/sens_var_dist.' + dataset + '.' + model + '.t' + str(t) + '.np' + str(num_pairs) + '.DIFFINIT'*diffinit + '.csv'
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print('ERROR: Compute sens and var dist for', path)
        return False
    fig, axarr = plt.subplots(nrows=2, ncols=1)
    axarr[0].scatter(df['sensitivity'], df['variability'], s=4, alpha=0.5)
    sns.kdeplot(df['sensitivity'], df['variability'], ax=axarr[1], shade=True, cbar=False, shade_lowest=False)
    axarr[0].set_xlim(axarr[1].get_xlim())
    axarr[0].set_ylim(axarr[1].get_ylim())
    for ax in axarr:
        ax.set_xlabel('sensitivity')
        ax.set_ylabel('variability')
    axarr[0].set_title('dataset: ' + dataset + ', model: ' + model + ', t: ' + str(t) + ' (variable init)'*diffinit)
    plt.tight_layout()
    beautify_axes(axarr)
    return True
