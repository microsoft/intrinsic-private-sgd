#!/usr/bin/env ipython
# perform analyses on the results!

# may need this for fonts
# sudo apt-get install ttf-mscorefonts-installer
# sudo apt-get install texlive-full

import numpy as np
import pandas as pd
import re
from scipy.stats import ttest_rel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#params = {#'text.usetex': True,
        #        'text.latex.preamble': [r'\usepackage{cmbright}', r'\usepackage{amsmath}'],
params={'font.family': 'sans-serif',
        #        'font.sans-serif': 'Andale',
        'font.size': 10}
plt.rcParams.update(params)

from matplotlib import cm
import random
import seaborn as sns
import ipdb

import eval_utils
import vis_utils

### DEFINE SOME CONSTANTS !
augment_colour = '#14894e'
both_colour = 'black'
augment_diffinit_colour = '#441e85'
both_diffinit_colour = 'black'
bolton_colour = '#c3871c'

#N_WEIGHTS = 407050

def calculate_epsilon(dataset, model, t, use_bound=False, diffinit=True, num_deltas='max'):
    """
    just get the intrinsic epsilon
    """
    task, batch_size, lr, n_weights, N = eval_utils.get_experiment_details(dataset, model)
    delta = 1.0/N
    if use_bound:
        sensitivity = eval_utils.compute_wu_bound(lipschitz_constant=np.sqrt(2), t=t, N=N, batch_size=batch_size, eta=lr)
    else:
        sensitivity = eval_utils.estimate_sensitivity_empirically(dataset, model, t, num_deltas=num_deltas, diffinit=diffinit)
    variability = eval_utils.estimate_variability(dataset, model, t, by_parameter=False, diffinit=diffinit)
    print('sensitivity:', sensitivity)
    print('variability:', variability)
    print('delta:', delta)
    c = np.sqrt(2 * np.log(1.25/delta))
    epsilon = c * sensitivity / variability
    return epsilon

def accuracy_at_eps(dataset, model, t, use_bound=False, num_experiments=50, num_deltas='max', epsilon=1, do_test=False):
    """
    """
    path = './fig_data/utility.' + str(dataset) + '.' + str(model) + '.t' + str(t) + '.nd_' + str(num_deltas) + '.ne_' + str(num_experiments) + '.csv'
    try:
        utility_data = pd.read_csv(path)
    except:
        print('ERROR: couldn\'t load', path)
        return False
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
        #wat = diff/gap
        #wat = wat[np.isfinite(wat)]
        #print(100*wat.mean())
        frac_improvement = diff/gap
        frac_improvement[~np.isfinite(frac_improvement)] = 0
        print('Average percent difference of gap:', np.mean(100*frac_improvement))
                #print('Average percent difference of gap:', np.nanmean(100*(diff/gap).dropna()))
        
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
        #frac_improvement = frac_improvement[np.isfinite(frac_improvement)]
        print('Average percent difference of gap:', np.mean(100*frac_improvement))

    results = {'acc': [mean_accuracy, std_accuracy],
            'acc_diffinit': [mean_accuracy_diffinit, std_accuracy_diffinit],
            'bolton': [mean_bolton, std_bolton],
            'noiseless': [mean_noiseless, std_noiseless]}
    return results

### WRAPPER OF WRAPPERS ! ###
def generate_amortised_data(dataset, model, num_pairs, num_experiments, t, metric_to_report='binary_accuracy'):
    """
    need to generate the
    - delta distribution
    - sens-var distribution
    """
    print('delta histogram')
    delta_histogram(dataset, model, t=t, num_deltas='max')
    print('epsilon distribution')
    epsilon_distribution(dataset, model, t=t, delta=None, n_pairs=num_pairs, sensitivity_from='empirical')
    if model == 'logistic':
        epsilon_distribution(dataset, model, t=t, delta=None, n_pairs=num_pairs, sensitivity_from='wu_bound')
    print('utility curve')
    utility_curve(dataset, model, delta=None, t=t, metric_to_report=metric_to_report, verbose=True, num_deltas=num_deltas, diffinit=False, num_experiments=num_experiments)
    return True

def weight_evolution(dataset, model, n_seeds=50, replace_indices=None, iter_range=(None, None), params=['#4', '#2'], diffinit=False, aggregate=False):
    """
    """
    plt.clf()
    plt.close()
    fig, axarr = plt.subplots(nrows=len(params), ncols=1, sharex=True, figsize=(4, 3))
    if aggregate:
        colours = cm.get_cmap('Set1')(np.linspace(0.2, 0.8, len(replace_indices)))
        assert n_seeds > 1
        for i, replace_index in enumerate(replace_indices):
            vary_S = eval_utils.get_posterior_samples(dataset, iter_range, model, replace_index=replace_index, params=params, seeds='all', n_seeds=n_seeds, diffinit=diffinit)
            vary_S_min = vary_S.groupby('t').min()
            vary_S_std = vary_S.groupby('t').std()
            vary_S_max = vary_S.groupby('t').max()
            vary_S_mean = vary_S.groupby('t').mean()
            for j, p in enumerate(params):
                axarr[j].fill_between(vary_S_min.index, vary_S_min[p], vary_S_max[p], alpha=0.1, color=colours[i], label='_legend_')
                axarr[j].fill_between(vary_S_mean.index, vary_S_mean[p] - vary_S_std[p], vary_S_mean[p] + vary_S_std[p], alpha=0.1, color=colours[i], label='_nolegend_', linestyle='--')
                axarr[j].plot(vary_S_min.index, vary_S_mean[p], color=colours[i], alpha=0.7, label='D -' + str(replace_index))
                axarr[j].set_ylabel('weight ' + p)
    else:
        colours = cm.get_cmap('plasma')(np.linspace(0.2, 0.8, n_seeds))
        assert len(replace_indices) == 1
        replace_index = replace_indices[0]
        vary_S = eval_utils.get_posterior_samples(dataset, iter_range, model, replace_index=replace_index, params=params, seeds='all', n_seeds=n_seeds, diffinit=diffinit)
        seeds = vary_S['seed'].unique()
        for i, s in enumerate(seeds):
            vary_Ss = vary_S.loc[vary_S['seed'] == s, :]
            for j, p in enumerate(params):
                axarr[j].plot(vary_Ss['t'], vary_Ss[p], color=colours[i], label='seed ' + str(s), alpha=0.8)
                #axarr[j].scatter(vary_Ss['t'], vary_Ss[p], s=4)
                if i == 0:
                    axarr[j].set_ylabel(r'$\mathbf{w}^{' + p[1:] + '}$')

    #axarr[0].legend()
    axarr[-1].set_xlabel('training steps')
    #axarr[0].set_title('dataset: ' + dataset + ', model: ' + model)
    vis_utils.beautify_axes(np.array([axarr]))
    plt.tight_layout()
    plt.savefig('./plots/analyses/weight_trajectory.' + str(dataset) + '.' + str(model) + '.png')
    plt.savefig('./plots/analyses/weight_trajectory.' + str(dataset) + '.' + str(model) + '.pdf')
    return True

def weight_posterior(dataset, model, replace_indices=None, t=500, param='#0', overlay_normal=False):
    """
    show posterior of weight for two datasets, get all the samples
    """
    iter_range = (t, t+1)
    if replace_indices is None:
        # TODO: sve/load etc.
        print('Loading most different pairs for this setting...')
        pairs = eval_utils.find_different_experiments(dataset, model, iter_range, vary_seed=False, vary_data=True, delta_percentile=99, num_deltas=1000)
        # get first element, get the first part of the pair (this is the drop index)
        replace_indices = [p[0] for p in pairs[0]]
        print('Different pair:', replace_indices)
        print(pairs[0])
    elif replace_indices == 'random':
        print('Picking two *random* replace indices for this setting...')
        df = eval_utils.get_available_results(dataset, model)
        replace_counts = df['replace'].value_counts()
        replaces = replace_counts[replace_counts > 2].index.values
        replace_indices = np.random.choice(replaces, 2, replace=False)
    assert len(replace_indices) == 2
    # now load the data!
    df_1 = eval_utils.get_posterior_samples(dataset, iter_range, model, replace_index=replace_indices[0], params=[param], seeds='all')
    df_2 = eval_utils.get_posterior_samples(dataset, iter_range, model, replace_index=replace_indices[1], params=[param], seeds='all')
    print('Loaded', df_1.shape[0], 'and', df_2.shape[0], 'samples respectively')
    fig, axarr = plt.subplots(nrows=1, ncols=1)
    n_bins = 25
    sns.distplot(df_1[param], ax=axarr, label='D - ' + str(replace_indices[0]), kde=True, color='blue', bins=n_bins, norm_hist=True)
    sns.distplot(df_2[param], ax=axarr, label='D - ' + str(replace_indices[1]), kde=True, color='red', bins=n_bins, norm_hist=True)
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
    plt.savefig('./plots/analyses/weight_posterior.' + str(dataset) + '.' + str(model) + '.' + param + '.png')
    plt.savefig('./plots/analyses/weight_posterior.' + str(dataset) + '.' + str(model) + '.' + param + '.pdf')
    return True

def delta_histogram(dataset, model, num_deltas='max', t=500, include_bounds=False, xlim=None, ylim=None, data_privacy='all', plot=True):
    """
    num_deltas is the number of examples we're using to estimate the histograms
    """
    plt.clf()
    plt.close()
    path_string = './fig_data/delta_histogram.' + str(dataset) + '.' + data_privacy + '.' + str(model) + '.nd_' + str(num_deltas) + '.t_' + str(t) + '.npy'
    try:
        plot_data = np.load(path_string).item()
        vary_both = plot_data['vary_both']
        vary_S = plot_data['vary_S']
        vary_r = plot_data['vary_r']
        print('Loaded from file:', path_string)
    except FileNotFoundError:
        print('Couldn\'t find', path_string)
        # vary-both
        vary_both, identifiers_both = eval_utils.get_deltas(dataset, iter_range=(t, t+1), model=model, vary_seed=True, vary_data=True, num_deltas=num_deltas, diffinit=False, data_privacy=data_privacy)
        # vary-S
        vary_S, identifiers_S = eval_utils.get_deltas(dataset, iter_range=(t, t+1), model=model, vary_seed=False, vary_data=True, num_deltas=num_deltas, diffinit=False, data_privacy=data_privacy)
        # vary-r
        vary_r, identifiers_r = eval_utils.get_deltas(dataset, iter_range=(t, t+1), model=model, vary_seed=True, vary_data=False, num_deltas=num_deltas, diffinit=False, data_privacy=data_privacy)

        # save plot data
        plot_data = {'vary_both': vary_both, 
                'both_identifiers': identifiers_both,
                'vary_S': vary_S, 
                'S_identifiers': identifiers_S,
                'vary_r': vary_r,
                'r_identifiers': identifiers_r}
        np.save(path_string, plot_data)
        print('Saved to file:', path_string)

    path_string_diffinit = './fig_data/delta_histogram.' + str(dataset) + '.' + data_privacy + '.' + str(model) + '.nd_' + str(num_deltas) + '.t_' + str(t) + '.DIFFINIT.npy'
    try:
        plot_data_diffinit = np.load(path_string_diffinit).item()
        vary_both_diffinit = plot_data_diffinit['vary_both']
        vary_S_diffinit = plot_data_diffinit['vary_S']
        vary_r_diffinit = plot_data_diffinit['vary_r']
        print('Loaded from file:', path_string_diffinit)
    except FileNotFoundError:
        # vary-both
        vary_both_diffinit, identifiers_both_diffinit = eval_utils.get_deltas(dataset, iter_range=(t, t+1), model=model, vary_seed=True, vary_data=True, num_deltas=num_deltas, diffinit=True, data_privacy=data_privacy)
        # vary-S
        vary_S_diffinit, identifiers_S_diffinit = eval_utils.get_deltas(dataset, iter_range=(t, t+1), model=model, vary_seed=False, vary_data=True, num_deltas=num_deltas, diffinit=True, data_privacy=data_privacy)
        # vary-r
        vary_r_diffinit, identifiers_r_diffinit = eval_utils.get_deltas(dataset, iter_range=(t, t+1), model=model, vary_seed=True, vary_data=False, num_deltas=num_deltas, diffinit=True, data_privacy=data_privacy)

        # save plot data
        plot_data_diffinit = {'vary_both': vary_both_diffinit, 
                'both_identifiers': identifiers_both_diffinit,
                'vary_S': vary_S_diffinit, 
                'S_identifiers': identifiers_S_diffinit,
                'vary_r': vary_r_diffinit,
                'r_identifiers': identifiers_r_diffinit}
        np.save(path_string_diffinit, plot_data_diffinit)
        print('Saved to file:', path_string_diffinit)

    if plot:
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
                #            norm_hist=True, 
                kde_kws={'linestyle': '--'})
        sns.distplot(vary_r_diffinit, ax=axarr, color=augment_diffinit_colour, label=r'$\Delta_V^{vary}$', kde=True, norm_hist=True)
        sns.distplot(vary_both_diffinit, ax=axarr, 
                color=both_diffinit_colour, 
                label=r'$\Delta_{S+V}^{vary}$', 
                kde=True, 
                hist=False,
                #norm_hist=True
                #            norm_hist=True, 
                kde_kws={'linestyle': ':', 'lw': 2})

        #axarr.set_title('Dataset: ' +  dataset + ', model: ' + model +  ', t:' + str(t))
        if include_bounds:
            assert model == 'logistic'
            lipschitz_constant = np.sqrt(2.0)
            #_, _, lipschitz_constant = eval_utils.estimate_empirical_lipschitz(dataset, model, diffinit=True, iter_range=(None, t+1), n_samples=50)
            #print('using empirical lipschitz constant of', lipschitz_constant)
            _, batch_size, lr, _, N = eval_utils.get_experiment_details(dataset, model, verbose=True)
            wu_bound = eval_utils.compute_wu_bound(lipschitz_constant, t=t, N=N, batch_size=batch_size, eta=lr)
            #axarr.axvline(x=wu_bound, ls='--', color=bolton_colour, label=r'bound on $\Delta_r$')
            axarr.axvline(x=wu_bound, ls='--', color=bolton_colour, label=r'$\hat{\Delta}_S$')
        axarr.legend()
        axarr.set_xlabel(r'$\|w - w^\prime\|$')
        axarr.set_ylabel('density')

        if not xlim is None:
            axarr.set_xlim(xlim)
        if not ylim is None:
            axarr.set_ylim(ylim)

        vis_utils.beautify_axes(np.array([axarr]))
        plt.tight_layout()

        plt.savefig('./plots/analyses/delta_histogram_' + dataset + '_' + data_privacy + '_' + model + '_t' + str(t) + '.png')
        plt.savefig('./plots/analyses/delta_histogram_' + dataset + '_' + data_privacy + '_' + model + '_t' + str(t) + '.pdf')
    return True

def epsilon_distribution(dataset, model, t, delta, n_pairs, 
        which='both',
        sensitivity_from='local', sharex=False, 
        variability_from='empirical', xlim=None, ylim=None,
        data_privacy='all'):
    """
    overlay epsilon dist with and without diffinit
    which  takes values both, vary, fix
    """
    path = './fig_data/sens_var_dist.' + dataset + '.' + data_privacy + '.' + model + '.t' + str(t) + '.np' + str(n_pairs) + '.csv'
    path_diffinit = './fig_data/sens_var_dist.' + dataset + '.' + data_privacy + '.' + model + '.t' + str(t) + '.np' + str(n_pairs) + '.DIFFINIT.csv'
    try:
        df = pd.read_csv(path)
        print('Loaded from file', path)
    except FileNotFoundError:
        print('Couldn\'t load sens and var values from', path, '- computing')
        df = eval_utils.get_sens_and_var_distribution(dataset, model, t, n_pairs=n_pairs, by_parameter=False, diffinit=False)
        df.to_csv(path, header=True, index=False)
    try:
        df_diffinit = pd.read_csv(path_diffinit)
        print('Loaded from file', path_diffinit)
    except FileNotFoundError:
        print('Couldn\'t load sens and var values from', path_diffinit, '- computing')
        df_diffinit = eval_utils.get_sens_and_var_distribution(dataset, model, t, n_pairs=n_pairs, by_parameter=False, diffinit=True)
        df_diffinit.to_csv(path_diffinit, header=True, index=False)
    # now set it all up
    _, batch_size, eta, _, N = eval_utils.get_experiment_details(dataset, model)
    if delta is None:
        delta = 1.0/N
        print('Delta:', delta)
    if not n_pairs is None:
        if df.shape[0] > n_pairs:
            pick_rows = np.random.choice(df.shape[0], n_pairs, replace=False)
            df = df.iloc[pick_rows, :]
        if df_diffinit.shape[0] > n_pairs:
            pick_rows = np.random.choice(df_diffinit.shape[0], n_pairs, replace=False)
            df_diffinit = df_diffinit.iloc[pick_rows, :]
    if sensitivity_from == 'wu_bound':
        assert model == 'logistic'
        lipschitz_constant = np.sqrt(2) 
        sensitivity = eval_utils.compute_wu_bound(lipschitz_constant, t, N, batch_size, eta, verbose=True)
        sensitivity_diffinit = sensitivity
        print('Wu sensitivity bound:', sensitivity)
        #print('Empirical sensitivity as a fraction of this:', df['sensitivity'] / sensitivity)
    elif sensitivity_from == 'empirical':
        sensitivity = eval_utils.estimate_sensitivity_empirically(dataset, model, t, num_deltas='max', diffinit=False)
        sensitivity_diffinit = eval_utils.estimate_sensitivity_empirically(dataset, model, t, num_deltas='max', diffinit=True)
    else:
        sensitivity = df['sensitivity']
        sensitivity_diffinit = df_diffinit['sensitivity']
    c = np.sqrt(2 * np.log(1.25 / delta))
    if variability_from == 'local':
        variability = df['variability']
        variability_diffinit = df_diffinit['variability']
    else:
        variability = eval_utils.estimate_variability(dataset, model, t, by_parameter=False, diffinit=False)
        variability_diffinit = eval_utils.estimate_variability(dataset, model, t, by_parameter=False, diffinit=True)
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
        sns.distplot(epsilon, ax=axarr[0], label=r'$\epsilon^{fix}$', color=augment_colour, bins=n_bins, norm_hist=True, kde=kde)
        axarr[0].axvline(x=max(epsilon), ls='--', color=augment_colour, alpha=1)
    if which in ['both', 'vary']:
        sns.distplot(epsilon_diffinit, ax=axarr[-1], label=r'$\epsilon^{vary}$', color=augment_diffinit_colour, bins=n_bins, norm_hist=True, kde=kde)
        axarr[-1].axvline(x=max(epsilon_diffinit), ls='--', color=augment_diffinit_colour, alpha=1)
    axarr[0].set_xlabel('')
    axarr[-1].set_xlabel(r'pairwise $\epsilon$')
    for ax in axarr:
        ax.set_ylabel('density')
    #if sensitivity_from == 'wu_bound':
    #    axarr[0].set_title('dataset: ' + dataset + ', model: ' + model + ', t: ' + str(t) + ', delta:' + str(np.round(delta, 6)) + '\nsensitivity computed from bound')
    #elif sensitivity_from == 'empirical':
    #    axarr[0].set_title('dataset: ' + dataset + ', model: ' + model + ', t: ' + str(t) + ', delta:' + str(np.round(delta, 6)) + '\nsensitivity computed empirically')
    #elif sensitivity_from == 'local':
    #    axarr[0].set_title('dataset: ' + dataset + ', model: ' + model + ', t: ' + str(t) + ', delta:' + str(np.round(delta, 6)) + '\nsensitivity computed locally')
    #else:
    #    raise ValueError
    if which == 'both':
        for ax in axarr:
            ax.legend()
    if not xlim is None:
        for ax in axarr:
            ax.set_xlim(xlim)
    if not ylim is None:
        for ax in axarr:
            ax.set_ylim(ylim)
    plt.tight_layout()
    vis_utils.beautify_axes(axarr)
    plt.savefig('./plots/analyses/epsilon_distribution_' + str(dataset) + '_' + str(model) + '_' + sensitivity_from + '_' + which + '.png')
    plt.savefig('./plots/analyses/epsilon_distribution_' + str(dataset) + '_' + str(model) + '_' + sensitivity_from + '_' + which + '.pdf')
    return True

def utility_curve(dataset, model, delta, t, metric_to_report='binary_accuracy', verbose=True, num_deltas='max', 
        diffinit=False, num_experiments=50, xlim=None, ylim=None, identifier=None, include_fix=False):
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
    #epsilons = np.array([0.1, 1.0, 10.0])
    epsilons = np.array([0.1, 0.5, 0.625, 0.75, 0.875, 1.0, 1.5, 2.0, 3.0, 3.5, 4.0, 6.0, 7.5, 8.5, 10.0, 15.0, 18.0, 20.0])
    try:
        utility_data = pd.read_csv(path)
        print('Loaded from', path)
    except FileNotFoundError:
        print('Couldn\'t find', path, ' - computing')
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
        df = eval_utils.get_available_results(dataset, model, diffinit=diffinit)
        random_experiments = df.iloc[np.random.choice(df.shape[0], num_experiments), :]
        for i, exp in random_experiments.iterrows():
            exp_seed = exp['seed']
            exp_replace = exp['replace']
            for sensitivity_from_bound in [True, False]:
                if sensitivity_from_bound:
                    if not model == 'logistic':
                        print('Skipping because model is', model, ' - cant get sensitivity from bound')
                        # bound isnt meaningful for this model
                        continue
                for eps in epsilons:
                    results = eval_utils.test_model_with_noise(dataset=dataset, model=model, replace_index=exp_replace, 
                            seed=exp_seed, t=t, epsilon=eps, delta=delta, sensitivity_from_bound=sensitivity_from_bound, 
                            metric_to_report=metric_to_report, verbose=verbose, num_deltas=num_deltas, diffinit=diffinit)
                    noiseless_at_eps, bolton_at_eps, augment_at_eps, augment_with_diffinit_at_eps = results
                    seed.append(exp_seed)
                    replace.append(exp_replace)
                    eps_array.append(eps)
                    noiseless.append(noiseless_at_eps)
                    bolton.append(bolton_at_eps)
                    augment.append(augment_at_eps)
                    augment_diffinit.append(augment_with_diffinit_at_eps)
                    sens_from.append(sensitivity_from_bound)
        utility_data = pd.DataFrame({'seed': seed, 'replace': replace, 'epsilon': eps_array, 
            'noiseless': noiseless, 'bolton': bolton, 'augment': augment, 'augment_diffinit': augment_diffinit, 'sensitivity_from_bound': sens_from})
        utility_data.to_csv(path, header=True, index=False, mode='a')
    # NOW FOR PLOTTING!
    #if True in utility_data['sensitivity_from_bound'].unique():
    #    with_bound = True
    #else:
    #    with_bound = False
    
    #if with_bound:
    fig, axarr = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(4, 2.1))
    #else:
    #    fig, axarr = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
    #    axarr = np.array([axarr])
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
            #minn = 0
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
            axarr[j].plot(df_mean.index, df_mean['noiseless'], label='noiseless', alpha=0.5, c='black')
            axarr[j].fill_between(df_mean.index, df_min['noiseless'], df_max['noiseless'], label='_nolegend_', alpha=0.1, color='black')
            # bolton
            axarr[j].plot(df_mean.index, df_mean['bolton'], label='bolton', alpha=0.5, c=bolton_colour)
            axarr[j].fill_between(df_mean.index, df_min['bolton'], df_max['bolton'], label='_nolegend_', alpha=0.1, color=bolton_colour)
            if include_fix:
                # augment
                axarr[j].plot(df_mean.index, df_mean['augment'], label='augment', alpha=0.5, c=augment_colour)
                axarr[j].fill_between(df_mean.index, df_min['augment'], df_max['augment'], label='_nolegend_', alpha=0.1, color=augment_colour)
            # augment with diffinit
            axarr[j].plot(df_mean.index, df_mean['augment_diffinit'], label='augment_diffinit', alpha=0.5, c=augment_diffinit_colour)
            axarr[j].fill_between(df_mean.index, df_min['augment_diffinit'], df_max['augment_diffinit'], label='_nolegend_', alpha=0.1, color=augment_diffinit_colour)
        else:
            linestyle = '--' if sensitivity_from_bound == False else '-'
            size = 6
            line_alpha = 0.75
            axarr.scatter(df['epsilon'], df['bolton'], label='_nolegend_', 
                    s=size, c=bolton_colour)
            axarr.plot(df['epsilon'], df['bolton'], 
                    label=r'$\sigma_{target}$' if j == 1 else '_nolegend_', 
                    alpha=line_alpha, c=bolton_colour, ls=linestyle)
            if include_fix:
                axarr.scatter(df['epsilon'], df['augment'], label='_nolegend_', s=size, c=augment_colour)
                axarr.plot(df['epsilon'], df['augment'], label=r'$\sigma_{augment}^{fix}$' if j == 1 else '_nolegend_', alpha=line_alpha, c=augment_colour, ls=linestyle)
            axarr.scatter(df['epsilon'], df['augment_diffinit'], label='_nolegend_', s=size, c=augment_diffinit_colour)
            axarr.plot(df['epsilon'], df['augment_diffinit'], label=r'$\sigma_{augment}$' if j == 1 else '_nolegend_', alpha=line_alpha, c=augment_diffinit_colour, ls=linestyle)
    axarr.legend()
    axarr.set_ylabel(label_stub)
    axarr.set_xlabel(r'$\epsilon$')
    if not xlim is None:
        axarr.set_xlim(xlim)
    if not ylim is None:
        axarr.set_ylim(ylim)
    vis_utils.beautify_axes(np.array([axarr]))
    plt.tight_layout()
    print('Reminder, the identifier was', identifier)
    plt.savefig('./plots/analyses/utility_' + str(dataset) + '_' + str(model) + '_withfix'*include_fix + '.png')
    plt.savefig('./plots/analyses/utility_' + str(dataset) + '_' + str(model) + '_withfix'*include_fix + '.pdf')
    return True

def sens_and_var_over_time(dataset, model, num_deltas=500, iter_range=(0, 1000), data_privacy='all', metric='binary_crossentropy'):
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
        print('Didn\'t find', path, ' - creating!')
        # get experiment details
        if model == 'logistic':
            _, batch_size, lr, _, N = eval_utils.get_experiment_details(dataset, model, data_privacy=data_privacy)
            L = np.sqrt(2)
        assert not None in iter_range
        t_range = np.arange(iter_range[0], iter_range[1], 200)
        n_T = len(t_range)
        theoretical_sensitivity_list = [np.nan]*n_T
        empirical_sensitivity_list = [np.nan]*n_T
        variability_fixinit_list = [np.nan]*n_T
        variability_diffinit_list = [np.nan]*n_T
        for i, t in enumerate(t_range):
            # sensitivity
            if model == 'logistic':
                theoretical_sensitivity = eval_utils.compute_wu_bound(L, t=t, N=N, batch_size=batch_size, eta=lr)
            else:
                theoretical_sensitivity = np.nan
            empirical_sensitivity = eval_utils.estimate_sensitivity_empirically(dataset, model, t, num_deltas=num_deltas, diffinit=True, data_privacy=data_privacy)
            if not empirical_sensitivity:
                print('Running delta histogram...')
                delta_histogram(dataset, model, num_deltas=num_deltas, t=t, include_bounds=False, xlim=None, ylim=None, data_privacy=data_privacy, plot=False)
                print('Rerunning empirical sensitivity estimate...')
                empirical_sensitivity = eval_utils.estimate_sensitivity_empirically(dataset, model, t, num_deltas=num_deltas, diffinit=True, data_privacy=data_privacy)
                assert not empirical_sensitivity is False
            # variability
            variability_fixinit = eval_utils.estimate_variability(dataset, model, t, by_parameter=False, diffinit=False, data_privacy=data_privacy)
            variability_diffinit = eval_utils.estimate_variability(dataset, model, t, by_parameter=False, diffinit=True, data_privacy=data_privacy)
            # record everything
            theoretical_sensitivity_list[i] = theoretical_sensitivity
            empirical_sensitivity_list[i] = empirical_sensitivity
            variability_fixinit_list[i] = variability_fixinit
            variability_diffinit_list[i] = variability_diffinit
        df = pd.DataFrame({'t': t_range, 
            'theoretical_sensitivity': theoretical_sensitivity_list,
            'empirical_sensitivity': empirical_sensitivity_list,
            'variability_fixinit': variability_fixinit_list,
            'variability_diffinit': variability_diffinit_list})
        df.set_index('t', inplace=True)
        # now join the losses... 
        # (actually we can just load the losses as needed)
        losses = eval_utils.aggregated_loss(dataset, model, iter_range=iter_range, data_privacy=data_privacy)
        df = df.join(losses)
        ###
        df.to_csv(path)
    fig, axarr = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(3.5, 4.2))
    # losse 
    losses = eval_utils.aggregated_loss(dataset, model, iter_range=iter_range, data_privacy=data_privacy)
    losses = losses.loc[losses.index < iter_range[1], :]
    train_loss = losses[metric + '_mean_train']
    vali_loss = losses[metric + '_mean_vali']
    train_loss_std = losses[metric + '_std_train']
    vali_loss_std = losses[metric + '_std_vali']
    axarr[0].scatter(losses.index, train_loss, label='_nolegend_', s=6, c='black')
    axarr[0].plot(losses.index, train_loss, alpha=0.5, label='train', c='black')
    axarr[0].fill_between(losses.index, train_loss - train_loss_std, train_loss + train_loss_std, label='_nolegend_', alpha=0.2, color='black')
    axarr[0].scatter(losses.index, vali_loss, label='_nolegend_', s=6, c='grey')
    axarr[0].plot(losses.index, vali_loss, ls='--', label='validation', alpha=0.5, c='grey')
    axarr[0].fill_between(losses.index, vali_loss - vali_loss_std, vali_loss + vali_loss_std, label='_nolegend_', alpha=0.2, color='grey')
    axarr[0].set_ylabel(re.sub('_', '\n', metric))
    # sensitivity
    #axarr[1].scatter(df['t'], df['theoretical_sensitivity'], label='_nolegend_', s=6, c=bolton_colour)
    ds = [np.nan]*df.shape[0]
    for i, ts in enumerate(df['theoretical_sensitivity'].values):
        ds[i] = eval_utils.discretise_theoretical_sensitivity(dataset, model, ts)
    axarr[1].plot(df['t'], ds, label='theoretical', alpha=0.5, c=bolton_colour, ls='--')
    axarr[1].scatter(df['t'], df['empirical_sensitivity'], label='_nolegend_', s=6, c=bolton_colour)
    axarr[1].plot(df['t'], df['empirical_sensitivity'], label='empirical', alpha=0.5, c=bolton_colour)
    axarr[1].set_ylabel('sensitivity')
    # variability
    axarr[2].scatter(df['t'], df['variability_diffinit'], label='variable init', s=6, c=augment_diffinit_colour)
    axarr[2].plot(df['t'], df['variability_diffinit'], label='_nolegend_', alpha=0.5, c=augment_diffinit_colour)
    axarr[2].scatter(df['t'], df['variability_fixinit'], label='fixed init', s=6, c=augment_colour)
    axarr[2].plot(df['t'], df['variability_fixinit'], label='_nolegend_', alpha=0.5, c=augment_colour)
    axarr[2].set_ylabel(r'$\sigma_i$')
    # shared things
    axarr[-1].set_xlabel('steps of SGD')
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
