#!/usr/bin/env ipython
# This script is for performing specific analyses/generating figures
# It relies on e.g. statistics computed across experiments - "derived results"

# may need this for fonts
# sudo apt-get install ttf-mscorefonts-installer
# sudo apt-get install texlive-full

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


def generate_plots(cfg_name: str, model: str, t=None) -> None:
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
        delta_histogram_xlim = plot_options['delta_histogram']['xlim']
        delta_histogram_ylim = plot_options['delta_histogram']['ylim']
        epsilon_distribution_xlim = plot_options['epsilon_distribution']['xlim']
        epsilon_distribution_ylim = plot_options['epsilon_distribution']['ylim']
        versus_time_acc_lims = plot_options['versus_time']['acc_lim']
    except FileNotFoundError:
        assert t is not None
        delta_histogram_xlim = None
        delta_histogram_ylim = None
        epsilon_distribution_xlim = None
        epsilon_distribution_ylim = None
        versus_time_acc_lims = None

    plot_delta_histogram(cfg_name, model, t=t, include_bounds=(model == 'logistic'),
                         xlim=delta_histogram_xlim, ylim=delta_histogram_ylim)
    plot_epsilon_distribution(cfg_name, model, t=t,
                              xlim=epsilon_distribution_xlim, ylim=epsilon_distribution_ylim)
    plot_sens_and_var_over_time(cfg_name, model, iter_range=(0, int(t*1.2)),
                                acc_lims=versus_time_acc_lims)
    plot_stability_of_estimated_values(cfg_name, model, t)

    return


def plot_delta_histogram(cfg_name: str, model: str, num_deltas='max', t=500,
                         include_bounds=False, xlim=None, ylim=None,
                         data_privacy='all', multivariate=False) -> None:
    """
    num_deltas is the number of examples we're using to estimate the histograms
    """

    if multivariate:
        raise NotImplementedError('Multivariate plotting is not implemented')
    delta_histogram = dr.DeltaHistogram(cfg_name, model, num_deltas, t, data_privacy, multivariate)
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
    sns.distplot(vary_S, ax=axarr,
                 color=em.dp_colours['bolton'],
                 label=r'$\Delta_S$',
                 kde=True,
                 norm_hist=True)
    print('Plotting varying r... number of deltas:', vary_r.shape[0])
    sns.distplot(vary_r, ax=axarr,
                 color=em.dp_colours['augment'],
                 label=r'$\Delta_V^{fix}$',
                 kde=True,
                 norm_hist=True)
    sns.distplot(vary_r_diffinit, ax=axarr,
                 color=em.dp_colours['augment_diffinit'],
                 label=r'$\Delta_V^{vary}$',
                 kde=True,
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

    axarr.legend()
    axarr.set_xlabel(r'$\|w - w^\prime\|$')
    axarr.set_ylabel('density')

    if xlim is not None:
        axarr.set_xlim(xlim)

    if ylim is not None:
        axarr.set_ylim(ylim)

    vis_utils.beautify_axes(np.array([axarr]))
    plt.tight_layout()

    figure_identifier = f'delta_histogram_{cfg_name}_{data_privacy}_{model}_t{t}'
    plt.savefig((FIGS_DIR / figure_identifier).with_suffix('.png'))
    plt.savefig((FIGS_DIR / figure_identifier).with_suffix('.pdf'))

    return


def plot_epsilon_distribution(cfg_name, model, t, delta=None, num_pairs='max',
                              which='vary',
                              sensitivity_from='local', sharex=True,
                              variability_from='empirical', xlim=None, ylim=None,
                              data_privacy='all') -> None:
    """
    overlay epsilon dist with and without diffinit
    which  takes values both, vary, fix
    """
    sens_var = dr.SensVar(cfg_name, model, data_privacy=data_privacy, t=t, num_pairs=num_pairs)
    df = sens_var.load(diffinit=False)
    df_diffinit = sens_var.load(diffinit=True)

    # Now set it all up
    _, batch_size, eta, _, N = em.get_experiment_details(cfg_name, model)

    if delta is None:
        delta = 1.0/(N**2)
        print('Delta:', delta)

    if not (num_pairs is None or num_pairs == 'max'):
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
        sensitivity = dr.estimate_sensitivity_empirically(cfg_name, model,
                                                          t, num_deltas='max',
                                                          diffinit=False)
        sensitivity_diffinit = dr.estimate_sensitivity_empirically(cfg_name, model,
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
        variability = dr.estimate_variability(cfg_name, model, t, multivariate=False, diffinit=False)
        variability_diffinit = dr.estimate_variability(cfg_name, model, t, multivariate=False, diffinit=True)
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
                     color=em.dp_colours['augment'], bins=n_bins, norm_hist=True, kde=kde)

    if which in ['both', 'vary']:
        sns.distplot(epsilon_diffinit, ax=axarr[-1], label=r'$\epsilon^{vary}$',
                     color=em.dp_colours['augment_diffinit'], bins=n_bins, norm_hist=True, kde=kde)
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

    figure_identifier = f'epsilon_distribution_{which}_{cfg_name}_{data_privacy}_{model}_t{t}_{sensitivity_from}'
    plt.savefig((FIGS_DIR / figure_identifier).with_suffix('.png'))
    plt.savefig((FIGS_DIR / figure_identifier).with_suffix('.pdf'))

    return


def plot_sens_and_var_over_time(cfg_name, model, num_deltas='max', iter_range=(0, 1000),
                                data_privacy='all', metric='binary_crossentropy', acc_lims=None) -> None:
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
    df = dr.VersusTime(cfg_name, model).load()

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
        ds[i] = test_private_model.discretise_theoretical_sensitivity(cfg_name, model, ts)
    axarr[1].plot(df['t'], ds, label='theoretical', alpha=0.5, c=em.dp_colours['bolton'], ls='--')
    axarr[1].scatter(df['t'], df['empirical_sensitivity'], label='_nolegend_', s=6, c=em.dp_colours['bolton'])
    axarr[1].plot(df['t'], df['empirical_sensitivity'], label='empirical', alpha=0.5, c=em.dp_colours['bolton'])
    axarr[1].set_ylabel('sensitivity')
    # variability
    axarr[2].scatter(df['t'], df['variability_diffinit'],
                     label='variable init', s=6,
                     c=em.dp_colours['augment_diffinit'])
    axarr[2].plot(df['t'], df['variability_diffinit'],
                  label='_nolegend_', alpha=0.5,
                  c=em.dp_colours['augment_diffinit'])
    axarr[2].scatter(df['t'], df['variability_fixinit'],
                     label='fixed init', s=6,
                     c=em.dp_colours['augment'])
    axarr[2].plot(df['t'], df['variability_fixinit'],
                  label='_nolegend_', alpha=0.5,
                  c=em.dp_colours['augment'])
    axarr[2].set_ylabel(r'$\sigma_i(\mathcal{D})$')
    # shared things
    axarr[-1].set_xlabel('T (number of steps)')

    if model == 'logistic':
        convergence_points = em.lr_convergence_points
        title = em.dataset_names[cfg_name] + ' (logistic regression)'
    else:
        convergence_points = em.nn_convergence_points
        title = em.dataset_names[cfg_name] + ' (neural network)'
    convergence_point = convergence_points[cfg_name]

    for ax in axarr:
        if convergence_point < iter_range[1]:
            # only include the convergence point if it fits in the plot as specified
            ax.axvline(x=convergence_point, ls='--', color='black', alpha=0.5)
        ax.legend()
    axarr[0].set_title(title)

    for ax in axarr:
        ax.legend()
    vis_utils.beautify_axes(np.array([axarr]))

    # save
    plt.tight_layout()
    figure_identifier = f'versus_time_{cfg_name}_{data_privacy}_{model}_nd{num_deltas}'
    plt.savefig((FIGS_DIR / figure_identifier).with_suffix('.png'))
    plt.savefig((FIGS_DIR / figure_identifier).with_suffix('.pdf'))

    plt.clf()
    plt.close()

    return


def plot_stability_of_estimated_values(cfg_name, model, t) -> None:
    """
    """
    stability = dr.Stability(cfg_name, model, t)
    stability_dict = stability.load()

    # lets just do 3 separate plots
    figsize = (3.5, 2.8)
    size = 6
    # SIGMA V N SEEDS
    print('Plotting sigma v seeds')
    sigma_df = stability_dict['sigma']
    sigma_v_seed = sigma_df[['num_seeds', 'sigma']]
#    sigma_v_seed = sigma_df[sigma_df['num_replaces'] == sigma_df['num_replaces'].max()][['num_seeds', 'sigma']]
    fig, axarr = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    axarr.scatter(sigma_v_seed['num_seeds'], sigma_v_seed['sigma'], s=size, c=em.dp_colours['augment_diffinit'])
    sigma_we_use = dr.estimate_variability(cfg_name, model, t, diffinit=True)
    axarr.axhline(y=sigma_we_use, ls='--', c=em.dp_colours['augment_diffinit'], alpha=0.4)
    axarr.set_xlabel('number of random seeds')
    axarr.set_ylabel(r'estimated $\sigma_i(\mathcal{D})$')
    axarr.set_title(em.dataset_names[cfg_name] + ' (' + em.model_names[model] + ')')
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
    axarr.set_title(em.dataset_names[cfg_name] + ' (' + em.model_names[model] + ')')
    vis_utils.beautify_axes(np.array([axarr]))
    plt.tight_layout()

    figure_identifier = f'stability_sens_v_deltas_{cfg_name}_{model}_t{t}'
    plt.savefig((FIGS_DIR / figure_identifier).with_suffix('.png'))
    plt.savefig((FIGS_DIR / figure_identifier).with_suffix('.pdf'))

    plt.clf()
    plt.close()

    return


def compare_mnist_variants() -> None:
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
    figure_identifier = 'mnist_preprocessing_comparison'
    plt.savefig((FIGS_DIR / figure_identifier).with_suffix('.png'))
    plt.savefig((FIGS_DIR / figure_identifier).with_suffix('.pdf'))

    plt.clf()
    plt.close()
    # ------- compare the traces ----- #
    pca_loss = dr.AggregatedLoss('mnist_binary_pca', 'logistic').load(diffinit=True)
    grp_loss = dr.AggregatedLoss('mnist_binary', 'logistic').load(diffinit=True)
    cropped_loss = dr.AggregatedLoss('mnist_binary_cropped', 'logistic').load(diffinit=True)
    fig, axarr = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(3.5, 3.5))
    size = 6
    # first row is crossentropy, second is accuracy

    for i, metric in enumerate(['binary_crossentropy', 'binary_accuracy']):
        axarr[i].scatter(pca_loss.index, pca_loss[f'{metric}_mean_train'], label='PCA', s=size, color=colours[0])
        axarr[i].scatter(grp_loss.index, grp_loss[f'{metric}_mean_train'], label='GRP', s=size, color=colours[1])
        axarr[i].scatter(cropped_loss.index, cropped_loss[f'{metric}_mean_train'],
                         label='Crop', s=size, color=colours[2])
        axarr[i].plot(pca_loss.index, pca_loss[f'{metric}_mean_train'],
                      label='_nolegend_', color=colours[0], alpha=0.5)
        axarr[i].plot(grp_loss.index, grp_loss[f'{metric}_mean_train'],
                      label='_nolegend_', color=colours[1], alpha=0.5)
        axarr[i].plot(cropped_loss.index, cropped_loss[f'{metric}_mean_train'],
                      label='_nolegend_', color=colours[2], alpha=0.5)
        axarr[i].fill_between(pca_loss.index,
                              pca_loss[f'{metric}_mean_train'] - pca_loss[f'{metric}_std_train'],
                              pca_loss[f'{metric}_mean_train'] + pca_loss[f'{metric}_std_train'],
                              label='_nolegend_', alpha=0.2, color=colours[0])
        axarr[i].fill_between(grp_loss.index,
                              grp_loss[f'{metric}_mean_train'] - grp_loss[f'{metric}_std_train'],
                              grp_loss[f'{metric}_mean_train'] + grp_loss[f'{metric}_std_train'],
                              label='_nolegend_', alpha=0.2, color=colours[1])
        axarr[i].fill_between(cropped_loss.index,
                              cropped_loss[f'{metric}_mean_train'] - cropped_loss[f'{metric}_std_train'],
                              cropped_loss[f'{metric}_mean_train'] + cropped_loss[f'{metric}_std_train'],
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
    figure_identifier = 'mnist_preprocessing_comparison_trace'
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
    figure_identifier = f'pval_histogram_{plot_label}_{model}'
    plt.savefig((FIGS_DIR / figure_identifier).with_suffix('.png'))
    plt.savefig((FIGS_DIR / figure_identifier).with_suffix('.pdf'))

    return


def overlay_eps_plot(model='logistic', cfg_names=None, xlim=None, ylim=None, title=None) -> None:
    figsize = (3.7, 2.9)
    n_bins = 50

    if model == 'logistic':
        convergence_points = em.lr_convergence_points
    else:
        convergence_points = em.nn_convergence_points

    if cfg_names is None:
        cfg_names = em.dataset_colours.keys()
        plot_label = '_'
    else:
        plot_label = '_'.join(cfg_names) + '_'
    fig, axarr = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    for ds in cfg_names:
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
    figure_identifier = f'epsilon_distribution_{plot_label}_{model}'
    plt.savefig((FIGS_DIR / figure_identifier).with_suffix('.png'))
    plt.savefig((FIGS_DIR / figure_identifier).with_suffix('.pdf'))

    plt.clf()
    plt.close()

    return
