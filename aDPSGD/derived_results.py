#!/usr/bin/env ipython
# Generate derived results
# NO PLOTTING HERE
###

# may need this for fonts
# sudo apt-get install ttf-mscorefonts-installer
# sudo apt-get install texlive-full

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
import ipdb

import test_private_model
import results_utils
import stats_utils
import experiment_metadata


def calculate_epsilon(dataset, model, t, use_bound=False, diffinit=True, num_deltas='max', multivariate=False):
    """
    just get the intrinsic epsilon
    """
    task, batch_size, lr, n_weights, N = test_private_model.get_experiment_details(dataset, model)
    delta = 1.0/(N**2)
    variability = test_private_model.estimate_variability(dataset, model, t, multivariate=False,
                                                          diffinit=diffinit)

    if use_bound:
        sensitivity = test_private_model.compute_wu_bound(lipschitz_constant=np.sqrt(2), t=t, N=N,
                                                          batch_size=batch_size, eta=lr)

        if multivariate:
            sensitivity = np.array([sensitivity]*len(variability))
    else:
        sensitivity = test_private_model.estimate_sensitivity_empirically(dataset, model, t,
                                                                          num_deltas=num_deltas,
                                                                          diffinit=diffinit,
                                                                          multivariate=multivariate)
    print('sensitivity:', sensitivity)
    print('variability:', variability)
    print('delta:', delta)
    c = np.sqrt(2 * np.log(1.25/delta))
    epsilon = c * sensitivity / variability

    return epsilon


def accuracy_at_eps(dataset, model, t, use_bound=False, num_experiments=500,
                    num_deltas='max', epsilon=1, do_test=False):
    """
    """
    path = './fig_data/utility.' + str(dataset) + '.' + str(model) + '.t' + str(t) + '.nd_' + str(num_deltas) + '.ne_' + str(num_experiments) + '.csv'
    try:
        utility_data = pd.read_csv(path)
    except FileNotFoundError:
        print('ERROR: couldn\'t load', path)

        return False

    if use_bound:
        utility_data = utility_data.loc[utility_data['sensitivity_from_bound'] is True, :]
    else:
        utility_data = utility_data.loc[utility_data['sensitivity_from_bound'] is False, :]
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


def generate_amortised_data(dataset, model, num_pairs, num_experiments, t, metric_to_report='binary_accuracy'):
    """
    need to generate the
    - delta distribution
    - sens-var distribution
    """
    print('delta histogram')
    generate_delta_histogram(dataset, model, t=t, num_deltas='max')
    print('epsilon distribution')

    if model == 'logistic':
        generate_epsilon_distribution(dataset, model, t=t, delta=None, n_pairs=num_pairs, sensitivity_from='wu_bound')
    else:
        generate_epsilon_distribution(dataset, model, t=t, delta=None, n_pairs=num_pairs, sensitivity_from='empirical')
    print('utility curve')
    generate_utility_curve(dataset, model, delta=None, t=t, metric_to_report=metric_to_report,
                           verbose=True, num_deltas='max', diffinit=False, num_experiments=num_experiments)

    return True


def generate_delta_histogram(dataset, model, num_deltas='max', t=500, include_bounds=False,
                             xlim=None, ylim=None, data_privacy='all', multivariate=False):
    """
    num_deltas is the number of examples we're using to estimate the histograms
    """
    path_string = './fig_data/delta_histogram.' + str(dataset) + '.' + data_privacy + '.' + str(model) + '.nd_' + str(num_deltas) + '.t_' + str(t) + 'MULTIVAR'*multivariate + '.npy'
    try:
        plot_data = np.load(path_string).item()
        vary_both = plot_data['vary_both']
        vary_S = plot_data['vary_S']
        vary_r = plot_data['vary_r']
        print('Loaded from file:', path_string)
    except FileNotFoundError:
        print('Couldn\'t find', path_string)
        # vary-both
        vary_both, identifiers_both = get_deltas(dataset, iter_range=(t, t+1), model=model,
                                                 vary_seed=True, vary_data=True,
                                                 num_deltas=num_deltas, diffinit=False,
                                                 data_privacy=data_privacy, multivariate=multivariate)
        # vary-S
        vary_S, identifiers_S = get_deltas(dataset, iter_range=(t, t+1), model=model,
                                           vary_seed=False, vary_data=True,
                                           num_deltas=num_deltas, diffinit=False,
                                           data_privacy=data_privacy, multivariate=multivariate)
        # vary-r
        vary_r, identifiers_r = get_deltas(dataset, iter_range=(t, t+1), model=model,
                                           vary_seed=True, vary_data=False,
                                           num_deltas=num_deltas, diffinit=False,
                                           data_privacy=data_privacy, multivariate=multivariate)

        # save plot data
        plot_data = {'vary_both': vary_both,
                     'both_identifiers': identifiers_both,
                     'vary_S': vary_S,
                     'S_identifiers': identifiers_S,
                     'vary_r': vary_r,
                     'r_identifiers': identifiers_r}
        np.save(path_string, plot_data)
        print('Saved to file:', path_string)

    path_string_diffinit = './fig_data/delta_histogram.' + str(dataset) + '.' + data_privacy + '.' + str(model) + '.nd_' + str(num_deltas) + '.t_' + str(t) + '.DIFFINIT' + 'MULTIVAR'*multivariate + '.npy'
    try:
        plot_data_diffinit = np.load(path_string_diffinit).item()
        vary_both_diffinit = plot_data_diffinit['vary_both']
        vary_S_diffinit = plot_data_diffinit['vary_S']
        vary_r_diffinit = plot_data_diffinit['vary_r']
        print('Loaded from file:', path_string_diffinit)
    except FileNotFoundError:
        # vary-both
        vary_both_diffinit, identifiers_both_diffinit = get_deltas(dataset, iter_range=(t, t+1),
                                                                   model=model, vary_seed=True,
                                                                   vary_data=True, num_deltas=num_deltas,
                                                                   diffinit=True,
                                                                   data_privacy=data_privacy,
                                                                   multivariate=multivariate)
        # vary-S
        vary_S_diffinit, identifiers_S_diffinit = get_deltas(dataset, iter_range=(t, t+1),
                                                             model=model, vary_seed=False,
                                                             vary_data=True, num_deltas=num_deltas,
                                                             diffinit=True,
                                                             data_privacy=data_privacy,
                                                             multivariate=multivariate)
        # vary-r
        vary_r_diffinit, identifiers_r_diffinit = get_deltas(dataset, iter_range=(t, t+1),
                                                             model=model, vary_seed=True,
                                                             vary_data=False, num_deltas=num_deltas,
                                                             diffinit=True,
                                                             data_privacy=data_privacy,
                                                             multivariate=multivariate)

        # save plot data
        plot_data_diffinit = {'vary_both': vary_both_diffinit,
                              'both_identifiers': identifiers_both_diffinit,
                              'vary_S': vary_S_diffinit,
                              'S_identifiers': identifiers_S_diffinit,
                              'vary_r': vary_r_diffinit,
                              'r_identifiers': identifiers_r_diffinit}
        np.save(path_string_diffinit, plot_data_diffinit)
        print('Saved to file:', path_string_diffinit)

    return True


def generate_epsilon_distribution(dataset, model, t, delta, n_pairs,
                                  which='both',
                                  sensitivity_from='local', sharex=False,
                                  variability_from='empirical', xlim=None, ylim=None,
                                  data_privacy='all'):
    """
    """
    path = './fig_data/sens_var_dist.' + dataset + '.' + data_privacy + '.' + model + '.t' + str(t) + '.np' + str(n_pairs) + '.csv'
    path_diffinit = './fig_data/sens_var_dist.' + dataset + '.' + data_privacy + '.' + model + '.t' + str(t) + '.np' + str(n_pairs) + '.DIFFINIT.csv'
    try:
        df = pd.read_csv(path)
        print('Loaded from file', path)
    except FileNotFoundError:
        print('Couldn\'t load sens and var values from', path, '- computing')
        df = get_sens_and_var_distribution(dataset, model, t, n_pairs=n_pairs,
                                           multivariate=False, diffinit=False)
        df.to_csv(path, header=True, index=False)
    try:
        df_diffinit = pd.read_csv(path_diffinit)
        print('Loaded from file', path_diffinit)
    except FileNotFoundError:
        print('Couldn\'t load sens and var values from', path_diffinit, '- computing')
        df_diffinit = get_sens_and_var_distribution(dataset, model, t, n_pairs=n_pairs,
                                                    multivariate=False, diffinit=True)
        df_diffinit.to_csv(path_diffinit, header=True, index=False)

    return df, df_diffinit


def generate_utility_curve(dataset, model, delta, t, metric_to_report='binary_accuracy',
                           verbose=True, num_deltas='max', diffinit=False,
                           num_experiments=50, xlim=None, ylim=None, identifier=None, include_fix=False):
    """
    for a single model (this is how it is right now), plot
    performance v. epsilon at fixed delta
    for
    - noiseless
    - bolt-on
    - augmenting SGD (us) (+ including random initialisation)
    """
    path = './fig_data/utility.' + str(dataset) + '.' + str(model) + '.t' + str(t) + '.nd_' + str(num_deltas) + '.ne_' + str(num_experiments) + '.csv'
    try:
        utility_data = pd.read_csv(path)
        print('Loaded from', path)
    except FileNotFoundError:
        print('Couldn\'t find', path, ' - computing')
        epsilons = np.array([0.1, 0.5, 0.625, 0.75, 0.875, 1.0, 1.5, 2.0, 3.0, \
                             3.5, 4.0, 6.0, 7.5, 8.5, 10.0, 15.0, 18.0, 20.0])
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
        df = results_utils.get_available_results(dataset, model, diffinit=diffinit)
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
                    results = test_private_model.test_model_with_noise(dataset=dataset, model=model,
                                                                       replace_index=exp_replace,
                                                                       seed=exp_seed, t=t, epsilon=eps,
                                                                       delta=delta,
                                                                       sensitivity_from_bound=sensitivity_from_bound,
                                                                       metric_to_report=metric_to_report,
                                                                       verbose=verbose,
                                                                       num_deltas=num_deltas,
                                                                       diffinit=diffinit)
                    noiseless_at_eps, bolton_at_eps, augment_at_eps, augment_with_diffinit_at_eps = results
                    seed.append(exp_seed)
                    replace.append(exp_replace)
                    eps_array.append(eps)
                    noiseless.append(noiseless_at_eps)
                    bolton.append(bolton_at_eps)
                    augment.append(augment_at_eps)
                    augment_diffinit.append(augment_with_diffinit_at_eps)
                    sens_from.append(sensitivity_from_bound)
        utility_data = pd.DataFrame({'seed': seed, 'replace': replace,
                                     'epsilon': eps_array, 'noiseless': noiseless,
                                     'bolton': bolton, 'augment': augment,
                                     'augment_diffinit': augment_diffinit,
                                     'sensitivity_from_bound': sens_from})
        utility_data.to_csv(path, header=True, index=False, mode='a')

    return True


def sens_and_var_over_time(dataset, model, num_deltas=500, iter_range=(0, 1000),
                           data_privacy='all', metric='binary_crossentropy', cadence=200):
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
            _, batch_size, lr, _, N = experiment_metadata.get_experiment_details(dataset, model,
                                                                                 data_privacy=data_privacy)
            L = np.sqrt(2)
        assert None not in iter_range
        t_range = np.arange(iter_range[0], iter_range[1], cadence)
        n_T = len(t_range)
        theoretical_sensitivity_list = [np.nan]*n_T
        empirical_sensitivity_list = [np.nan]*n_T
        variability_fixinit_list = [np.nan]*n_T
        variability_diffinit_list = [np.nan]*n_T

        for i, t in enumerate(t_range):
            # sensitivity

            if model == 'logistic':
                theoretical_sensitivity = test_private_model.compute_wu_bound(L, t=t, N=N,
                                                                              batch_size=batch_size, eta=lr)
            else:
                theoretical_sensitivity = np.nan
            empirical_sensitivity = estimate_sensitivity_empirically(dataset, model, t,
                                                                     num_deltas=num_deltas,
                                                                     diffinit=True, data_privacy=data_privacy)

            if not empirical_sensitivity:
                print('Running delta histogram...')
                generate_delta_histogram(dataset, model, num_deltas=num_deltas,
                                         t=t, include_bounds=False,
                                         xlim=None, ylim=None,
                                         data_privacy=data_privacy, plot=False)
                print('Rerunning empirical sensitivity estimate...')
                empirical_sensitivity = estimate_sensitivity_empirically(dataset, model, t,
                                                                         num_deltas=num_deltas,
                                                                         diffinit=True,
                                                                         data_privacy=data_privacy)
                assert empirical_sensitivity is not False
            # variability
            variability_fixinit = estimate_variability(dataset, model, t, multivariate=False,
                                                       diffinit=False, data_privacy=data_privacy)
            variability_diffinit = estimate_variability(dataset, model, t, multivariate=False,
                                                        diffinit=True, data_privacy=data_privacy)
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
        losses = aggregated_loss(dataset, model, iter_range=iter_range, data_privacy=data_privacy)
        df = df.join(losses)
        ###
        df.to_csv(path)

    return True


def estimate_empirical_lipschitz(dataset, model, diffinit, iter_range, n_samples=5000):
    """
    get the biggest gradient during training

    NOTE: using 10k samples and all time-points,
    - on housing+linear we get max_L ~ 2.49, ave_L ~ 0.33, min_L ~ 0.06

    TODO: amortise
    """
    max_norm = 0
    min_norm = 50
    cumulative = 0
    cum_count = 0
    df = results_utils.get_available_results(dataset, model, replace_index=None, diffinit=diffinit, data_privacy='all')
    n_exp = df.shape[0]

    if n_samples is None:
        print('Selecting', n_exp, 'experiments')
        experiments = df
    elif n_samples > n_exp:
        print('WARNING: Only', n_exp, 'experiments available - selecting all')
        experiments = df
    else:
        row_picks = np.random.choice(n_exp, n_samples, replace=False)
        experiments = df.iloc[row_picks, :]

    for row, exp in experiments.iterrows():
        replace = exp['replace']
        seed = exp['seed']
        experiment = results_utils.ExperimentIdentifier(dataset, model, replace, seed, diffinit)
        gradients = experiment.load_gradients(iter_range=iter_range)
        grad_norm = np.linalg.norm(gradients.iloc[:, 2:], axis=1)
        cumulative += np.sum(grad_norm)
        cum_count += grad_norm.shape[0]
        max_grad = np.max(grad_norm)
        min_grad = np.min(grad_norm)

        if max_grad > max_norm:
            max_norm = max_grad

        if min_grad < min_norm:
            min_norm = min_grad
    ave_norm = cumulative/cum_count

    return min_norm, ave_norm, max_norm


def estimate_sensitivity_empirically(dataset, model, t, num_deltas, diffinit=False,
                                     data_privacy='all', multivariate=False):
    """ pull up the histogram
    """
    path_string = './fig_data/delta_histogram.' + str(dataset) + '.' + data_privacy + '.' + str(model) + '.nd_' + str(num_deltas) + '.t_' + str(t) + '.DIFFINIT'*diffinit + 'MULTIVAR'*multivariate + '.npy'
    try:
        plot_data = np.load(path_string).item()
    except FileNotFoundError:
        print('[estimate_sensitivty_empirically] ERROR: Run delta_histogram for this setting first:', path_string)

        return False
    vary_data_deltas = plot_data['vary_S']
    sensitivity = np.nanmax(vary_data_deltas, axis=0)

    return sensitivity


def get_deltas(dataset, iter_range, model,
               vary_seed=True, vary_data=True, params=None, num_deltas=100,
               include_identifiers=False, diffinit=False, data_privacy='all',
               multivariate=False, verbose=False):
    """
    collect samples of weights from experiments on dataset+model, varying:
    - seed (vary_seed)
    - data (vary_data)

    to clarify, we want to estimate |w(S, r) - w(S', r')|,
    with potentially S' = S (vary_data = False), or r' = r (vary_seed = False)

    we need to make sure that we only compare like-with-like!

    we want to get num_deltas values of delta in the end
    """
    df = results_utils.get_available_results(dataset, model, diffinit=diffinit, data_privacy=data_privacy)

    if num_deltas == 'max':
        num_deltas = int(df.shape[0]/2)
        print('Using num_deltas:', num_deltas)

    if df.shape[0] < 2*num_deltas:
        print('ERROR: Run more experiments, or set num_deltas to be at most', int(df.shape[0]/2))

        return False
    w_rows = np.random.choice(df.shape[0], num_deltas, replace=False)
    remaining_rows = [x for x in range(df.shape[0]) if x not in w_rows]
    df_remaining = df.iloc[remaining_rows]
    seed_options = df_remaining['seed'].unique()

    if len(seed_options) < 2:
        print('ERROR: Insufficient seeds!')

        return False
    data_options = df_remaining['replace'].unique()

    if len(data_options) == 1:
        print('ERROR: Insufficient data!')

        return False

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

    deltas = np.zeros(shape=num_deltas)

    for i in range(num_deltas):
        replace_index = w.iloc[i]['replace']
        seed = w.iloc[i]['seed']

        exp = results_utils.ExperimentIdentifier(dataset, model, replace_index, seed, diffinit, data_privacy)
        if exp.exists():
            w_weights = exp.load_weights(iter_range=iter_range, params=params, verbose=False).values[:, 1:]
            # the first column is the time-step
        else:
            print('WARNING: Missing data for (seed, replace) = (', seed, replace_index, ')')
            w_weights = np.array([np.nan])
        replace_index_p = wp.iloc[i]['replace']
        seed_p = wp.iloc[i]['seed']
    
        exp_p = results_utils.ExperimentIdentifier(dataset, model, replace_index_p, seed_p, diffinit, data_privacy)
        if exp_p.exists():
            wp_weights = exp_p.load_weights(iter_range=iter_range, params=params, verbose=False).values[:, 1:]
        else:
            print('WARNING: Missing data for (seed, replace) = (', seed_p, replace_index_p, ')')
            wp_weights = np.array([np.nan])

        if multivariate:
            delta = np.abs(w_weights - wp_weights)
        else:
            delta = np.linalg.norm(w_weights - wp_weights)
        deltas[i] = delta
    w_identifiers = list(zip(w['replace'], w['seed']))
    wp_identifiers = list(zip(wp['replace'], wp['seed']))
    identifiers = np.array(list(zip(w_identifiers, wp_identifiers)))

    return deltas, identifiers


def aggregated_loss(dataset, model, iter_range=(None, None), diffinit=False, data_privacy='all'):
    """ maybe i should include save/load here """
    path = 'fig_data/aggregated_loss.' + dataset + '.' + model + '.' + data_privacy + '.csv'
    try:
        df = pd.read_csv(path)
        df.set_index('t', inplace=True)
    except FileNotFoundError:
        print('Couldn\'t load from', path)

        df = results_utils.get_available_results(dataset, model)
        train_list = []
        vali_list = []

        for i, row in df.iterrows():
            experiment = results_utils.ExperimentIdentifier(dataset, model, replace_index=row['replace'],
                                                            seed=row['seed'], diffinit=diffinit,
                                                            data_privacy=data_privacy)
            loss = experiment.load_loss(iter_range=iter_range, verbose=False)
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
        df.to_csv(path, header=True, index=True)

    return df


def estimate_statistics_through_training(what, dataset, model, replace_index, seed, df=None,
                                         params=None, iter_range=(None, None), diffinit=True):
    """
    Grab a trace file for a model, estimate the alpha value for gradient noise throughout training
    NOTE: All weights taken together as IID (in the list of params supplied)

    TODO: save, amortise
    """
    assert what in ['gradients', 'weights']

    experiment = results_utils.ExperimentIdentifier(dataset, model, replace_index, seed, diffinit)
    if df is None:
        if what == 'gradients':
            df = experiment.load_gradients(noise=True, params=params, iter_range=iter_range)
        else:
            print('Getting posterior for weights, seed is irrelevant')
            df = results_utils.get_posterior_samples(dataset, model=model, replace_index=replace_index,
                                                     iter_range=iter_range, params=params, diffinit=diffinit)

        if df is False:
            print('ERROR: No data found')

            return False

    # now go through the iterations
    iterations = df['t'].unique()
    # store the results in this dataframe
    df_fits = pd.DataFrame(index=iterations)
    df_fits['N'] = np.nan
    df_fits['alpha'] = np.nan
    df_fits['alpha_fit'] = np.nan

    for t in iterations:
        df_t = df.loc[df['t'] == t, :]
        # columns are all gradient noise
        X = df_t.iloc[:, 2:].values.reshape(-1, 1)
        N = X.shape[0]
        df_fits['N'] = N
        # fit alpha_stable
        alpha, fit = stats_utils.fit_alpha_stable(X)
        df_fits.loc[t, 'alpha'] = alpha
        df_fits.loc[t, 'alpha_fit'] = fit
        # fit gaussian
        mu, sigma, W, p = stats_utils.fit_normal(X)
        df_fits.loc[t, 'norm_mu'] = mu
        df_fits.loc[t, 'norm_sigma'] = sigma
        df_fits.loc[t, 'norm_W'] = W
        df_fits.loc[t, 'norm_p'] = p
        # fit laplace
        loc, scale, D, p = stats_utils.fit_laplace(X)
        df_fits.loc[t, 'lap_loc'] = loc
        df_fits.loc[t, 'lap_scale'] = scale
        df_fits.loc[t, 'lap_D'] = D
        df_fits.loc[t, 'lap_p'] = p

    return df_fits


def get_sens_and_var_distribution(dataset, model, t, n_pairs=None, multivariate=False, diffinit=False):
    """
    TODO: amortise, save
    """
    df = results_utils.get_available_results(dataset, model)
    replace_counts = df['replace'].value_counts()
    replaces = replace_counts[replace_counts > 10].index.values
    print('Found', len(replaces), 'datasets with at least 10 seeds')
    # for ecah pair of drops...
    n_replaces = len(replaces)
    sens_array = []
    var_array = []
    overlap_array = []
    pairs_array = []

    for i, di in enumerate(replaces):
        for j in range(i + 1, n_replaces):
            dj = replaces[j]
            pairs_array.append((di, dj))

    if n_pairs is not None:
        total_pairs = len(pairs_array)
        print(total_pairs)
        pair_picks = np.random.choice(total_pairs, n_pairs, replace=False)
        pairs_array = [pairs_array[i] for i in pair_picks]
    print('Computing "local" epsilon for', len(pairs_array), 'pairs of datasets!')

    for di, dj in pairs_array:
        pair_sensitivity, pair_variability, n_seeds = compute_pairwise_sens_and_var(dataset, model, t,
                                                                                    replace_indices=[di, dj],
                                                                                    multivariate=multivariate,
                                                                                    verbose=False,
                                                                                    diffinit=diffinit)
        sens_array.append(pair_sensitivity)
        var_array.append(pair_variability)
        overlap_array.append(n_seeds)
    df = pd.DataFrame({'pair': pairs_array,
                       'sensitivity': sens_array,
                       'variability': var_array,
                       'overlapping_seeds': overlap_array})

    return df


def compute_pairwise_sens_and_var(dataset, model, t, replace_indices,
                                  multivariate=False, verbose=True, diffinit=False):
    """
    for a pair of experiments...
    estimate sensitivity (distance between means)
    estimate variability (variance about means .. both?)
    given delta
    return this epsilon!
    optionally, by parameter (returns an array!)
    """

    if multivariate:
        raise NotImplementedError
    samples_1 = results_utils.get_posterior_samples(dataset, (t, t+1), model,
                                                    replace_index=replace_indices[0],
                                                    params=None, seeds='all',
                                                    verbose=verbose, diffinit=diffinit)
    samples_2 = results_utils.get_posterior_samples(dataset, (t, t+1), model,
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
    n_seeds = len(intersection)

    if len(intersection) < 30:
        print('WARNING: Experiments with replace indices', replace_indices, 'only have', n_seeds, 'overlapping seeds:', intersection)

        return np.nan, np.nan, n_seeds
    samples_1_intersection = samples_1.loc[intersection, :]
    samples_2_intersection = samples_2.loc[intersection, :]
    # compute the distances on the same seed
    distances = np.linalg.norm(samples_1_intersection - samples_2_intersection, axis=1)
    sensitivity = np.max(distances)

    if verbose:
        print('Max sensitivity from same seed diff data:', sensitivity)
    #### compute distance by getting average value and comparing
    mean_1 = samples_1.mean(axis=0)
    mean_2 = samples_2.mean(axis=0)
    sensitivity_bymean = np.linalg.norm(mean_1 - mean_2)

    if verbose:
        print('Sensitivity from averaging posteriors and comparing:', sensitivity_bymean)
    variability_1 = (samples_1 - mean_1).values.std()
    variability_2 = (samples_2 - mean_2).values.std()
    # NOT SURE ABOUT THIS
    variability = 0.5*(variability_1 + variability_2)

    if verbose:
        print('Variability:', variability)

    return sensitivity, variability, n_seeds


def estimate_variability(dataset, model, t, multivariate, diffinit=False,
                         data_privacy='all', n_replaces='max', n_seeds='max',
                         ephemeral=False, verbose=True):
    """
    As for estimating the sensitivity, we want to grab a bunch of posteriors and estimate the variability
    """

    if ephemeral:
        # this is a hack sorry
        data_path = './fig_data/a_file_that_doesnt_exist.npy'
        verbose=False
    else:
        data_path = './fig_data/sigmas.' + dataset + '.' + data_privacy + '.' + model + '.t' + str(t) + '.DIFFINIT'*diffinit + 'MULTIVAR'*multivariate + '.npy'
    try:
        data = np.load(data_path).item()
        sigmas = data['sigmas']
        replaces = data['replaces']

        if verbose:
            print('Loaded sigmas from file', data_path)
    except FileNotFoundError:
        print('[estimate_variability] Failed to load', data_path)
        df = results_utils.get_available_results(dataset, model, data_privacy=data_privacy)
        replace_counts = df['replace'].value_counts()
        replaces = replace_counts[replace_counts > 2].index.values

        if verbose:
            print(f'Estimating variability across {len(replaces)} datasets!')
            print('Warning: this can be slow...')
        sigmas = []

        if ephemeral:
            replaces = np.random.choice(replaces, n_replaces, replace=False)

        for replace_index in replaces:
            if verbose:
                print('replace index:', replace_index)
            samples = results_utils.get_posterior_samples(dataset, (t, t+1), model,
                                                          replace_index=replace_index,
                                                          params=None, seeds='all',
                                                          verbose=verbose,
                                                          diffinit=diffinit,
                                                          data_privacy=data_privacy,
                                                          n_seeds=n_seeds)
            try:
                params = samples.columns[2:]

                if multivariate:
                    this_sigma = samples.std(axis=0)
                    this_sigma = this_sigma[params]
                else:
                    params_vals = samples[params].values
                    params_norm = params_vals - params_vals.mean(axis=0)
                    params_flat = params_norm.flatten()
                    this_sigma = np.std(params_flat)
            except AttributeError:
                print('WARNING: data from', replace_index, 'is bad - skipping')
                assert samples is False
                this_sigma = np.nan
            sigmas.append(this_sigma)
        sigmas = np.array(sigmas)
        data = {'sigmas': sigmas, 'replaces': replaces}

        if not ephemeral:
            np.save(data_path, data)

    if n_replaces == 'max':
        sigmas = sigmas
    else:
        assert type(n_replaces) == int

        if n_replaces > len(sigmas):
            print(f'WARNING: Can\'t select {n_replaces} sigmas, falling back to max ({len(sigmas)})')
            sigmas = sigmas
        else:
            if verbose:
                print(f'Sampling {n_replaces} random sigmas')
            sigmas = np.random.choice(sigmas, n_replaces, replace=False)

    if verbose:
        print('Estimated variability using', len(sigmas[~np.isnan(sigmas)]), 'replaces')
    estimated_variability = np.nanmin(sigmas, axis=0)

    return estimated_variability


def validate_sigmas_sens_var(dataset, model, t, n_pairs, diffinit):
    """
    when we compute variability using sens_var_dist, it should be the same as the average of the sigmas from sigmas
    this is just validating that the amortised data is consistent!
    """
    sens_and_var_path = './fig_data/sens_var_dist.' + dataset + '.' + model + '.t' + str(t) + '.np' + str(n_pairs) + '.DIFFINIT'*diffinit + '.csv'
    try:
        sens_and_var_df = pd.read_csv(sens_and_var_path)
    except FileNotFoundError:
        print('ERROR: Couldn\'t find', sens_and_var_path)

        return False
    sigmas_path = './fig_data/sigmas.' + dataset + '.' + model + '.t' + str(t) + '.DIFFINIT'*diffinit + '.npy'
    try:
        sigmas = np.load(sigmas_path).item()
    except FileNotFoundError:
        print('ERROR: Couldn\'t find', sigmas_path)

        return False
    # now go through the pairs from sens_and_var
    # make a df out of sigmas
    sigmas_df = pd.DataFrame(sigmas)
    sigmas_df.set_index('replaces',  inplace=True)
    bad_pairs = set()

    for i, row in sens_and_var_df.iterrows():
        if i % 100 == 0:
            print(i)
        pair = row['pair']
        from_sens_var = row['variability']
        pair1 = pair.split(',')[0][2:-1]
        pair2 = pair.split(',')[1][2:-2]
        from_sigmas = 0.5*(sigmas_df.loc[pair1].sigmas + sigmas_df.loc[pair2].sigmas)
        try:
            assert np.abs(from_sens_var - from_sigmas) < 1e-5
        except AssertionError:
            print('Found bad pair', pair)
            print(from_sens_var, from_sigmas)
            ipdb.set_trace()
            bad_pairs.add(pair)
    print('Found', len(bad_pairs), 'bad pairs! That\'s', np.round(100*len(bad_pairs)/sens_and_var_df.shape[0], 2), '%')

    return bad_pairs


def find_convergence_point_for_single_experiment(dataset, model, replace_index,
                                                 seed, diffinit=False, tolerance=3,
                                                 metric='ce', verbose=False,
                                                 data_privacy='all'):
    # load the trace
    experiment = results_utils.ExperimentIdentifier(dataset, model, replace_index,
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
    # was there a decrease at that time point? (1 if yes --> good)
    decrease = (delta_vali < 0)
    counter = 0

    for t, dec in decrease.items():
        if not dec:
            counter += 1
        else:
            counter = 0

        if counter >= tolerance:
            convergence_point = t

            break
    else:
        if verbose:
            print('Did not find instance of validation loss failing to decrease for', tolerance, 'steps - returning nan')
        convergence_point = np.nan

    return convergence_point


def find_convergence_point(dataset, model, diffinit, tolerance, metric, data_privacy='all'):
    """ wrapper for the whole experiment """
    results = results_utils.get_available_results(dataset, model, diffinit=diffinit, data_privacy=data_privacy)
    n_results = results.shape[0]
    points = np.zeros(n_results)

    for index, row in results.iterrows():
        replace_index = row['replace']
        seed = row['seed']
        try:
            point = find_convergence_point_for_single_experiment(dataset, model, replace_index,
                                                                 seed, diffinit=diffinit,
                                                                 tolerance=tolerance,
                                                                 metric=metric,
                                                                 data_privacy=data_privacy)
        except:
            point = np.nan
        points[index] = point
    print('For dataset', dataset, 'and model', model, 'with diffinit', diffinit, 'we have:')
    print('STDEV:', np.nanstd(points))
    print('MEDIAN:', np.nanmedian(points))
    print('MEAN:', np.nanmean(points))
    print('FRACTION INVALID:', np.mean(np.isnan(points)))
    convergence_point = np.nanmedian(points)
    valid_frac = np.mean(np.isfinite(points))
    print('Selecting median as convergence point:', convergence_point)

    return convergence_point, valid_frac
