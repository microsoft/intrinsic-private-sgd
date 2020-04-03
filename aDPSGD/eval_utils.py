#!/usr/bin/env ipython

import numpy as np
import pandas as pd
import ipdb
import paths
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns
from scipy.stats import kstest, norm, laplace, shapiro, anderson

import vis_utils
import data_utils
import model_utils
import results_utils


def get_experiment_details(dataset, model, verbose=False, data_privacy='all'):
    if dataset == 'housing_binary':
        task = 'binary'
        batch_size = 28
        lr = 0.1
        if model in ['linear', 'logistic']:
            n_weights = 14
        elif model == 'mlp':
            n_weights = 121
        N = 364
    elif 'mnist_binary' in dataset:
        task = 'binary'
        batch_size = 32
        if 'buggy' in dataset:
            lr = 0.1
        else:
            lr = 0.5
        if model == 'logistic':
            if 'cropped' in dataset:
                n_weights = 101
            else:
                n_weights = 51
        elif model == 'mlp':
            n_weights = 521
        N = 10397
    elif 'cifar10_binary' in dataset:
        task = 'binary'
        batch_size = 32
        lr = 0.5
        if model == 'logistic':
            n_weights = 51
        N = 9000
    elif dataset == 'protein':
        task = 'binary'
        batch_size = 50
        lr = 0.01
        assert model == 'logistic'
        n_weights = 75
        N = 65589
    elif 'forest' in dataset:
        task = 'binary'
        batch_size = 50
        lr = 1.0
        assert model == 'logistic'
        n_weights = 50
        N = 378783
    elif 'adult' in dataset:
        task = 'binary'
        batch_size = 32
        lr = 0.5            # possibly check this
        if 'pca' in dataset:
            if model == 'logistic':
                n_weights = 51
            else:
                raise ValueError(model)
        else:
            if model == 'logistic':
                n_weights = 101
            else:
                n_weights = 817
        N = 29305
    elif dataset in ['mnist', 'mnist_square']:
        task = 'classification'
        batch_size = 32
        lr = 0.1
        if model == 'mlp':
            n_weights = 986
        elif model == 'cnn':
            n_weights = 1448
        N = 54000
    elif dataset == 'cifar10':
        task = 'classification'
        batch_size = 32
        lr = 0.01
        if model == 'mlp':
            n_weights = 7818
        elif model == 'cnn':
            raise NotImplementedError
        N = 45000
    else:
        raise ValueError(dataset)
    if not data_privacy == 'all':
        N = N/2
    if verbose:
        print('Experiment details:')
        print('\tDataset:', dataset)
        print('\tModel:', model)
        print('\tTask:', task)
        print('\tbatch size:', batch_size)
        print('\tlr:', lr)
        print('\tn_weights:', n_weights)
        print('\tN:', N)
    return task, batch_size, lr, n_weights, N

def get_target_noise_for_model(dataset, model, t, epsilon, delta, sensitivity, verbose):
    target_noise = compute_gaussian_noise(epsilon, delta, sensitivity)
    if verbose: 
        print('[test] Target noise:', target_noise)
   
    # without different initiaisation
    intrinsic_noise = estimate_variability(dataset, model, t, by_parameter=False, diffinit=False)
    if intrinsic_noise < target_noise:
        noise_to_add = compute_additional_noise(target_noise, intrinsic_noise)
    else:
        noise_to_add = 0
    print('[augment_sgd] \nintrinsic noise:', intrinsic_noise, '\nnoise to add:', noise_to_add)
    if np.abs(noise_to_add) < 1e-5:
        print('[augment_sgd] Hurray! Essentially no noise required!')
  
    # noise using different initialisation
    intrinsic_noise_diffinit = estimate_variability(dataset, model, t, by_parameter=False, diffinit=True)
    if intrinsic_noise_diffinit < target_noise:
        noise_to_add_diffinit = compute_additional_noise(target_noise, intrinsic_noise_diffinit)
    else:
        noise_to_add_diffinit = 0
    print('[augment_sgd_diffinit] \nintrinsic noise:', intrinsic_noise_diffinit, '\nnoise to add:', noise_to_add_diffinit)
    if np.abs(noise_to_add_diffinit) < 1e-5:
        print('[augment_sgd] Hurray! Essentially no noise required!')

    if noise_to_add_diffinit > noise_to_add:
        print('WARNING: Noise from diffinit is... lower than without it?')
        ipdb.set_trace()

    return target_noise, noise_to_add, noise_to_add_diffinit

def debug_just_test(dataset, model, replace_index, seed, t, diffinit=False, use_vali=False):
    task, batch_size, lr, n_weights, N = get_experiment_details(dataset, model)
    _, _, x_vali, y_vali, x_test, y_test = data_utils.load_data(data_type=dataset, replace_index=replace_index)
    
    weights_path = results_utils.trace_path_stub(dataset, model, replace_index, seed, diffinit=diffinit) + '.weights.csv'
    
    metrics = None
    metric_names = None

    # DEBUG
    model_object = model_utils.build_model(model_type=model, data_type=dataset, init_path=weights_path, t=t)
    model_utils.prep_for_training(model_object, seed=0, lr=0, task_type=task)                                      # literally so i can evaluate it later

    if use_vali:
        #     print('Evaluating on validation set')
        metrics = model_object.compute_metrics(x_vali, y_vali)
    else:
        metrics = model_object.compute_metrics(x_test, y_test)

    metric_names = model_object.metric_names
    results = dict(zip(metric_names, metrics))
    # not sure if there is a memory leak or whatever
    del model_object
    del x_vali
    del y_vali
    del x_test
    del y_test
    model_utils.K.backend.clear_session()
    return results

def test_model_with_noise(dataset, model, replace_index, seed, t, epsilon=None, delta=None, sensitivity_from_bound=True, metric_to_report='binary_accuracy', verbose=False, num_deltas=1000, diffinit=False, data_privacy='all'):
    """
    test the model on the test set of the respective dataset
    """
    task, batch_size, lr, _, N = get_experiment_details(dataset, model, data_privacy)
    # load the test set
    _, _, _, _, x_test, y_test = data_utils.load_data(data_type=dataset, replace_index=replace_index)      # the drop index doesnt' actually matter for test set
  
    # we always add noise!
    if epsilon is None:
        epsilon = 1.0
    if delta is None:
        delta = 1.0/N
    if verbose:
        print('Adding noise for epsilon, delta = ', epsilon, delta)

    if sensitivity_from_bound:
        if model == 'logistic':
            lipschitz_constant = np.sqrt(2)
        else:
            raise ValueError(model)
            # optionally estimate empirical lipschitz ...?
        sensitivity = compute_wu_bound(lipschitz_constant, t=t, N=N, batch_size=batch_size, eta=lr)
    else:
        # compute sensitivity empirically!
        # diffinit set to False beacuse it doesn't make a differnce
        sensitivity = estimate_sensitivity_empirically(dataset, model, t, num_deltas=num_deltas, diffinit=False, data_privacy=data_privacy)
    if sensitivity is False:
        print('ERROR: Empirical sensitivity not available.')
        return False
    print('Sensitivity:', sensitivity)

    target_noise, noise_to_add, noise_to_add_diffinit = get_target_noise_for_model(dataset, model, t, epsilon, delta, sensitivity, verbose)    
    
    weights_path = results_utils.trace_path_stub(dataset, model, replace_index, seed, diffinit=diffinit) + '.weights.csv'
    print('Evaluating model from', weights_path)
    
    model_object = model_utils.build_model(model_type=model, data_type=dataset, init_path=weights_path, t=t)
    model_utils.prep_for_training(model_object, seed=0, lr=0, task_type=task)                                      # literally so i can evaluate it later

    metrics = model_object.compute_metrics(x_test, y_test)
    metric_names = model_object.metric_names
    if verbose: print('PERFORMANCE (no noise):')
    for (n, v) in zip(metric_names, metrics):
        if verbose: print(n, v)
        if n == metric_to_report:
            noiseless_performance = v

    noise_options = {'bolton': target_noise,
            'augment_sgd': noise_to_add,
            'augment_sgd_diffinit': noise_to_add_diffinit}
    noise_performance = {'bolton': np.nan,
            'augment_sgd': np.nan,
            'augment_sgd_diffinit': np.nan}

    n_weights = len(model_object.get_weights(flat=True))
    # generate standard gaussian noise
    standard_noise = np.random.normal(size=n_weights, loc=0, scale=1)

    for setting in noise_options:
        model_object = model_utils.build_model(model_type=model, data_type=dataset, init_path=weights_path, t=t)
        model_utils.prep_for_training(model_object, seed=0, lr=0, task_type=task)                                      # literally so i can evaluate it later
        weights = model_object.get_weights(flat=True)
        noise = noise_options[setting]
        #noisy_weights = model_utils.add_gaussian_noise(weights, noise)
        noisy_weights = weights + standard_noise * noise
        unflattened_noisy_weights = model_object.unflatten_weights(noisy_weights)
        model_object.set_weights(unflattened_noisy_weights)
    
        metrics = model_object.compute_metrics(x_test, y_test)
        if verbose: print('PERFORMANCE (' + setting + '):')
        for (n, v) in zip(metric_names, metrics):
            if verbose: print(n, v)
            if n == metric_to_report:
                noise_performance[setting] = v
                break
        del model_object

    # extract the performances
    bolton_performance = noise_performance['bolton']
    augment_performance = noise_performance['augment_sgd']
    augment_performance_diffinit = noise_performance['augment_sgd_diffinit']
    # tidy up so we dont get a horrible memory situation 
    model_utils.K.backend.clear_session()
    return noiseless_performance, bolton_performance, augment_performance, augment_performance_diffinit

def compute_gaussian_noise(epsilon, delta, sensitivity, verbose=True):
    """
    using gaussian mechanism assumption
    """
    c = np.sqrt(2 * np.log(1.25 / delta))
    if verbose:
        print('[compute_gaussian_noise] C is', c)
        print('[compute_gaussian_noise] Ratio of sens and eps is', sensitivity/epsilon)
    sigma = c * (sensitivity/epsilon)
    return sigma

def compute_additional_noise(target_noise, intrinsic_noise):
    """
    assuming we want to get to target_noise STDDEV, from intrinsic noise,
    independent gaussians
    """
    additional_noise = np.sqrt(target_noise**2 - intrinsic_noise**2)
    return additional_noise

def estimate_empirical_lipschitz(dataset, model, diffinit, iter_range, n_samples=5000):
    """
    get the biggest gradient during training

    NOTE: using 10k samples and all time-points, 
    - on housing+linear we get max_L ~ 2.49, ave_L ~ 0.33, min_L ~ 0.06
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
        gradients = results_utils.load_gradients(dataset, model, replace_index=replace, seed=seed, iter_range=iter_range, diffinit=diffinit)
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

def find_different_datasets(dataset, model, num_deltas, t):
    """ Using data computed in the delta histogram """
    path_string = './fig_data/delta_histogram.' + str(dataset) + '.' + str(model) + '.nd_' + str(num_deltas) + '.t_' + str(t) + '.npy'
    try:
        plot_data = np.load(path_string).item()
    except FileNotFoundError:
        print('[find_different_datasets] ERROR: Run delta_histogram for this setting first')
        return False
    vary_data_deltas = plot_data['vary_S']
    vary_data_identifiers = plot_data['S_identifiers']
    # just get the top 10 biggest
    biggest_idx = np.argsort(-vary_data_deltas)[:10]
    biggest_deltas = vary_data_deltas[biggest_idx]
    biggest_identifiers = vary_data_identifiers[biggest_idx]
    return True

def estimate_sensitivity_empirically(dataset, model, t, num_deltas, diffinit=False, data_privacy='all'):
    """ pull up the histogram
    """
    path_string = './fig_data/delta_histogram.' + str(dataset) + '.' + data_privacy + '.' + str(model) + '.nd_' + str(num_deltas) + '.t_' + str(t) + '.DIFFINIT'*diffinit + '.npy'
    try:
        plot_data = np.load(path_string).item()
    except FileNotFoundError:
        print('[estimate_sensitivty_empirically] ERROR: Run delta_histogram for this setting first:', path_string)
        return False
    vary_data_deltas = plot_data['vary_S']
    sensitivity = np.nanmax(vary_data_deltas)
    return sensitivity

def get_deltas(dataset, iter_range, model, 
        vary_seed=True, vary_data=True, params=None, num_deltas=100,
        include_identifiers=False, diffinit=False, data_privacy='all'):
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
    remaining_rows = [x for x in range(df.shape[0]) if not x in w_rows]
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
        if results_utils.check_if_experiment_exists(dataset, model, replace_index, seed, diffinit, data_privacy=data_privacy):
            w_weights = results_utils.load_weights(dataset, model, replace_index=replace_index, seed=seed, iter_range=iter_range, params=params, verbose=False, diffinit=diffinit, data_privacy=data_privacy).values[:, 1:] # the first column is the time-step
        else:
            print('WARNING: Missing data for (seed, replace) = (', seed, replace_index, ')')
            w_weights = np.array([np.nan])
        replace_index_p = wp.iloc[i]['replace']
        seed_p = wp.iloc[i]['seed']
        if results_utils.check_if_experiment_exists(dataset, model, replace_index_p, seed_p, diffinit, data_privacy=data_privacy):
            wp_weights = results_utils.load_weights(dataset, model, replace_index=replace_index_p, seed=seed_p, iter_range=iter_range, params=params, verbose=False, diffinit=diffinit, data_privacy=data_privacy).values[:, 1:] # the first column is the time-step
        else:
            print('WARNING: Missing data for (seed, replace) = (', seed_p, replace_index_p, ')')
            wp_weights = np.array([np.nan])
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
            loss = results_utils.load_loss(dataset, model, replace_index=row['replace'],
                    seed=row['seed'], iter_range=iter_range, diffinit=diffinit, verbose=False, data_privacy=data_privacy)
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

def estimate_statistics_through_training(what, dataset, identifier, df=None, params=None, iter_range=(None, None)):
    """
    Grab a trace file for a model, estimate the alpha value for gradient noise throughout training
    NOTE: All weights taken together as IID (in the list of params supplied)
    """
    assert what in ['gradients', 'weights']
    if df is None:
        assert not identifier is None
        # get from the all_gradients file
        model, replace_index, seed = identifier
        if what == 'gradients':
            df = results_utils.load_gradients(dataset, model, replace_index, seed, noise=True, params=params, iter_range=iter_range)
        else:
            print('Getting posterior for weights, seed is irrelevant')
            df = results_utils.get_posterior_samples(dataset, model=model, replace_index=replace_index, iter_range=iter_range, params=params, diffinit=diffinit)
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
        alpha, fit = fit_alpha_stable(X)
        df_fits.loc[t, 'alpha'] = alpha
        df_fits.loc[t, 'alpha_fit'] = fit
        # fit gaussian
        mu, sigma, W, p = fit_normal(X)
        df_fits.loc[t, 'norm_mu'] = mu
        df_fits.loc[t, 'norm_sigma'] = sigma
        df_fits.loc[t, 'norm_W'] = W
        df_fits.loc[t, 'norm_p'] = p
        # fit laplace
        loc, scale, D, p= fit_laplace(X)
        df_fits.loc[t, 'lap_loc'] = loc
        df_fits.loc[t, 'lap_scale'] = scale
        df_fits.loc[t, 'lap_D'] = D 
        df_fits.loc[t, 'lap_p'] = p 
        # logistic
        #mean, s, D, p = fit_logistic(X)
        #df_fits.loc[t, 'logo_m'] = mean
        #df_fits.loc[t, 'log_s'] = s
        #df_fits.loc[t, 'log_D'] = D
        #df_fits.loc[t, 'log_p'] = p
    return df_fits

def fit_alpha_stable(X):
    """
    """
    N = X.shape[0]
    # copied from umut
    for i in range(1, 1 + int(np.sqrt(N))):
        if N % i == 0:
            m = i
    alpha = alpha_estimator(m, X)
    # dont' know how to estimate goodness of fit for this distribution yet
    goodness_of_fit = np.nan
    return alpha, goodness_of_fit

def alpha_estimator(m, X):
    """
    this is taken from 
    https://github.com/umutsimsekli/sgd_tail_index/blob/master/utils.py
    and modified to remove torchiness
    # Corollary 2.4 in Mohammadi 2014

    X: gradient noise (grad - minibatch grad)
    m: K1 I think (n is K2)
    """
    # X is N by d matrix
    N = len(X)           # number of gradients, basically
    n = int(N/m) # must be an integer: this is K2 in the theorem
    Y = np.sum(X.reshape(n, m, -1), axis=1)      # produce Y by first reshaping X to be n x m (x the rest), summing over m'th dimension
    eps = np.spacing(1)
    Y_log_norm = (np.log(np.linalg.norm(Y, axis=1) + eps)).mean()
    X_log_norm = (np.log(np.linalg.norm(X, axis=1) + eps)).mean()
    diff = (Y_log_norm - X_log_norm) / np.log(m)
    return 1.0 / diff

def fit_normal(X):
    if X.shape[0] > 5000:
        # the p-value estimate in shapiro is not accurate for N > 5000 for some reason
        idx = np.random.choice(X.shape[0], 4999, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X
    mean = np.mean(X_sub)
    std = np.std(X_sub)
    # shapiro-wilk test against gaussian
    Dval_gauss, pval_gauss = shapiro((X_sub - mean)/std)
    return mean, std, Dval_gauss, pval_gauss

def fit_laplace(X):
    loc = np.median(X)
    scale = np.mean(np.abs(X) - loc)
    # I think the kstest isn't very good for testing laplace fit, the p-value has a very high variance even when I run the test on
    # 1000000 iid laplace RVs
    # need to find a better test
    Dval_lap, pval_lap = kstest(X, laplace(loc=loc, scale=scale).cdf)
    return loc, scale, Dval_lap, pval_lap

def fit_logistic(X):
    if X.shape[0] > 5000:
        # the p-value estimate in shapiro is not accurate for N > 5000 for some reason
        idx = np.random.choice(X.shape[0], 4999, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X
    mean = np.mean(X_sub)
    s = np.sqrt(3)*np.std(X_sub)/np.pi
    Dval_log, critical_values, significance_level = anderson(X_sub.reshape(-1), dist='logistic')
    pval_log = np.nan
    return mean, s, Dval_log, pval_log

def test_alpha_estimator(N=100, d=1):
    """
    Estimate ~sensitivity and specificity of the estimator
    """
    for i in range(1, 1+int(np.sqrt(N))):
        if N % i == 0:
            m = i
    print(m)
    # generate gaussian data (alpha = 2)
    X_norm = np.random.normal(size=(N, d))
    alpha_norm = alpha_estimator(m, X_norm)
    # future: generate arbitrary alpha-stable RVs, see here: https://en.wikipedia.org/wiki/Stable_distribution#Simulation_of_stable_variables
    # generate beta distribution (NOT a stable distribution)
    beta_a = np.abs(np.random.normal())
    beta_b = np.abs(np.random.normal())
    print('beta: a:', beta_a, 'b:', beta_b)
    X_beta = np.random.beta(a=beta_a, b=beta_b, size=(N, d))
    alpha_beta = alpha_estimator(m, X_beta)

    print('norm:', alpha_norm)
    print('beta:', alpha_beta)

def compute_wu_bound_strong(lipschitz_constant, gamma, n_samples, batch_size, verbose=True):
    """
    compute the bound in the strongly convex case
    basically copied from 
	https://github.com/tensorflow/privacy/blob/1ce8cd4032b06e8afa475747a105cfcb01c52ebe/tensorflow_privacy/privacy/bolt_on/optimizers.py
    """
    # note that for the strongly convex setting, the learning rate at every point is the minimum of (1/beta, 1/(eta *t))
    # this sin't really important here, it's just good to remember that if this wasn't the case, this bound doesn't hold! (that we know of)
    l2_sensitivity = (2 * lipschitz_constant) / \
            (gamma * n_samples * batch_size)
    if verbose: print('[eval_utils] Bound on L2 sensitivity:', l2_sensitivity)
    return l2_sensitivity

def compute_wu_bound(lipschitz_constant, t, N, batch_size, eta, verbose=True):
    # k is the number of time you went through the data
    batches_per_epoch = N // batch_size
    # t is the number of batches
    n_epochs = t / batches_per_epoch
    if n_epochs < 1:
        print('WARNING: <1 pass competed')
        # TODO: make sure we can treat k like this
    l2_sensitivity = 2 * n_epochs * lipschitz_constant * eta / batch_size
    #l2_sensitivity = 2 * n_epochs * lipschitz_constant * eta 
    if verbose: print('[eval_utils] Bound on L2 sensitivity:', l2_sensitivity)
    return l2_sensitivity
 
def get_sens_and_var_distribution(dataset, model, t, n_pairs=None, by_parameter=False, diffinit=False):
    """
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
    if not n_pairs is None:
        total_pairs = len(pairs_array)
        print(total_pairs)
        pair_picks = np.random.choice(total_pairs, n_pairs, replace=False)
        pairs_array = [pairs_array[i] for i in pair_picks]
    print('Computing "local" epsilon for', len(pairs_array), 'pairs of datasets!')
    for di, dj in pairs_array: 
        pair_sensitivity, pair_variability, n_seeds = compute_pairwise_sens_and_var(dataset, model, t, replace_indices=[di, dj], by_parameter=by_parameter, verbose=False, diffinit=diffinit)
        sens_array.append(pair_sensitivity)
        var_array.append(pair_variability)
        overlap_array.append(n_seeds)
    df = pd.DataFrame({'pair': pairs_array, 'sensitivity': sens_array, 'variability': var_array, 'overlapping_seeds': overlap_array})
    return df

def compute_pairwise_sens_and_var(dataset, model, t, replace_indices, by_parameter=False, verbose=True, diffinit=False):
    """
    for a pair of experiments...
    estimate sensitivity (distance between means)
    estimate variability (variance about means .. both?)
    given delta
    return this epsilon!
    optionally, by parameter (returns an array!)
    """
    if by_parameter:
        raise NotImplementedError
    samples_1 = results_utils.get_posterior_samples(dataset, (t, t+1), model, replace_index=replace_indices[0], params=None, seeds='all', verbose=False, diffinit=diffinit)
    samples_2 = results_utils.get_posterior_samples(dataset, (t, t+1), model, replace_index=replace_indices[1], params=None, seeds='all', verbose=False, diffinit=diffinit)
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
    ### compute the distances on the same seed
    distances = np.linalg.norm(samples_1_intersection - samples_2_intersection, axis=1)
    sensitivity = np.max(distances)
    if verbose: print('Max sensitivity from same seed diff data:', sensitivity)
    #### compute distance by getting average value and comparing
    mean_1 = samples_1.mean(axis=0)
    mean_2 = samples_2.mean(axis=0)
    sensitivity_bymean = np.linalg.norm(mean_1 - mean_2)
    if verbose: print('Sensitivity from averaging posteriors and comparing:', sensitivity_bymean)
    variability_1 = (samples_1 - mean_1).values.std()
    variability_2 = (samples_2 - mean_2).values.std()
    # NOT SURE ABOUT THIS
    variability = 0.5*(variability_1 + variability_2)
    if verbose: print('Variability:', variability)
    return sensitivity, variability, n_seeds

def estimate_variability(dataset, model, t, by_parameter, replaces=None, diffinit=False, data_privacy='all'):
    """
    As for estimating the sensitivity, we want to grab a bunch of posteriors and estimate the variability
    """
    data_path = './fig_data/sigmas.' + dataset + '.' + data_privacy + '.' + model + '.t' + str(t) + '_byparameter'*by_parameter + '.DIFFINIT'*diffinit + '.npy'
    try:
        data = np.load(data_path).item()
        sigmas = data['sigmas']
        replaces = data['replaces']
        print('Loaded sigmas from file', data_path)
    except FileNotFoundError:
        print('[estimate_variability] Failed to load', data_path)
        if replaces is None:
            df = results_utils.get_available_results(dataset, model, data_privacy=data_privacy)
            replace_counts = df['replace'].value_counts()
            replaces = replace_counts[replace_counts > 2].index.values
        else:
            assert type(replaces) == list
        n_replaces = len(replaces)
        print('Estimating variability across', n_replaces, 'datasets!')
        print('Warning: this can be slow...')
        sigmas = []
        for replace_index in replaces:
            samples = results_utils.get_posterior_samples(dataset, (t, t+1), model, replace_index=replace_index, params=None, seeds='all', verbose=False, diffinit=diffinit, data_privacy=data_privacy)
            try:
                params = samples.columns[2:]
                if by_parameter:
                    this_sigma = samples.std(axis=0)
                    this_sigma = this_sigma[params]
                else:
                    params_vals = samples[params].values
                    params_norm = params_vals - params_vals.mean(axis=0)
                  #  params_norm = params_vals
                    params_flat = params_norm.flatten()
                    this_sigma = np.std(params_flat) ## just one sigma really
            except AttributeError:
                print('WARNING: data from', replace_index, 'is bad - skipping')
                assert samples is False
                this_sigma = np.nan
            sigmas.append(this_sigma)
        sigmas = np.array(sigmas)
        data = {'sigmas': sigmas, 'replaces': replaces}
        np.save(data_path, data)
    estimated_variability = np.nanmedian(sigmas, axis=0)
    return estimated_variability

 ### VESTIGIAL ###

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
            assert np.abs(from_sens_var- from_sigmas) < 1e-5
        except AssertionError:
            print('Found bad pair', pair)
            print(from_sens_var, from_sigmas)
            ipdb.set_trace()
            bad_pairs.add(pair)
    print('Found', len(bad_pairs), 'bad pairs! That\'s', np.round(100*len(bad_pairs)/sens_and_var_df.shape[0], 2), '%')
    return bad_pairs


def find_convergence_point_for_single_experiment(dataset, model, replace_index, seed, diffinit=False, tolerance=3, 
        metric='ce', verbose=False, data_privacy='all'):
    """
    """
    # load the trace
    loss = results_utils.load_loss(dataset, model, replace_index, seed, iter_range=(None, None), diffinit=diffinit, data_privacy=data_privacy)
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
            point = find_convergence_point_for_single_experiment(dataset, model, replace_index, seed, diffinit=diffinit, tolerance=tolerance, metric=metric, data_privacy=data_privacy)
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

def delta_v_distance(dataset, model, num_deltas, t, predict=False):
    """
    """
    if dataset == 'forest':
        print('WARNING: No cosine distances for now, data too big for memory')
    path = './fig_data/delta_histogram.' + dataset + '.' + model + '.nd_' + str(num_deltas) + '.t_' + str(t) + '.npy'
    try:
        deltas = np.load('./fig_data/delta_histogram.' + dataset + '.' + model + '.nd_' + str(num_deltas) + '.t_' + str(t) + '.npy').item()
    except FileNotFoundError:
        print('ERROR: Run delta histogram first for this setting:', path)
    delta_data = deltas['vary_S']
    delta_pairs = deltas['S_identifiers']
    keep_idx = np.isfinite(delta_data)
    delta_data = delta_data[keep_idx]
    delta_pairs = delta_pairs[keep_idx]
    pairs = [(int(x[0, 0]), int(x[1, 0])) for x in delta_pairs]
    paired_distance_data = data_utils.compute_distance_for_pairs(dataset, pairs)

    if predict:
        reg = LinearRegression().fit(paired_distance_data, delta_data)
        r2 = reg.score(paired_distance_data, delta_data)
        print(r2)
        # now predict
        predictions = reg.predict(paired_distance_data)
        xvals = predictions
    else:
        xvals = paired_distance_data
   
    fig, axarr = plt.subplots(nrows=1, ncols=1)
    axarr.scatter(xvals, delta_data, s=4)
    if predict:
        axarr.set_xlabel('predicted |w(i) - w(j)|')
    else:
        axarr.set_xlabel('cosine distance')
        #axarr.set_xlabel('d(xi, xj)')
        #axarr.set_xlabel('|xi - xj|')
    if predict:
        axarr.plot(np.array(axarr.get_xlim()), np.array(axarr.get_xlim()), ls='--', color='red')
        axarr.text(0.8, 0.2, 'R2: ' + str(np.round(r2, 3)), transform=axarr.transAxes, color='red')
    axarr.set_ylabel('|w(i) - w(j)|')
    axarr.set_title('dataset: ' + dataset +', model: ' + model + ', t: ' + str(t))
    vis_utils.beautify_axes(np.array([axarr]))
    return True

def discretise_theoretical_sensitivity(dataset, model, theoretical_sensitivity):
    """
    stop treating k as a float, and turn it back into an integer!
    """
    _, batch_size, lr, _, N = get_experiment_details(dataset, model)
    if model == 'logistic':
        L = np.sqrt(2)
    else:
        raise ValueError(model)
    k = batch_size * theoretical_sensitivity/(2 * L * lr)
    # we take the ceiling, basically because this is an upper bound
    discrete_k = np.ceil(k)
    discretised_sensitivity = 2 * discrete_k * L * lr / batch_size
    return discretised_sensitivity
