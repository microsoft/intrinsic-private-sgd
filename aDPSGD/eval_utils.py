#!/usr/bin/env ipython

import numpy as np
import pandas as pd
import ipdb
import os
import seaborn as sns
from scipy.stats import kstest, norm, laplace, shapiro, anderson

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
