#!/usr/bin/env ipython
# The functions in this file relate to evaluating the performance of the model
# Specifically we are interested in the utility of models with different privacy levels

import numpy as np
import ipdb

import data_utils
import model_utils
from results_utils import ExperimentIdentifier
import derived_results
import experiment_metadata as em
from run_experiment import load_cfg

# --- to do with testing the model's performance --- #


def get_target_noise_for_model(cfg_name, model, t, epsilon, delta, sensitivity, verbose):
    target_noise = compute_gaussian_noise(epsilon, delta, sensitivity)

    if verbose:
        print('[test] Target noise:', target_noise)

    # without different initiaisation
    intrinsic_noise = derived_results.estimate_variability(cfg_name, model, t, multivariate=False, diffinit=False)

    if intrinsic_noise < target_noise:
        noise_to_add = compute_additional_noise(target_noise, intrinsic_noise)
    else:
        noise_to_add = 0
    print('[augment_sgd] \nintrinsic noise:', intrinsic_noise, '\nnoise to add:', noise_to_add)

    if np.abs(noise_to_add) < 1e-5:
        print('[augment_sgd] Hurray! Essentially no noise required!')

    # noise using different initialisation
    intrinsic_noise_diffinit = derived_results.estimate_variability(cfg_name, model, t,
                                                                    multivariate=False, diffinit=True)

    if intrinsic_noise_diffinit < target_noise:
        noise_to_add_diffinit = compute_additional_noise(target_noise, intrinsic_noise_diffinit)
    else:
        noise_to_add_diffinit = 0
    print(f'[augment_sgd_diffinit] \nintrinsic noise: {intrinsic_noise_diffinit}\nnoise to add {noise_to_add_diffinit}')

    if np.abs(noise_to_add_diffinit) < 1e-5:
        print('[augment_sgd] Hurray! Essentially no noise required!')

    if noise_to_add_diffinit > noise_to_add:
        print('WARNING: Noise from diffinit is... lower than without it?')
        ipdb.set_trace()

    return target_noise, noise_to_add, noise_to_add_diffinit


def test_model_with_noise(cfg_name, replace_index, seed, t,
                          epsilon=None, delta=None,
                          sens_from_bound=True,
                          metric_to_report='binary_accuracy',
                          verbose=False, num_deltas=1000,
                          data_privacy='all'):
    """
    test the model on the test set of the respective dataset
    """
    cfg = load_cfg(cfg_name)
    model = cfg['model']['architecture']
    experiment = ExperimentIdentifier(cfg_name=cfg_name, model=model,
                                      replace_index=replace_index, seed=seed,
                                      diffinit=True)
    task, batch_size, lr, _, N = em.get_experiment_details(cfg_name, model, data_privacy)
    # load the test set
    # TODO this is a hack, fix it
    _, _, _, _, x_test, y_test = data_utils.load_data(cfg['data'], replace_index=replace_index)

    if epsilon is None:
        epsilon = 1.0

    if delta is None:
        delta = 1.0/(N**2)

    if verbose:
        print('Adding noise for epsilon, delta = ', epsilon, delta)

    if sens_from_bound:
        if model == 'logistic':
            lipschitz_constant = np.sqrt(2)
        else:
            raise ValueError(model)
        # optionally estimate empirical lipschitz ...?
        sensitivity = compute_wu_bound(lipschitz_constant, t=t, N=N, batch_size=batch_size, eta=lr)
    else:
        # compute sensitivity empirically!
        # diffinit set to False beacuse it doesn't make a differnce
        sensitivity = derived_results.estimate_sensitivity_empirically(cfg_name, model, t,
                                                                       num_deltas=num_deltas,
                                                                       diffinit=False,
                                                                       data_privacy=data_privacy)

    if sensitivity is False:
        print('ERROR: Empirical sensitivity not available.')

        return False
    print('Sensitivity:', sensitivity)

    target_noise, noise_to_add, noise_to_add_diffinit = get_target_noise_for_model(cfg_name, model, t,
                                                                                   epsilon, delta,
                                                                                   sensitivity, verbose)

    weights_path = experiment.path_stub().with_name(experiment.path_stub().name + '.weights.csv')
    print('Evaluating model from', weights_path)

    model_object = model_utils.build_model(**cfg['model'], init_path=weights_path, t=t)
    # this is for initialising the model etc.
    model_utils.prep_for_training(model_object, seed=0,
                                  optimizer_settings=cfg['training']['optimization_algorithm'],
                                  task_type=cfg['model']['task_type'])

    metric_names = model_object.metric_names
    metric_functions = model_utils.define_metric_functions(metric_names)
    metrics = model_object.compute_metrics(x_test, y_test, metric_functions=metric_functions)
    metrics = [m.numpy() for m in metrics]

    if verbose:
        print('PERFORMANCE (no noise):')

    for (n, v) in zip(metric_names, metrics):
        if verbose:
            print(n, v)

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
        model_object = model_utils.build_model(**cfg['model'], init_path=weights_path, t=t)
        model_utils.prep_for_training(model_object, seed=0,
                                      optimizer_settings=cfg['training']['optimization_algorithm'],
                                      task_type=cfg['model']['task_type'])
        weights = model_object.get_weights(flat=True)
        noise = noise_options[setting]
        noisy_weights = weights + standard_noise * noise
        unflattened_noisy_weights = model_object.unflatten_weights(noisy_weights)
        model_object.set_weights(unflattened_noisy_weights)

        metrics = model_object.compute_metrics(x_test, y_test, metric_functions=metric_functions)
        metrics = [m.numpy() for m in metrics]

        if verbose:
            print(f'PERFORMANCE ({setting}):')

        for (n, v) in zip(metric_names, metrics):
            if verbose:
                print(n, v)

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


def compute_wu_bound_strong(lipschitz_constant, gamma, n_samples, batch_size, verbose=True):
    """
    compute the bound in the strongly convex case
    basically copied from
    https://github.com/tensorflow/privacy/blob/1ce8cd4032b06e8afa475747a105cfcb01c52ebe/tensorflow_privacy/privacy/bolt_on/optimizers.py
    """
    # note that for the strongly convex setting, the learning rate at every point is the minimum of (1/beta, 1/(eta *t))
    # this isn't really important here
    # it's just good to remember that if this wasn't the case, this bound doesn't hold! (that we know of)
    l2_sensitivity = (2 * lipschitz_constant) / (gamma * n_samples * batch_size)

    if verbose:
        print('[test_private_model] Bound on L2 sensitivity:', l2_sensitivity)

    return l2_sensitivity


def compute_wu_bound(lipschitz_constant, t, N, batch_size, eta, verbose=True):
    # k is the number of time you went through the data
    batches_per_epoch = N // batch_size
    # t is the number of batches
    n_epochs = t / batches_per_epoch

    if n_epochs < 1:
        if verbose:
            print('WARNING: <1 pass competed')
        # TODO: make sure we can treat k like this
    l2_sensitivity = 2 * n_epochs * lipschitz_constant * eta / batch_size

    if verbose:
        print('[test_private_model] Bound on L2 sensitivity:', l2_sensitivity)

    return l2_sensitivity


def discretise_theoretical_sensitivity(cfg_name, model, theoretical_sensitivity):
    """
    stop treating k as a float, and turn it back into an integer!
    """
    _, batch_size, lr, _, N = em.get_experiment_details(cfg_name, model)

    if model == 'logistic':
        L = np.sqrt(2)
    else:
        raise ValueError(model)
    k = batch_size * theoretical_sensitivity/(2 * L * lr)
    # we take the ceiling, basically because this is an upper bound
    discrete_k = np.ceil(k)
    discretised_sensitivity = 2 * discrete_k * L * lr / batch_size

    return discretised_sensitivity
