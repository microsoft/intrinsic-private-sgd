#!/usr/bin/env ipython
# The functions in this file relate to evaluating the performance of the model
# Specifically we are interested in the utility of models with different privacy levels

import numpy as np
import experiment_metadata as em


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


def compute_additional_noise(target_sigma, intrinsic_noise):
    """
    assuming we want to get to target_sigma STDDEV, from intrinsic noise,
    independent gaussians
    """
    additional_noise = np.sqrt(target_sigma**2 - intrinsic_noise**2)

    if type(additional_noise) == np.ndarray:
        additional_noise[intrinsic_noise > target_sigma] = 0

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
        print('[noise_utils] Bound on L2 sensitivity:', l2_sensitivity)

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
