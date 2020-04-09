#!/usr/bin/env ipython
# Functions mostly around fitting distributions

import numpy as np
from scipy.stats import kstest, laplace, shapiro, anderson

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
