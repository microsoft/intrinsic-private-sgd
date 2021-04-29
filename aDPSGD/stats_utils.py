#!/usr/bin/env ipython
# Functions mostly around fitting distributions

import numpy as np
import pandas as pd
import ipdb
# from pingouin import multivariate_normality
from scipy.stats import kstest, laplace, shapiro, anderson, invwishart
import results_utils

smaller_N = 10000  # due to memory errors


def fit_alpha_stable(X):
    N = X.shape[0]
    # copied from umut

    for i in range(1, 1 + int(np.sqrt(N))):
        if N % i == 0:
            m = i
    alpha = alpha_estimator(m, X)
    # dont' know how to estimate goodness of fit for this distribution yet
    goodness_of_fit = np.nan

    return alpha, goodness_of_fit


def fit_multivariate_normal(X):
    from pingouin import multivariate_normality
    try:
        _, pval = multivariate_normality(X, alpha=.05)
    except MemoryError:
        print(f'WARNING: X with size {X.shape} is too big for multivariate normal fit!')
        N = X.shape[0]
        X_smaller = X[np.random.choice(N, smaller_N, replace=False), :]
        print(f'Trying with smaller X of size {X_smaller.shape}!')
        _, pval = multivariate_normality(X_smaller, alpha=.05)
    mean = X.mean(axis=0)
    cov = np.cov(X.T)

    return mean, cov, None, pval


def fit_laplace(X):
    loc = np.median(X)
    scale = np.mean(np.abs(X) - loc)
    #Dval_lap, pval_lap = kstest(X, laplace(loc=loc, scale=scale).cdf)
    Dval_lap, pval_lap = anderson(X, laplace(loc=loc, scale=scale).cdf)         # anderson seems better than kstest, lower variance
    return loc, scale, Dval_lap, pval_lap


def test_laplace():
    pvals_gauss = []
    pvals_laplace = []
    pvals_uniform = []
    ns = []
    for r in range(1, 10):
        for n in [500, 625, 750, 875, 1000, 1500, 2000, 5000]:
            gauss = np.random.normal(size=n)
            laplace = np.random.laplace(loc=0, scale=1, size=n)
            uniform = np.random.uniform(size=n)
            _, _, _, pval_laplace = fit_laplace(laplace)
            _, _, _, pval_gauss = fit_laplace(gauss)
            _, _, _, pval_uniform = fit_laplace(uniform)
            pvals_gauss.append(pval_gauss)
            pvals_laplace.append(pval_laplace)
            pvals_uniform.append(pval_uniform)
            ns.append(n)
    df = pd.DataFrame({'pval_gauss': pvals_gauss,
                       'pval_laplace': pvals_laplace,
                       'pval_uniform': pvals_uniform,
                       'n': ns})
    return df


def test_multivariate_normal():
    """ compute pval across grid of N and d for diagonal Gaussian, non-diagonal Gaussian, Laplace """
    max_d = 60  ## the HZ test implementation in pingouin (maybe generally) fails d larger than this...
    ns = []
    ds = []
    pvals_diagonal_gauss = []
    pvals_nondiag_gauss = []
    pvals_laplace = []
    replicate = []
    for r in range(1, 10): # replicates
        fixed_cov = invwishart.rvs(df=max_d, scale=np.eye(max_d))
        for d in range(5, max_d, 5):
            for n in [75, 100, 250, 500, 625, 750, 875, 1000]:
                if n < d:
                    continue
                diagonal_gauss = np.random.multivariate_normal(mean=np.zeros(d), cov=np.eye(d), size=n)
                nondiag_gauss = np.random.multivariate_normal(mean=np.zeros(d), cov=fixed_cov[:d, :d], size=n)
                laplace = np.random.laplace(loc=0, scale=1, size=(n, d))

                _, _, _, pval_diagonal_gauss = fit_multivariate_normal(diagonal_gauss)
                _, _, _, pval_nondiag_gauss = fit_multivariate_normal(nondiag_gauss)
                _, _, _, pval_laplace = fit_multivariate_normal(laplace)
                if np.isnan(pval_diagonal_gauss):
                    print(f'd: {d}, n: {n}')
                    ipdb.set_trace()

                pvals_diagonal_gauss.append(pval_diagonal_gauss)
                pvals_nondiag_gauss.append(pval_nondiag_gauss)
                pvals_laplace.append(pval_laplace)
                ns.append(n)
                ds.append(d)
                replicate.append(r)

    results = pd.DataFrame({'n': ns, 'd': ds,
                            'pval_diagonal_gauss': pvals_diagonal_gauss,
                            'pval_nondiag_gauss': pvals_nondiag_gauss,
                            'pval_laplace': pvals_laplace,
                            'replicate': replicate})
    return results


def alpha_estimator(m, X):
    """
    this is taken from
    https://github.com/umutsimsekli/sgd_tail_index/blob/master/utils.py
    and modified to remove torchiness
    # Corollary 2.4 in Mohammadi 2014

    X: gradient noise (grad - minibatch grad)
    m: K1 I think (n is K2)
    """
    print(f'alpha estimator using m = {m}')
    # X is N by d matrix
    N = len(X)           # number of gradients, basically
    n = int(N/m)        # must be an integer: this is K2 in the theorem
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
    try:
        Dval_lap, pval_lap = kstest(X, laplace(loc=loc, scale=scale).cdf)
    except MemoryError:
        print(f'WARNING: X with size {X.shape} is too big for Laplace fit!')
        N = X.shape[0]
        X_smaller = X[np.random.choice(N, smaller_N, replace=False)]
        print(f'Trying with smaller X of size {X_smaller.shape}!')
        Dval_lap, pval_lap = kstest(X_smaller, laplace(loc=loc, scale=scale).cdf)

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


def mvg_sigma_bound(gamma=None, sensitivity=0.3, delta=1e-5, epsilon=1,
                    m=1, n=1, sigma=0.1, Psi=None):
    if gamma is None:
        gamma = sensitivity/2
    r = min(m, n)
    # harmonic number
    harmonic_r = sum([1/x for x in range(1, r+1)])
    # generalised harmonic number
    harmonic_r12 = sum([1/np.sqrt(x) for x in range(1, r+1)])
    alpha = (harmonic_r + harmonic_r12)*(gamma**2) + 2*harmonic_r*gamma*sensitivity
    print(f'alpha is {alpha}')
    zeta = 2*np.sqrt(-m*n*np.log(delta)) - 2*np.log(delta) + m*n
    print(f'zeta is {zeta}')
    # https://github.com/inspire-group/MVG-Mechansim/issues/1
    #zeta = np.sqrt(zeta)
    beta = 2*(m*n)**(0.25)*harmonic_r*sensitivity*zeta
    print(f'beta is {beta}')
    IB = (-beta + np.sqrt(beta**2 + 8*alpha*epsilon))**2/(4*alpha**2)
    print(f'bound on phi is {IB}')
    Psi = np.eye(1)
    Sigma = np.diag([sigma]*m)
    Psiinv = np.linalg.inv(Psi)
    Sigmainv = np.linalg.inv(Sigma)
    _, Psi_s, _ = np.linalg.svd(Psiinv)          # we could just take the eigenvalues but w/e
    _, Sigma_s, _ = np.linalg.svd(Sigmainv)
    print(Sigma_s)
    phi = np.sqrt(np.linalg.norm(Sigma_s)*np.linalg.norm(Psi_s))
    print(f'phi is {phi}')
    eps_bound = 0.5*(alpha*phi*phi + beta*phi)
    print(f'the bound on epsilon is therefore: {eps_bound}')
    return IB


def uni_sigma_bound(sensitivity=0.3, delta=1e-5, epsilon=1):
    c = np.sqrt(2*np.log(1.25/delta) + 1e-5)
    bound = c*sensitivity/epsilon
    print(f'c is {c}')
    print(f'bound is {bound}')


def estimate_statistics_through_training(what, cfg_name, model, replace_index,
                                         seed, df=None, params=None, sort=False,
                                         iter_range=(None, None), diffinit=True,
                                         include_mvn: bool = True):
    """
    Grab a trace file for a model, estimate the alpha value for gradient noise throughout training
    NOTE: All weights taken together as IID (in the list of params supplied)
    """
    assert what in ['gradients', 'weights']

    if df is None:
        if replace_index is None:
            replace_index = results_utils.get_replace_index_with_most_seeds(cfg_name, model, diffinit=diffinit)
        if what == 'gradients':
            if sort:
                raise ValueError(sort)
            df = results_utils.get_posterior_samples(cfg_name, model=model, replace_index=replace_index,
                                                     iter_range=iter_range, params=params, diffinit=diffinit,
                                                     what='gradients')
        else:
            print('Getting posterior for weights, seed is irrelevant')
            df = results_utils.get_posterior_samples(cfg_name, model=model, replace_index=replace_index,
                                                     iter_range=iter_range, params=params, diffinit=diffinit, sort=sort)

        if df is False:
            print('ERROR: No data found')

            return False

    if include_mvn:
        assert df.shape[1] > 2

    # now go through the iterations
    iterations = df['t'].unique()
    # store the results in this dataframe
    df_fits = pd.DataFrame(index=iterations)
    df_fits.index.name = 't'
    df_fits['N'] = np.nan
    df_fits['alpha'] = np.nan
    df_fits['alpha_fit'] = np.nan

    for t in iterations:
        df_t = df.loc[df['t'] == t, :]
        # zero it out by seed
        if what == 'gradients':
            seed_means = df_t.groupby('seed').transform('mean')
            df_t = (df_t - seed_means).drop(columns=['seed', 't'])
            X = df_t.values
        else:
            X = df_t.iloc[:, 2:].values
            X = X - X.mean(axis=0)
        df_fits['N'] = X.shape[0]
        # fit alpha_stable
        alpha, fit = fit_alpha_stable(X)
        df_fits.loc[t, 'alpha'] = alpha
        df_fits.loc[t, 'alpha_fit'] = fit
        if include_mvn:
            # fit multivariate gaussian - dont record the params since they don't fit...
            _, _, _, p = fit_multivariate_normal(X)
            df_fits.loc[t, 'mvnorm_mu'] = np.nan
            df_fits.loc[t, 'mvnorm_sigma'] = np.nan
            df_fits.loc[t, 'mvnorm_W'] = np.nan
            df_fits.loc[t, 'mvnorm_p'] = p
        # Now flatten and look at univariate distributions
        X_flat = X.reshape(-1, 1)
        df_fits['N_flat'] = X_flat.shape[0]
        # fit univariate gaussian
        mu, sigma, W, p = fit_normal(X_flat)
        df_fits.loc[t, 'norm_mu'] = mu
        df_fits.loc[t, 'norm_sigma'] = sigma
        df_fits.loc[t, 'norm_W'] = W
        df_fits.loc[t, 'norm_p'] = p
        # fit laplace
        loc, scale, D, p = fit_laplace(X_flat)
        df_fits.loc[t, 'lap_loc'] = loc
        df_fits.loc[t, 'lap_scale'] = scale
        df_fits.loc[t, 'lap_D'] = D
        df_fits.loc[t, 'lap_p'] = p

    # Attach what the fit was on
    df_fits.columns = [f'{what}_{x}' for x in df_fits.columns]
    return df_fits
