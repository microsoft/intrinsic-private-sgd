#!usr/bin/env ipython
# Functions related to loading, saving, processing datasets

import tensorflow.keras.datasets as datasets
import numpy as np
import pandas as pd
import os
from scipy.stats import entropy
from scipy.spatial.distance import cosine
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA
import ipdb

# CONSTANTS
FOREST_PATH = os.path.join('data', 'covtype.data')
ADULT_PATH = os.path.join('data', 'adult.data')
ADULT_TEST_PATH = os.path.join('data', 'adult.test')


def min_max_rescale(df_train, df_test, good_columns=None):
    if good_columns is None:
        col_mins = df_train.min(axis=0)
        col_maxs = df_train.max(axis=0)
        col_ranges = col_maxs - col_mins
        good_columns = (col_ranges > 0)
    print('Deleting', df_train.shape[1] - sum(good_columns), 'columns for not exhibiting variability')
    df_train = df_train[:, good_columns]
    df_test = df_test[:, good_columns]
    print('Rescaling to [0, 1]...')
    col_mins = df_train.min(axis=0)
    col_maxs = df_train.max(axis=0)
    col_ranges = np.float32(col_maxs - col_mins)
    # if there's no variability, basically just mapping it to 0.5
    col_ranges[col_ranges == 0] = 2*col_maxs[col_ranges == 0] + 1e-5
    df_train = (df_train - col_mins)/col_ranges
    df_test = (df_test - col_mins)/col_ranges
    assert np.isnan(df_train).sum() == 0
    assert np.isnan(df_test).sum() == 0

    return df_train, df_test


def load_data(options, replace_index):
    # these are shared options
    data_type = options['name']
    data_privacy = 'all'
    print('WARNING: Data privacy is fixed to all right now')

    if data_type == 'mnist':
        flatten = options['flatten']
        binary = options['binary']

        if binary:
            # only care about doing this for binary classification atm, could just make an option
            enforce_max_norm = True
        else:
            enforce_max_norm = False

        if 'preprocessing' in options:
            if options['preprocessing'] == 'PCA':
                project = True
                pca = True
                crop = False
            elif options['preprocessing'] == 'GRP':
                project = True
                pca = False
                crop = False
            elif options['preprocessing'] == 'crop':
                project = False
                pca = False
                crop = True
        else:
            project = False
            pca = False
            crop = False
        x_train, y_train, x_test, y_test = load_mnist(binary=binary,
                                                      enforce_max_norm=enforce_max_norm,
                                                      flatten=flatten,
                                                      data_privacy=data_privacy,
                                                      project=project,
                                                      crop=crop,
                                                      pca=pca)
    elif data_type == 'cifar10':
        flatten = options['flatten']
        binary = options['binary']

        if binary:
            enforce_max_norm = True
        else:
            enforce_max_norm = False

        if flatten:
            project = True
            pca = True
        else:
            project = False
            pca = False
        x_train, y_train, x_test, y_test = load_cifar10(binary=binary,
                                                        enforce_max_norm=enforce_max_norm,
                                                        flatten=flatten,
                                                        data_privacy=data_privacy,
                                                        project=project,
                                                        pca=pca)
    elif data_type == 'forest':
        x_train, y_train, x_test, y_test = load_forest(data_privacy=data_privacy)
    elif data_type == 'adult':
        if options['preprocessing'] == 'PCA':
            pca = True
        else:
            pca = False
        x_train, y_train, x_test, y_test = load_adult(data_privacy=data_privacy, pca=pca)
    else:
        raise ValueError(data_type)

    x_train, y_train, x_vali, y_vali, x_test, y_test = validation_split(x_train, y_train, x_test, y_test, replace_index)

    return x_train, y_train, x_vali, y_vali, x_test, y_test


def validation_split(x_train, y_train, x_test, y_test, replace_index):
    # we need to generate a validation set (do it from the train set)
    N = x_train.shape[0]
    n_vali = int(0.1*N)
    vali_idx = range(n_vali)
    train_idx = [i for i in range(N) if i not in vali_idx]
    assert len(set(vali_idx).intersection(set(train_idx))) == 0
    x_vali = x_train[vali_idx]
    y_vali = y_train[vali_idx]
    x_train = x_train[train_idx]
    y_train = y_train[train_idx]

    if replace_index:
        replace_index = int(replace_index)
        # we always replace with ELEMENT 0 (wlog, ish), then don't use the first row
        # (this is to avoid an effect where experiments where the replace_index is low encounter an unusually
        # low-variance batch at the start of training!)
        special_idx = 0
        x_special = x_train[special_idx]
        y_special = y_train[special_idx]
        x_train[replace_index] = x_special
        y_train[replace_index] = y_special
        x_train = np.delete(x_train, special_idx, axis=0)
        y_train = np.delete(y_train, special_idx, axis=0)

    return x_train, y_train, x_vali, y_vali, x_test, y_test


def load_forest(data_privacy='all'):
    path = os.path.join('data', 'forest_' + data_privacy + '.npy')
    try:
        data = np.load(path, allow_pickle=True).item()
        x_train = data['x_train']
        x_test = data['x_test']
        y_train = data['y_train']
        y_test = data['y_test']
    except FileNotFoundError:
        print('Loading...')
        all_data = pd.read_csv(FOREST_PATH, header=None)
        # select just types 1 and 2 (these are the most common)
        print('Selecting classes 1 and 2')
        binary_data = all_data.loc[all_data.iloc[:, -1].isin({1, 2}), :]
        # split into features and labels
        y = binary_data.iloc[:, -1].values
        # rescale to 0 and 1!
        y = y - 1
        assert set(y) == set([0, 1])
        features = binary_data.iloc[:, :-1].values
        assert features.shape[1] == 54
        N = features.shape[0]
        print('Resulting number of examples:', N)
        # test-train split
        print('Doing test-train split')
        train_frac = 0.85
        n_train = int(N*train_frac)
        train_idx = np.random.choice(N, n_train, replace=False)
        test_idx = [x for x in range(N) if x not in train_idx]
        print('n train:', n_train, 'n test:', len(test_idx))
        x_train = features[train_idx, :]
        x_test = features[test_idx, :]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # need to keep this to make sure the columns are all the same... when we do public/private split
        x_train_orig = x_train.copy()

        # do public/private split
        x_train, y_train, x_test, y_test = public_private_split('forest', data_privacy,
                                                                x_train, y_train,
                                                                x_test, y_test)

        # now we need to normalise this
        # rescale to 0-1 first
        col_mins = x_train_orig.min(axis=0)
        col_maxs = x_train_orig.max(axis=0)
        col_ranges = col_maxs - col_mins
        good_columns = (col_ranges > 0)
        del x_train_orig
        x_train, x_test = min_max_rescale(x_train, x_test, good_columns=good_columns)

        # and NOW we project to the unit sphere
        print('Projecting to sphere...')
        x_train = x_train / np.linalg.norm(x_train, axis=1).reshape(-1, 1)
        x_test = x_test / np.linalg.norm(x_test, axis=1).reshape(-1, 1)
        assert np.all(np.abs(np.linalg.norm(x_train, axis=1) - 1) < 1e-6)
        assert np.all(np.abs(np.linalg.norm(x_test, axis=1) - 1) < 1e-6)

        data = {'x_train': x_train,
                'x_test': x_test,
                'y_train': y_train,
                'y_test': y_test}
        print('Saving...')
        np.save(path, data)

    return x_train, y_train, x_test, y_test


def public_private_split(dataset, data_privacy, x_train, y_train, x_test, y_test):
    """
    """

    if data_privacy == 'all':
        print('Including all data')
    else:
        print('Splitting data into public/private!')
        split_path = os.path.join('data', dataset + '_public_private_split.npy')
        try:
            split = np.load(split_path, allow_pickle=True).item()
            print('Loaded pre-computed split from', split_path)
            public_train_idx = split['public_train_idx']
            public_test_idx = split['public_test_idx']
            private_train_idx = split['private_train_idx']
            private_test_idx = split['private_test_idx']
        except FileNotFoundError:
            print('No pre-defined split found!')
            N_train = x_train.shape[0]
            N_test = x_test.shape[0]

            public_train_idx = np.random.choice(N_train, int(0.5*N_train), replace=False)
            public_test_idx = np.random.choice(N_test, int(0.5*N_test), replace=False)
            private_train_idx = np.array([i for i in range(N_train) if i not in public_train_idx])
            private_test_idx = np.array([i for i in range(N_test) if i not in public_test_idx])
            assert len(set(public_train_idx).intersection(set(private_train_idx))) == 0
            assert len(set(public_test_idx).intersection(set(private_test_idx))) == 0

            split = {'public_train_idx': public_train_idx,
                     'public_test_idx': public_test_idx,
                     'private_train_idx': private_train_idx,
                     'private_test_idx': private_test_idx}
            np.save(split_path, split)
            print('Saved split to', split_path)

        if data_privacy == 'public':
            x_train = x_train[public_train_idx]
            y_train = y_train[public_train_idx]
            x_test = x_test[public_test_idx]
            y_test = y_test[public_test_idx]
        elif data_privacy == 'private':
            x_train = x_train[private_train_idx]
            y_train = y_train[private_train_idx]
            x_test = x_test[private_test_idx]
            y_test = y_test[private_test_idx]

    return x_train, y_train, x_test, y_test


def load_mnist(binary=False, enforce_max_norm=False, flatten=True,
               data_privacy='all', project=True, pca=False, crop=False):
    dataset_identifier = 'mnist' + '_' + data_privacy + '_binary'*binary + '_maxnorm'*enforce_max_norm + '_square'*(not flatten) + '_pca'*pca + '_crop'*crop + '.npy'
    dataset_string = os.path.join('data', dataset_identifier)
    try:
        data = np.load(dataset_string, allow_pickle=True).item()
        x_train = data['x_train']
        x_test = data['x_test']
        y_train = data['y_train']
        y_test = data['y_test']
        print('Loaded data from', dataset_string)
    except FileNotFoundError:
        print('Couldn\'t load data from', dataset_string)
        # cant load from file, build it up again
        mnist = datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train, y_train, x_test, y_test = public_private_split('mnist', data_privacy, x_train, y_train, x_test, y_test)

        if binary:
            # keep only 3 and 5 (I chose these randomly)
            keep_train = (y_train == 3) | (y_train == 5)
            keep_test = (y_test == 3) | (y_test == 5)
            x_train = x_train[keep_train]
            x_test = x_test[keep_test]
            y_train = y_train[keep_train]
            y_test = y_test[keep_test]
            # convert to binary (5 is 1, 3 is 0)
            y_train[y_train == 5] = 1
            y_train[y_train == 3] = 0
            y_test[y_test == 5] = 1
            y_test[y_test == 3] = 0
            # sanity check
            assert set(y_train) == {1, 0}
            assert set(y_test) == {1, 0}

        # typical normalisation
        x_train, x_test = x_train/255.0, x_test/255.0

        if crop:
            assert x_train.shape[1:] == (28, 28)
            assert x_test.shape[1:] == (28, 28)
            x_train = x_train[:, 9:19, 9:19]
            x_test = x_test[:, 9:19, 9:19]
            side_length = 10
        else:
            side_length = 28

        if flatten:
            x_train = x_train.reshape(-1, side_length*side_length)
            x_test = x_test.reshape(-1, side_length*side_length)

            if project:
                # you can only project flattened data
                # by default we do gaussian random projections

                if pca:
                    # do PCA down to 50
                    # in the Abadi paper they do 60 dimensions, but to help comparison with Wu I'd rather do 50 here
                    transformer = PCA(n_components=50)
                else:
                    # do random projection on MNIST
                    # in the Wu paper they project to 50 dimensions
                    transformer = GaussianRandomProjection(n_components=50)
                # fit to train data
                transformer.fit(x_train)
                # transform everything
                x_train = transformer.transform(x_train)
                x_test = transformer.transform(x_test)
                assert x_train.shape[1] == 50
                assert x_test.shape[1] == 50
        else:
            # keeping it not-flat
            # just add a sneaky little dimension on there for the CNN
            x_train = x_train.reshape(-1, side_length, side_length, 1)
            x_test = x_test.reshape(-1, side_length, side_length, 1)

        if enforce_max_norm:
            # slightly different normalisation to what's normal in MNIST

            if len(x_train.shape) == 2:
                axis = (1)
                train_norms = np.linalg.norm(x_train, axis=axis).reshape(-1, 1)
                test_norms = np.linalg.norm(x_test, axis=axis).reshape(-1, 1)
            elif len(x_train.shape) == 4:
                axis = (1, 2)
                train_norms = np.linalg.norm(x_train, axis=axis).reshape(-1, 1, 1, 1)
                test_norms = np.linalg.norm(x_test, axis=axis).reshape(-1, 1, 1, 1)
            else:
                raise ValueError(x_train.shape)
            x_train = np.where(train_norms > 1, x_train/train_norms, x_train)
            x_test = np.where(test_norms > 1, x_test/test_norms, x_test)
            assert np.all(np.abs(np.linalg.norm(x_train, axis=axis) - 1) < 1e-6)
            assert np.all(np.abs(np.linalg.norm(x_test, axis=axis) - 1) < 1e-6)

        data = {'x_train': x_train,
                'x_test': x_test,
                'y_train': y_train,
                'y_test': y_test}

        np.save(dataset_string, data)
        print('Saved data to', dataset_string)

    return x_train, y_train, x_test, y_test


def load_cifar10(binary=False, enforce_max_norm=False, flatten=True,
                 data_privacy='all', project=True, pca=False, crop=False):
    """
    copying what i did for mnist, but for cifar10
    cropping is also a 10x10 square in the middle
    """
    dataset_identifier = 'cifar10' + '_' + data_privacy + '_binary'*binary + '_maxnorm'*enforce_max_norm + '_square'*(not flatten) + '_pca'*pca + '_crop'*crop + '.npy'
    dataset_string = os.path.join('data', dataset_identifier)
    try:
        data = np.load(dataset_string, allow_pickle=True).item()
        x_train = data['x_train']
        x_test = data['x_test']
        y_train = data['y_train']
        y_test = data['y_test']
        print('Loaded data from', dataset_string)
    except FileNotFoundError:
        print('Couldn\'t load data from', dataset_string)
        cifar10 = datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = y_train[:, 0]
        y_test = y_test[:, 0]

        x_train, y_train, x_test, y_test = public_private_split('cifar10', data_privacy,
                                                                x_train, y_train,
                                                                x_test, y_test)

        if binary:
            # keep only 3 and 5
            # coincidentally, although i chose 3 and 5 randomly for MNIST,
            # in CIFAR10 these correspond to cats and dogs, which is a convenient pair
            keep_train = (y_train == 0) | (y_train == 2)
            keep_test = (y_test == 0) | (y_test == 2)
            x_train = x_train[keep_train]
            x_test = x_test[keep_test]
            y_train = y_train[keep_train]
            y_test = y_test[keep_test]
            # convert to binary (2 is 1, 0 is 0)
            y_train[y_train == 2] = 1
            y_train[y_train == 0] = 0
            y_test[y_test == 2] = 1
            y_test[y_test == 0] = 0
            # sanity check
            assert set(y_train) == {1, 0}
            assert set(y_test) == {1, 0}

        # typical normalisation
        x_train, x_test = x_train/255.0, x_test/255.0

        if crop:
            assert x_train.shape[1:] == (32, 32, 3)
            assert x_test.shape[1:] == (32, 32, 3)
            x_train = x_train[:, 11:21, 11:21, :]
            x_test = x_test[:, 11:21, 11:21, :]
            side_length = 10
        else:
            side_length = 32

        if flatten:
            # greyscale conversion from RGB
            # Y = 0.2989 R + 0.5870 G + 0.1140 B
            # greyscale_weights = [0.2989, 0.5870, 0.1140]
            # x_train = 1 - np.dot(x_train, greyscale_weights)
            # x_test = 1 - np.dot(x_test, greyscale_weights)
            x_train = x_train.reshape(-1, side_length*side_length*3)
            x_test = x_test.reshape(-1, side_length*side_length*3)

            if project:
                # you can only project flattened data
                n_dim = 50
                # by default we do gaussian random projections

                if pca:
                    # do PCA down to 50
                    # in the Abadi paper they do 60 dimensions, but to help comparison with Wu I'd rather do 50 here
                    transformer = PCA(n_components=n_dim)
                else:
                    # do random projection on MNIST
                    # in the Wu paper they project to 50 dimensions
                    transformer = GaussianRandomProjection(n_components=n_dim)
                # fit to train data
                transformer.fit(x_train)
                # transform everything
                x_train = transformer.transform(x_train)
                x_test = transformer.transform(x_test)
                assert x_train.shape[1] == n_dim
                assert x_test.shape[1] == n_dim
        else:
            # keeping it not-flat
            assert len(x_train.shape) == 4
            assert len(x_test.shape) == 4

        if enforce_max_norm:
            if len(x_train.shape) == 2:
                axis = (1)
                train_norms = np.linalg.norm(x_train, axis=axis).reshape(-1, 1)
                test_norms = np.linalg.norm(x_test, axis=axis).reshape(-1, 1)
            elif len(x_train.shape) == 4:
                axis = (1, 2)
                train_norms = np.linalg.norm(x_train, axis=axis).reshape(-1, 1, 1, 1)
                test_norms = np.linalg.norm(x_test, axis=axis).reshape(-1, 1, 1, 1)
            else:
                raise ValueError(x_train.shape)
            x_train = np.where(train_norms > 1, x_train/train_norms, x_train)
            x_test = np.where(test_norms > 1, x_test/test_norms, x_test)
            assert np.all(np.abs(np.linalg.norm(x_train, axis=axis) - 1) < 1e-6)
            assert np.all(np.abs(np.linalg.norm(x_test, axis=axis) - 1) < 1e-6)

        data = {'x_train': x_train,
                'x_test': x_test,
                'y_train': y_train,
                'y_test': y_test}

        np.save(dataset_string, data)
        print('Saved data to', dataset_string)

    return x_train, y_train, x_test, y_test


def load_adult(data_privacy='all', pca=False):
    """
    """
    path = os.path.join('data', 'adult' + '_' + data_privacy + '_pca'*pca + '.npy')
    try:
        data = np.load(path, allow_pickle=True).item()
        x_train = data['x_train']
        y_train = data['y_train']
        x_test = data['x_test']
        y_test = data['y_test']
        print('Loaded from file')
    except FileNotFoundError:
        adult_header = ['age',
                        'workclass',
                        'fnlwgt',
                        'education',
                        'education-num',
                        'marital-status',
                        'occupation',
                        'relationship',
                        'race',
                        'sex',
                        'capital-gain',
                        'capital-loss',
                        'hours-per-week',
                        'native-country',
                        'label']
        df = pd.read_csv(ADULT_PATH, sep=', ', header=None)
        df_test = pd.read_csv(ADULT_TEST_PATH, sep=', ', skiprows=1, header=None)
        df.columns = adult_header
        df_test.columns = adult_header
        label_replace_dict = {'>50K': 1, '<=50K': 0,
                              '>50K.': 1, '<=50K.': 0}
        y_train = df['label'].replace(label_replace_dict).values
        y_test = df_test['label'].replace(label_replace_dict).values
        assert set(y_train) == set([0, 1])
        assert set(y_test) == set([0, 1])
        x_train = df.iloc[:, :-1]
        x_test = df_test.iloc[:, :-1]

        # need to one-hot encode
        # pd.dummies does this, it is also smart about identifying categorical columns
        x_train = pd.get_dummies(x_train, drop_first=True)
        x_test = pd.get_dummies(x_test, drop_first=True)
        # need to make sure they have exactly the same columns
        missing_in_test = set(x_train.columns).difference(set(x_test.columns))
        print('Inserting columns into test:', missing_in_test)

        for col in missing_in_test:
            x_test[col] = 0
        missing_in_train = set(x_test.columns).difference(set(x_train.columns))
        print('Inserting columns into train:', missing_in_train)

        for col in missing_in_train:
            x_train[col] = 0
        assert set(x_test.columns) == set(x_train.columns)
        # now put them in the same order
        x_test = x_test[x_train.columns]
        assert np.all(x_train.columns == x_train.columns)

        # now convert to features
        x_train = x_train.values
        x_test = x_test.values

        x_train_orig = x_train.copy()
        # do public/private split
        x_train, y_train, x_test, y_test = public_private_split('adult', data_privacy,
                                                                x_train, y_train,
                                                                x_test, y_test)

        # now we need to normalise this
        # rescale to 0-1 first
        col_mins = x_train_orig.min(axis=0)
        col_maxs = x_train_orig.max(axis=0)
        col_ranges = col_maxs - col_mins
        good_columns = (col_ranges > 0)
        del x_train_orig
        # now normalise
        x_train, x_test = min_max_rescale(x_train, x_test, good_columns=good_columns)

        # pca, if pca

        if pca:
            print('doing PCA!')
            ipdb.set_trace()
            transformer = PCA(n_components=50)
            transformer.fit(x_train)
            # transform everything
            x_train = transformer.transform(x_train)
            x_test = transformer.transform(x_test)
            ipdb.set_trace()

        # now project to sphere
        print('Projecting to sphere...')
        x_train = x_train / np.linalg.norm(x_train, axis=1).reshape(-1, 1)
        x_test = x_test / np.linalg.norm(x_test, axis=1).reshape(-1, 1)
        assert np.all(np.abs(np.linalg.norm(x_train, axis=1) - 1) < 1e-6)
        assert np.all(np.abs(np.linalg.norm(x_test, axis=1) - 1) < 1e-6)

        # double-check sizes
        assert x_train.shape[0] == y_train.shape[0]
        assert x_test.shape[0] == y_test.shape[0]
        assert x_train.shape[1] == x_test.shape[1]

        # now save
        data = {'x_train': x_train,
                'x_test': x_test,
                'y_train': y_train,
                'y_test': y_test}
        print('Saving...')
        np.save(path, data)

    return x_train, y_train, x_test, y_test


def solve_with_linear_regression(dataset, replace_index=None):
    """
    assuming linear regression (mse loss, linear model) on dataset, compute the optimum value and the hessian at that point (on the test data)
    """
    x, y, _, _, _, _ = load_data(dataset, replace_index=replace_index)
    # for linear regression, the hessian is constant (although dependent on the data ofc)
    N, d = x.shape
    # have to add a column onto x to account for the bias in the linear model
    bias = np.ones((N, 1))
    x = np.hstack([x, bias])
    hessian = (2.0/N)*np.dot(x.T, x)
    assert hessian.shape[0] == hessian.shape[1]
    assert hessian.shape[0] == d + 1

    # optimum = np.dot(np.linalg.inv(hessian), np.dot(x.T, y))
    optimum = np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y))

    # report the loss
    mse = np.mean((np.dot(x, optimum) - y)**2)
    print(mse)

    return optimum, hessian


def compute_JS_distance(samples_A, samples_B, bins='auto'):
    """
    Assuming samples_A and samples_B are samples from distributions A and B,
    compute the (approximate) JS distance between them by:
    - converting each set of samples to a histogram defined over the same discretised space (with granularity given by bins)
    - computing the relative entropy both ways

    WARNING: not sure how the sensitivity to granularity may impact results here
    """
    # convert both to empirical PMFs
    hist_A, bin_edges = np.histogram(samples_A, density=True, bins=bins)
    hist_B, bin_edges_B = np.histogram(samples_B, bins=bin_edges, density=True)
    assert np.array_equal(bin_edges, bin_edges_B)
    # get the middle distribution
    hist_M = 0.5*(hist_A + hist_B)
    # compute the KL divergence both ways
    KL_AM = entropy(hist_A, hist_M)
    KL_BM = entropy(hist_B, hist_M)
    # now get the JS
    JS = 0.5*(KL_AM + KL_BM)

    return JS


def compute_cosine_distances_for_dataset(data_type):
    """
    compute the cosine distance between two samples of a dataset
    (assuming training data!)
    focusing on
    """
    path = os.path.join('data', data_type + '.cosine_distances.npy')
    try:
        data = np.load(path, allow_pickle=True).item()
        pairs = data['pairs']
        distances = data['distances']
        print('Loaded from file')
    except FileNotFoundError:
        x, y, _, _, _, _ = load_data(data_type, replace_index='NA')
        N = x.shape[0]
        n_distances = int(N*(N-1)/2)
        distances = np.zeros(n_distances)
        print('computing distances between', n_distances, 'pairs of training examples...!')
        pairs = [0]*n_distances
        counter = 0

        for i in range(0, N):
            for j in range(i+1, N):
                if counter % 10000 == 0:
                    print(counter)
                zi = np.append(x[i], y[i])
                zj = np.append(x[j], y[j])
                distances[counter] = cosine(zi, zj)
                pairs[counter] = (i, j)
                counter += 1
        assert len(distances) == len(pairs)
        data = {'pairs': pairs, 'distances': distances}
        np.save(path, data)

    return distances, pairs


def compute_distance_for_pairs(data_type, pairs):
    x, y, _, _, _, _ = load_data(data_type, replace_index='NA')
    d = x.shape[1]
    distances = np.zeros((len(pairs), 2*d + 2))

    for k, (idx1, idx2) in enumerate(pairs):
        z1 = np.append(x[idx1], y[idx1])
        z2 = np.append(x[idx2], y[idx2])
        distances[k] = np.append(z1, z2)

    return distances
