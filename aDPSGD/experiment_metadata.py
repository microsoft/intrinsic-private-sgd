#!/usr/bin/env ipython
# Define experimenta metadata

dataset_colours = {'cifar10_binary': '#A6373F',
                   'mnist_binary_pca': '#552B72',
                   'adult': '#db9302',
                   'forest': '#3C8D2f'}

dataset_names = {'cifar10_binary': 'CIFAR2',
                 'mnist_binary_pca': 'MNIST-binary',
                 'adult': 'Adult',
                 'forest': 'Forest',
                 'mnist_square': 'MNIST'}

dataset_sizes = {'cifar10_binary': 'CIFAR2',
                 'mnist_binary_pca': 'MNIST-binary',
                 'adult': 'Adult',
                 'forest': 'Forest',
                 'mnist_square': 'MNIST'}

model_names = {'logistic': 'logistic regression',
               'mlp': 'neural network',
               'cnn': 'CNN'}

lr_convergence_points = {'cifar10_binary': 2000,
                         'mnist_binary_pca': 1850,
                         'adult': 3400,
                         'forest': 8400}

nn_convergence_points = {'cifar10_binary': 2500,
                         'mnist_binary_pca': 4750,
                         'adult': 1850,
                         'forest': 3500}

dp_colours = {'augment': '#14894e',
              'both': 'black',
              'augment_diffinit': '#441e85',
              'both_diffinit': 'black',
              'bolton': '#c3871c'}


def get_dataset_size(data_cfg):
    name = data_cfg['name']
    if name == 'mnist':
        if data_cfg['binary']:
            N = 10397
        else:
            N = 54000
    else:
        raise NotImplementedError
    return N


def get_experiment_details(cfg_name, model, verbose=False, data_privacy='all'):
    if cfg_name == 'housing_binary':
        task = 'binary'
        batch_size = 28
        lr = 0.1

        if model in ['linear', 'logistic']:
            n_weights = 14
        elif model == 'mlp':
            n_weights = 121
        N = 364
    elif 'mnist_binary' in cfg_name:
        task = 'binary'
        batch_size = 32

        if 'buggy' in cfg_name:
            lr = 0.1
        else:
            lr = 0.5

        if model == 'logistic':
            if 'cropped' in cfg_name:
                n_weights = 101
            else:
                n_weights = 51
        elif model == 'mlp':
            n_weights = 521
        N = 10397
    elif 'cifar10_binary' in cfg_name:
        task = 'binary'
        batch_size = 32
        lr = 0.5

        if model == 'logistic':
            n_weights = 51
        N = 9000
    elif cfg_name == 'protein':
        task = 'binary'
        batch_size = 50
        lr = 0.01
        assert model == 'logistic'
        n_weights = 75
        N = 65589
    elif 'forest' in cfg_name:
        task = 'binary'
        batch_size = 50
        lr = 1.0

        if model == 'logistic':
            n_weights = 50
        else:
            n_weights = 511
        N = 378783
    elif 'adult' in cfg_name:
        task = 'binary'
        batch_size = 32
        lr = 0.5            # possibly check this

        if 'pca' in cfg_name:
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
    elif cfg_name in ['mnist', 'mnist_square']:
        task = 'classification'
        batch_size = 32
        lr = 0.1

        if model == 'mlp':
            n_weights = 986
        elif model == 'cnn':
            n_weights = 1448
        N = 54000
    elif cfg_name == 'cifar10':
        task = 'classification'
        batch_size = 32
        lr = 0.01

        if model == 'mlp':
            n_weights = 7818
        elif model == 'cnn':
            raise NotImplementedError
        N = 45000
    else:
        raise ValueError(cfg_name)

    if not data_privacy == 'all':
        N = N/2

    if verbose:
        print('Experiment details:')
        print('\tcfg name:', cfg_name)
        print('\tModel:', model)
        print('\tTask:', task)
        print('\tbatch size:', batch_size)
        print('\tlr:', lr)
        print('\tn_weights:', n_weights)
        print('\tN:', N)

    return task, batch_size, lr, n_weights, N
