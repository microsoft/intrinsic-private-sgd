#!/usr/bin/env ipython
# Define experimenta metadata
from cfg_utils import load_cfg

dataset_colours = {'cifar10_binary': '#A6373F',
                   'mnist_binary_pca': '#552B72',
                   'mnist_binary': '#552B72',       # TODO update
                   'adult': '#db9302',
                   'forest': '#3C8D2f'}

dataset_names = {'cifar10_binary': 'CIFAR2',
                 'mnist_binary_pca': 'MNIST-binary',
                 'mnist_binary': 'MNIST-binary',
                 'adult': 'Adult',
                 'forest': 'Forest',
                 'mnist_square': 'MNIST'}

model_names = {'logistic': 'logistic regression',
               'mlp': 'neural network',
               'cnn': 'CNN'}

lr_convergence_points = {'cifar10_binary': 2000,
                         'mnist_binary_pca': 1850,
                         'mnist_binary': 1850,       # TODO update
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
    """ Note that this is the size of the training dataset """
    name = data_cfg['name']
    if name == 'mnist':
        if data_cfg['binary']:
            N = 10397
        else:
            N = 54000
    elif name == 'cifar10':
        if data_cfg['binary']:
            N = 9000
        else:
            raise ValueError(data_cfg['binary'])
    else:
        raise NotImplementedError
    return N


def get_n_weights(cfg):
    model = cfg['model']['architecture']
    if model == 'logistic':
        n_weights = cfg['model']['input_size'] + 1
    else:
        dataset_name = cfg['data']['name']
        binary = cfg['data']['binary']
        if dataset_name == 'mnist':
            if binary:
                n_weights = 521
            else:
                if model == 'mlp':
                    n_weights = 986
                elif model == 'cnn':
                    n_weights = 1448
        elif dataset_name == 'forest':
            n_weights = 511
        elif dataset_name == 'adult':
            n_weights = 817

    return n_weights


def get_experiment_details(cfg_name, model, verbose=False, data_privacy='all'):
    cfg = load_cfg(cfg_name)
    task = cfg['model']['task_type']
    batch_size = cfg['training']['batch_size']
    lr = cfg['training']['optimization_algorithm']['learning_rate']
    N = get_dataset_size(cfg['data'])
    n_weights = get_n_weights(cfg)

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
