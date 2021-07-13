#!/usr/bin/env ipython
# Define experimenta metadata
from cfg_utils import load_cfg

dataset_colours = {'cifar2_lr': '#A6373F',
                   'cifar2_mlp': '#A6373F',
                   'cifar2_pretrain_lr': '#A6373F',
                   'cifar2_pretrain_mlp': '#A6373F',
                   'mnist_binary_lr': '#552B72',
                   'mnist_binary_mlp': '#552B72',
                   'adult_lr': '#db9302',
                   'adult_mlp': '#db9302',
                   'forest_lr': '#3C8D2f',
                   'forest_mlp': '#3C8D2f'}

dataset_names = {'cifar2_lr': 'CIFAR2',
                 'cifar2_mlp': 'CIFAR2',
                 'cifar2_pretrain_lr': 'CIFAR2',
                 'cifar2_pretrain_mlp': 'CIFAR2',
                 'mnist_binary_lr': 'MNIST-binary',
                 'mnist_binary_mlp': 'MNIST-binary',
                 'adult_lr': 'Adult',
                 'adult_mlp': 'Adult',
                 'forest_lr': 'Forest',
                 'forest_mlp': 'Forest',
                 'mnist_cnn': 'MNIST'}

model_names = {'logistic': 'logistic regression',
               'mlp': 'neural network',
               'cnn': 'CNN'}

lr_convergence_points = {'cifar2_lr': 2000,
                         'cifar2_pretrain_lr': 1000,
                         'mnist_binary_lr': 1900,
                         'adult_lr': 3400,
                         'forest_lr': 8400}

nn_convergence_points = {'cifar2_mlp': 2500,
                         'cifar2_pretrain_mlp': 1500,
                         'mnist_binary_mlp': 4750,
                         'adult_mlp': 1850,
                         'forest_mlp': 3500,
                         'mnist_square_mlp': 1000}

dp_colours = {'augment': '#14894e',
              'both': 'black',
              'augment_diffinit': '#441e85',
              'both_diffinit': 'black',
              'bolton': '#c3871c'}

## These ones are greyscale friendly
dp_colours_gs = {'augment': '#2ea71b',
              'both': 'black',
              'augment_diffinit': '#4a2189',
              'both_diffinit': 'black',
              'bolton': '#ca8621'}


def get_dataset_size(data_cfg):
    """ Note that this is the size of the training dataset """
    name = data_cfg['name']
    if name == 'mnist':
        if data_cfg['binary']:
            N = 10397
        else:
            N = 54000
    elif name in ['cifar10', 'cifar10_pretrain']:
        if data_cfg['binary']:
            N = 9000
        else:
            assert data_cfg['subset']
            N = 15000
    elif name == 'adult':
        N = 29305
    elif name == 'forest':
        N = 378783
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
        elif dataset_name == 'cifar10':
            n_weights = 521
        elif dataset_name == 'cifar10_pretrain':
            n_weights = 661

    return n_weights


def get_input_hidden_size(cfg_name):
    cfg = load_cfg(cfg_name)
    input_size = cfg['model']['input_size']
    hidden_size = cfg['model']['hidden_size']
    return input_size, hidden_size


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
