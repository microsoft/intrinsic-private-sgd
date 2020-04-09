#!/usr/bin/env ipython
# Define experimenta metadata

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
