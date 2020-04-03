#!/usr/bin/env ipython

import model_utils
import data_utils
import numpy as np
import ipdb

from sacred import Experiment

ex = Experiment('sgd_dp')

@ex.named_config
def cifar10_binary():
    data_type = 'cifar10_binary'
    data_privacy = 'all'
    model_type = 'logistic'
    n_epochs = 25
    init_path = 'models/' + model_type + '/cifar10_binary_init.h5'
    replace_index = 'NA'
    seed = 1
    lr = 0.5        # need to figure this out
    cadence = 50
    if init_path is None:
        identifier = data_type + '/' + data_privacy + '/' + model_type + '/' + model_type + '_DIFFINIT' + '.replace_' + str(replace_index) + '.seed_' + str(seed)
    else:
        identifier = data_type + '/' + data_privacy + '/' + model_type + '/' + model_type + '.replace_' + str(replace_index) + '.seed_' + str(seed)
    examine_gradients = False
    task_type = 'binary'
    batch_size = 32

@ex.named_config
def cifar10():
    data_type = 'cifar10'
    data_privacy = 'all'
    model_type = 'mlp'
    n_epochs = 20
    init_path = 'models/' + model_type + '/cifar_init.h5'
    replace_index = 'NA'
    seed = 1
    lr = 0.01        # this is better for this dataset!
    cadence = 100
    if init_path is None:
        identifier = data_type + '/' + data_privacy + '/' + model_type + '/' + model_type + '_DIFFINIT.drop_' + str(drop_index) + '.replace_' + str(replace_index) + '.seed_' + str(seed)
    else:
        identifier = data_type + '/' + data_privacy + '/' + model_type + '/' + model_type + '.drop_' + str(drop_index) + '.replace_' + str(replace_index) + '.seed_' + str(seed)
    examine_gradients = False
    task_type = 'classification'
    batch_size = 32

@ex.named_config
def cifar10_square():
    data_type = 'cifar10_square'
    data_privacy = 'all'
    model_type = 'cnn'
    n_epochs = 20
    init_path = 'models/' + model_type + '/cifar_square_init.h5'
    replace_index = 'NA'
    seed = 1
    #lr = 0.01        # old
    lr = 0.1        # drop = 1
    cadence = 100
    if init_path is None:
        identifier = data_type + '/' + data_privacy + '/' + model_type + '/' + model_type + '_DIFFINIT.drop_' + str(drop_index) + '.replace_' + str(replace_index) + '.seed_' + str(seed)
    else:
        identifier = data_type + '/' + data_privacy + '/' + model_type + '/' + model_type + '.drop_' + str(drop_index) + '.replace_' + str(replace_index) + '.seed_' + str(seed)
    examine_gradients = False
    task_type = 'classification'
    batch_size = 32

@ex.named_config
def mnist_square():
    # as opposed to flat
    data_type = 'mnist_square'
    data_privacy = 'all'
    model_type = 'cnn'
    n_epochs = 2
    init_path = 'models/' + model_type + '/mnist_square_init.h5'
    replace_index = 'NA'
    seed = 1
    lr = 0.1
    cadence = 50
    if init_path is None:
        identifier = data_type + '/' + data_privacy + '/' + model_type + '/' + model_type + '_DIFFINIT.drop_' + str(drop_index) + '.replace_' + str(replace_index) + '.seed_' + str(seed)
    else:
        identifier = data_type + '/' + data_privacy + '/' + model_type + '/' + model_type + '.drop_' + str(drop_index) + '.replace_' + str(replace_index) + '.seed_' + str(seed)
    examine_gradients = False
    task_type = 'classification'
    batch_size = 32

@ex.named_config
def mnist():
    data_type = 'mnist'
    data_privacy = 'all'
    model_type = 'mlp'
    if True:
        # small
        # hidden state 5
        n_epochs = 3   # 3 for hidden = 5
        init_path = 'models/' + model_type + '/mnist_small_init.h5'
    else:
        # hiden size 16
        n_epochs = 2   # yeah this is quick to converge
        init_path = 'models/' + model_type + '/mnist_init.h5'
    replace_index = 'NA'
    seed = 1
    lr = 0.1
    cadence = 50
    if init_path is None:
        identifier = data_type + '/' + data_privacy + '/' + model_type + '/' + model_type + '_DIFFINIT.drop_' + str(drop_index) + '.replace_' + str(replace_index) + '.seed_' + str(seed)
    else:
        identifier = data_type + '/' + data_privacy + '/' + model_type + '/' + model_type + '.drop_' + str(drop_index) + '.replace_' + str(replace_index) + '.seed_' + str(seed)
    examine_gradients = False
    task_type = 'classification'
    batch_size = 32

@ex.named_config
def mnist_binary():
    data_type = 'mnist_binary'
    data_privacy = 'all'
    model_type = 'mlp'
    #n_epochs = 10    ## was 25, i think 10 is fine at least for logistic...
    n_epochs = 20    ## was 25, i think 10 is fine at least for logistic...
    init_path = 'models/' + model_type + '/mnist_binary_init.h5'
    replace_index = 'NA'
    seed = 1
    #lr = 0.1
    lr = 0.5        # seems to work better
    cadence = 50
    if init_path is None:
        identifier = data_type + '/' + data_privacy + '/' + model_type + '/' + model_type + '_DIFFINIT.drop_' + str(drop_index) + '.replace_' + str(replace_index) + '.seed_' + str(seed)
    else:
        identifier = data_type + '/' + data_privacy + '/' + model_type + '/' + model_type + '.drop_' + str(drop_index) + '.replace_' + str(replace_index) + '.seed_' + str(seed)
    examine_gradients = False
    task_type = 'binary'
    batch_size = 32

@ex.named_config
def mnist_binary_cropped():
    data_type = 'mnist_binary_cropped'
    data_privacy = 'all'
    model_type = 'logistic'
    n_epochs = 15
    init_path = 'models/' + model_type + '/mnist_binary_cropped_init.h5'
    replace_index = 'NA'
    seed = 1
    lr = 0.5
    cadence = 50
    if init_path is None:
        identifier = data_type + '/' + data_privacy + '/' + model_type + '/' + model_type + '_DIFFINIT.drop_' + str(drop_index) + '.replace_' + str(replace_index) + '.seed_' + str(seed)
    else:
        identifier = data_type + '/' + data_privacy + '/' + model_type + '/' + model_type + '.drop_' + str(drop_index) + '.replace_' + str(replace_index) + '.seed_' + str(seed)
    examine_gradients = False
    task_type = 'binary'
    batch_size = 32

@ex.named_config
def protein():
    data_type = 'protein'
    data_privacy = 'all'
    model_type = 'logistic'
    n_epochs = 3   # they test k = 5, 10 only (but note that 10 is a lot for us anyway...), but I think we don't need so many, and that lets me run more experiments
    init_path = 'models/' + model_type + '/' + data_type + '_' + model_type + '_init.h5'
    replace_index = 'NA'
    seed = 1
    lr = 0.01           # note that in their paper they use eta = 1.0/sqrt(m) which is about 0.0037, but this works worse
    cadence = 100
    if init_path is None:
        identifier = data_type + '/' + data_privacy + '/' + model_type + '/' + model_type + '_DIFFINIT.drop_' + str(drop_index) + '.replace_' + str(replace_index) + '.seed_' + str(seed)
    else:
        identifier = data_type + '/' + data_privacy + '/' + model_type + '/' + model_type + '.drop_' + str(drop_index) + '.replace_' + str(replace_index) + '.seed_' + str(seed)
    examine_gradients = False
    task_type = 'binary'
    batch_size = 50

@ex.named_config
def forest():
    data_type = 'forest'
    data_privacy = 'all'
    model_type = 'logistic'
    if data_privacy == 'all':
        n_epochs = 2   # they test k = 5, 10 only (10 is going to take a million years!)
    else:
        n_epochs = 4
    init_path = 'models/' + model_type + '/' + data_type + '_' + model_type + '_init.h5'
    replace_index = 'NA'
    seed = 1
    #lr = 0.01 #1
    #lr = 0.1 #2
    #lr = 0.5 #3
    lr = 1 #4           # wins from HP opt
    #lr = 0.75   # 7
    #lr = 2 #6
    #lr = 0.001     # 5
    cadence = 100
    if init_path is None:
        identifier = data_type + '/' + data_privacy + '/' + model_type + '/' + model_type + '_DIFFINIT.drop_' + str(drop_index) + '.replace_' + str(replace_index) + '.seed_' + str(seed)
    else:
        identifier = data_type + '/' + data_privacy + '/' + model_type + '/' + model_type + '.drop_' + str(drop_index) + '.replace_' + str(replace_index) + '.seed_' + str(seed)
    examine_gradients = False
    task_type = 'binary'
    batch_size = 50

@ex.named_config
def adult():
    data_type = 'adult'
    data_privacy = 'all'
    model_type = 'logistic'
    n_epochs = 5           # HP (with lr = 0.5, 5 is enough)
    init_path = 'models/' + model_type + '/' + data_type + '_' + model_type + '_init.h5'
    replace_index = 'NA'
    seed = 1
    #lr = 0.1        # 1
    #lr = 0.01       # 2 (bad)
    lr = 0.5         #3 (winner!)
    #lr = 1.0        #4 (bit high)
    cadence = 50
    if init_path is None:
        identifier = data_type + '/' + data_privacy + '/' + model_type + '/' + model_type + '_DIFFINIT.drop_' + str(drop_index) + '.replace_' + str(replace_index) + '.seed_' + str(seed)
    else:
        identifier = data_type + '/' + data_privacy + '/' + model_type + '/' + model_type + '.drop_' + str(drop_index) + '.replace_' + str(replace_index) + '.seed_' + str(seed)
    examine_gradients = False
    task_type = 'binary'
    batch_size = 32

@ex.named_config
def housing():
    data_type = 'housing'
    data_privacy = 'all'
    model_type = 'linear'
    n_epochs = 50
    init_path = 'models/' + model_type + '/' + data_type + '_' + model_type + '_init.h5'
    replace_index = 'NA'
    seed = 1
    lr = 0.01
    cadence = 5
    if init_path is None:
        identifier = data_type + '/' + data_privacy + '/' + model_type + '/' + model_type + '_DIFFINIT.drop_' + str(drop_index) + '.replace_' + str(replace_index) + '.seed_' + str(seed)
    else:
        identifier = data_type + '/' + data_privacy + '/' + model_type + '/' + model_type + '.drop_' + str(drop_index) + '.replace_' + str(replace_index) + '.seed_' + str(seed)
    examine_gradients = False
    task_type = 'regression'
    batch_size = 28

@ex.named_config
def housing_binary():
    data_type = 'housing_binary'
    data_privacy = 'all'
    model_type = 'logistic'
    n_epochs = 250
    init_path = 'models/' + model_type + '/' + data_type + '_' + model_type + '_init.h5'
    replace_index = 'NA'
    seed = 1
    lr = 0.1
    cadence = 10
    if init_path is None:
        identifier = data_type + '/' + data_privacy + '/' + model_type + '/' + model_type + '_DIFFINIT.drop_' + str(drop_index) + '.replace_' + str(replace_index) + '.seed_' + str(seed)
    else:
        identifier = data_type + '/' + data_privacy + '/' + model_type + '/' + model_type + '.drop_' + str(drop_index) + '.replace_' + str(replace_index) + '.seed_' + str(seed)
    examine_gradients = False
    task_type = 'binary'
    batch_size = 28

@ex.named_config
def mvn():
    data_type = 'mvn'
    data_privacy = 'all'
    model_type = 'linear'
    n_epochs = 2
    init_path = 'models/' + model_type + '/' + data_type + '_' + model_type + '_init.h5'
    replace_index = 'NA'
    seed = 1
    lr = 0.01
    cadence = 5
    if init_path is None:
        identifier = data_type + '/' + data_privacy + '/' + model_type + '/' + model_type + '_DIFFINIT.drop_' + str(drop_index) + '.replace_' + str(replace_index) + '.seed_' + str(seed)
    else:
        identifier = data_type + '/' + data_privacy + '/' + model_type + '/' + model_type + '.drop_' + str(drop_index) + '.replace_' + str(replace_index) + '.seed_' + str(seed)
    examine_gradients = True
    task_type = 'regression'
    batch_size = 32

@ex.named_config
def mvn_2d():
    data_type = 'mvn_2d'
    data_privacy = 'all'
    model_type = 'linear'
    n_epochs = 2
    init_path = 'models/' + model_type + '/' + data_type + '_' + model_type + '_init.h5'
    replace_index = 'NA'
    seed = 1
    lr = 0.01
    cadence = 10
    if init_path is None:
        identifier = data_type + '/' + data_privacy + '/' + model_type + '/' + model_type + '_DIFFINIT.drop_' + str(drop_index) + '.replace_' + str(replace_index) + '.seed_' + str(seed)
    else:
        identifier = data_type + '/' + data_privacy + '/' + model_type + '/' + model_type + '.drop_' + str(drop_index) + '.replace_' + str(replace_index) + '.seed_' + str(seed)
    examine_gradients = True
    task_type = 'regression'
    batch_size = 32

@ex.automain
def main(n_epochs, init_path, seed, replace_index, lr, identifier, data_type, data_privacy, model_type, examine_gradients, cadence, task_type, batch_size):
    print('\t\tRunning experiment on identifier:', identifier)
    # loading mnist, ok
    x_train, y_train, x_vali, y_vali, x_test, y_test = data_utils.load_data(data_type=data_type, replace_index=replace_index, data_privacy=data_privacy)

    # defining the model, ok
    model = model_utils.build_model(model_type=model_type, data_type=data_type, init_path=init_path)

    # training the model, ok
    model_utils.prep_for_training(model, seed, lr, task_type)

    model_utils.train_model(model, n_epochs, x_train, y_train, x_vali, y_vali, examine_gradients, label=identifier, batch_size=batch_size, cadence=cadence)
    del model
    return True
