#!/usr/bin/env ipython

import tensorflow as tf
import abc
from tensorflow import keras as K
import numpy as np
import pandas as pd

class Inspector(object):
    """
    """
    def __init__(self, model, X, y, label, n_minibatches=300, cadence=100):
        """
        """
        self.model = model
        self.X = X
        self.y = y
        self.n_minibatches = n_minibatches
        if self.n_minibatches == 0:
            print('[inspector] No minibatches will be sampled')
        self.minibatch_size = 32
        self.cadence = cadence
        self.counter = 0
        
        self.weights_file = open('traces/' + label + '.weights.csv', 'w')
        self.grads_file = open('traces/' + label + '.all_gradients.csv', 'w')
        self.loss_file = open('traces/' + label + '.loss.csv', 'w')
        print('[inspector] Saving weights and gradient information to traces/' + label + '.{loss/weights/all_gradients}.csv')

    def initialise_files(self):
        """
        """
        header = '#METADATA: Minibatch size is ' + str(self.minibatch_size) + '\n'
        for f in [self.weights_file, self.grads_file, self.loss_file]:
            f.write(header)
        n_parameters = len(self.model.get_weights(flat=True))
        print('[inspector] There are', n_parameters, 'weights in the model!')
        self.weights_file.write('t,' + ','.join(['#' + str(x) for x in range(n_parameters)]) + '\n')
        # gradients and loss are computed by minibatches
        self.grads_file.write('t,minibatch_id,' + ','.join(['#' + str(x) for x in range(n_parameters)]) + '\n')
        metrics = self.model.metric_names
        self.loss_file.write('t,minibatch_id,' + ','.join(metrics) + '\n')

    def on_batch_end(self, X_vali, y_vali):
        """
        Internal counter of number of iterations...
        """
        if self.counter % self.cadence == 0:
            self.inspect_model(X=self.X, y=self.y, minibatch_id='ALL', include_weights=True)
            self.inspect_model(X_vali, y_vali, minibatch_id='VALI', include_weights=False)
            N = self.X.shape[0]
            # now over the minibatches
            for s in range(self.n_minibatches):
                minibatch_idx = np.random.choice(N, self.minibatch_size, replace=False)
                X_batch = self.X[minibatch_idx]
                y_batch = self.y[minibatch_idx]
                self.inspect_model(X=X_batch, y=y_batch, minibatch_id=str(s), include_weights=False)
            #N = self.X.shape[0]
        self.counter += 1

    def inspect_model(self, X, y, minibatch_id, include_weights=False):
        # get metrics (most likely loss, but defined by keras)
        metrics = self.model.compute_metrics(X, y)
        # get gradients
        gradients = self.model.compute_gradients(X, y)
        # now write
        self.loss_file.write(str(self.counter) + ',' + minibatch_id + ',' + ','.join(map(str, metrics)) + '\n')
        self.grads_file.write(str(self.counter) + ',' + minibatch_id)
        for g in gradients:
            self.grads_file.write(',')
            self.grads_file.write(','.join(map(str, g.flatten())))
        self.grads_file.write('\n')
        if include_weights:
            weights = self.model.get_weights(flat=False)            # will flatten in this function while writing
            self.weights_file.write(str(self.counter))
            for w in weights:
                self.weights_file.write(',')
                self.weights_file.write(','.join(map(str, w.flatten())))
            self.weights_file.write('\n')

    def on_epoch_end(self):
        for f in [self.weights_file, self.grads_file, self.loss_file]:
            f.flush()

def build_model(architecture, input_size, output_size, task_type, hidden_size, init_path, **kwargs, t=None):
    """
    Wrapper around defining the model architecture
    """
    if architecture == 'mlp':
        model = feedforward(input_size, output_size, task_type, init_path, hidden_size, t)
    elif architecture == 'linear':
        model = linear(input_size, output_size, task_type, init_path, hidden_size, t)
    elif architecture == 'logistic':
        model = logistic(input_size, output_size, task_type, init_path, hidden_size, t)
    elif architecture == 'cnn':
        model = cnn(input_size, output_size, task_type, init_path, hidden_size, t)
    else:
        raise ValueError(architecture)
    return model

class model(object):
    """
    """
    def __init__(self, input_size, output_size, task_type, init_path, hidden_size, t):
        model = self.define_model(input_size, output_size, task_type, hidden_size)
        self.model = model
        self.init_path = init_path
        self.task_type = task_type
        self.override = None
        self.grads = None
        self.metrics = None
        self.hessian = None

        if init_path is None:
            print('[model_utils] WARNING: No init path provided, not loading weights!')
        else:
            try:
                self.load_weights(self.init_path, t)
            except FileNotFoundError:
                print('WARNING: Could not load weights from', self.init_path)

    @abc.abstractmethod
    def define_model(self, input_size, output_size, task_type, hidden_size):
        pass

    def load_weights(self, path, t=None):
        print('Loading weights from', path)
        if '.csv' in path:
            if t is None:
                t = 0
            self.load_and_set_weights_from_flat(path, t)
        else:
            assert '.h5' in path
            assert t is None
            self.model.load_weights(path)

    def save_weights(self, path):
        print('Saving weights to', path)
        self.model.save_weights(path)

    def get_weights(self, flat=False):
        weights = self.model.get_weights()
        if flat:
            weights_flat = [w.flatten() for  w in weights]
            weights = np.concatenate(weights_flat)
        return weights

    def unflatten_weights(self, vector):
        shapes = self.get_shape_of_weights()
        assert np.sum([np.product(x) for x in shapes]) == len(vector)
        list_of_weights = []
        indicator = 0           # where we are in the vector
        for shape_size in shapes:
            weight_size = np.product(shape_size)
            weight_values = vector[indicator:(indicator + weight_size)]
            this_weight = weight_values.reshape(shape_size)
            list_of_weights.append(this_weight)
            indicator = indicator + weight_size
        return list_of_weights

    def get_shape_of_weights(self):
        weights = self.model.get_weights()
        shapes = [w.shape for w in weights]
        return shapes

    def load_and_set_weights_from_flat(self, path, t):
        print('[model_utils] Loading flattened weights from', path, 'at time', t)
        # WARNING: THIS IS SPECIFIC TO HOW I'VE ENCODED THE WEIGHTS
        weights = pd.read_csv(path, skiprows=1)
        if t not in weights['t'].unique():
            print('ERROR: Timepoint', t, ' is not available in file', path, '(largest t is', weights['t'].max(), ')')
            raise ValueError(t)
        weights_at_t = weights.loc[weights['t'] == t, :].values[0, 1:]
        list_of_weights = self.unflatten_weights(weights_at_t)
        self.set_weights(list_of_weights)

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def predict(self, x):
        y = self.model.predict(x)
        return y

    def compute_hessian(self, X, y):
        """
        you REALLY do not want to compute this for a large model!!!
        """
        if self.hessian is None:
            self.hessian = tf.hessians(self.model.total_loss, self.model.weights)
        feed_dict = {self.model.input: X, self.model.targets[0]: y.reshape(-1, 1)}
        hessian = K.backend.get_session().run([self.hessian], feed_dict=feed_dict)
        return hessian

    def compute_gradients(self, X, y):
        """
        """
        if self.grads is None: 
            # the loss only exists after the model has been compiled!
            self.grads = tf.gradients(self.model.total_loss, self.model.weights)
        feed_dict={self.model.input: X, self.model.targets[0]: y.reshape(-1, 1)}
        #feed_dict.update(temp_weights_dict)
        gradients = K.backend.get_session().run([self.grads], feed_dict=feed_dict)[0]
        return gradients
  
    def define_metrics(self, metric_names, use_keras=True):
        metrics = [0]*len(metric_names)
        y_ph = self.model.targets[0]
        y_pred = self.model.output
        for i, metric in enumerate(metric_names):
            if metric == 'mse':
                if use_keras:
                    # not sure why the mean is necessary here
                    metrics[i] = tf.reduce_mean(K.metrics.mse(y_true=y_ph, y_pred=y_pred))
                else:
                    raise NotImplementedError
            elif metric == 'accuracy':
                if use_keras:
                    metrics[i] = tf.reduce_mean(K.metrics.sparse_categorical_accuracy(y_true=y_ph, y_pred=y_pred))
                else:
                    raise NotImplementedError
            elif metric == 'ce':
                if use_keras:
                    metrics[i] = tf.reduce_mean(K.metrics.sparse_categorical_crossentropy(y_true=y_ph, y_pred=y_pred))
                else:
                    raise NotImplementedError
            elif metric == 'binary_crossentropy':
                if use_keras:
                    metrics[i] = tf.reduce_mean(K.metrics.binary_crossentropy(y_true=y_ph, y_pred=y_pred))
                else:
                    raise NotImplementedError
            elif metric == 'binary_accuracy':
                if use_keras:
                    metrics[i] = tf.reduce_mean(K.metrics.binary_accuracy(y_true=y_ph, y_pred=y_pred))
                else:
                    raise NotImplementedError
            else:
                raise ValueError(metric)
        self.metric_names = metric_names
        self.metrics = metrics

    def compute_metrics(self, X, y):
        assert self.metrics is not None
        feed_dict = {self.model.input: X, self.model.targets[0]: y.reshape(-1, 1)}
        metrics = K.backend.get_session().run(self.metrics, feed_dict=feed_dict)
        return metrics


class linear(model):
    """
    Massive overkill doing this in Keras
    """
    
    def define_model(self, input_size, output_size=1, task_type='regression', hidden_size=None):
        if not output_size == 1:
            print('WARNING: output size for linear model is forced to 1')
        if not task_type == 'regression':
            print('WARNING: linear model has continuous output type')
        model = K.models.Sequential([K.layers.Dense(1, activation='linear', input_shape=(input_size,))])
        return model

        
class logistic(model):
    """
    """
    def define_model(self, input_size, output_size=1, task_type='binary', hidden_size=None):
        if not output_size == 1:
            print('WARNING: output size for logistic model is forced to 1')
        if not task_type == 'binary':
            print('WARNING: logistic has binary output type')
        model = K.models.Sequential([K.layers.Dense(1, activation='sigmoid', input_shape=(input_size,))])
        return model

class feedforward(model):
    """
    This model was taken from the Tensorflow MNIST tutorial!
    """
    def define_model(self, input_size=(28, 28), output_size=10, task_type='classification', hidden_size=512):
        if type(input_size) == int:
            # no flatten required if input is a vector
            layers = [K.layers.Dense(hidden_size, input_dim=input_size, activation='relu')]
        else:
            layers = [K.layers.Flatten(input_shape=input_size),
                    K.layers.Dense(hidden_size, activation='relu')]
        # shared piece
        layers.append(K.layers.Dropout(rate=0.2))
        # output-size-dependent piece
        if task_type == 'classification':
            # use a softmax
            layers.append(K.layers.Dense(output_size, activation='softmax'))
        elif task_type == 'binary':
            # sigmoid
            layers.append(K.layers.Dense(output_size, activation='sigmoid'))
        elif task_type == 'regression':
            # linear
            layers.append(K.layers.Dense(output_size, activation='linear'))
        else:
            raise ValueError(task_type)
        model = K.models.Sequential(layers)
        
        return model

class cnn(model):
    """
    Trying to replicate the cuda-convnet model referenced in the Hardt paper
    "three convolutional layers each followed by a pooling operation"
    no dropout
    no mention of any other HPs in that paper from what I can tell
    """
    def define_model(self, input_size=(32, 32, 3), output_size=10, task_type='classification', hidden_size=512):
        # input validation
        if len(input_size) < 1:
            print('ERROR: CNN is not designed to take flat inputs!')
            raise ValueError(input_size)
        elif len(input_size) == 2:
            print('WARNING: Assuming a single channel provided') # e.g. for non-flat MNIST, if I do that
            input_size = (input_size[0], input_size[1], 1)
        elif len(input_size) == 3:
            pass
        else:
            raise ValueError(input_size)
        layers = [K.layers.Conv2D(8, (3, 3), padding='same', input_shape=input_size, activation='relu'),
                K.layers.MaxPooling2D(pool_size=(2, 2)),
                K.layers.Conv2D(8, (2, 2), activation='relu'),
                K.layers.MaxPooling2D(pool_size=(2, 2)),
                K.layers.Conv2D(8, (2, 2), padding='same', activation='relu'),
                K.layers.MaxPooling2D(pool_size=(2, 2)),
                K.layers.Flatten(),
                K.layers.Dense(hidden_size, activation='relu')]
        if task_type == 'classification':
            layers.append(K.layers.Dense(output_size, activation='softmax'))
        elif task_type == 'binary':
            # sigmoid
            layers.append(K.layers.Dense(output_size, activation='sigmoid'))
        elif task_type == 'regression':
            # linear
            layers.append(K.layers.Dense(output_size, activation='linear'))
        else:
            raise ValueError(task_type)
        model = K.models.Sequential(layers)
        
        return model

def prep_for_training(model_object, seed, lr, task_type):
    """
    """
    tf.random.set_random_seed(seed)         # not sure this one is actually necesary
    np.random.seed(seed)
    #sgd = K.optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    sgd = K.optimizers.SGD(lr=lr, decay=0, momentum=0, nesterov=False)
    # note: the tutorial I took the model from uses Adam, but SGD + momentum works okay too
    if task_type == 'classification':
        loss = 'sparse_categorical_crossentropy'
        metrics = ['ce', 'accuracy']
    elif task_type == 'regression':
        loss = 'mse'
        metrics = ['mse']
    elif task_type == 'binary':
        loss = 'binary_crossentropy'
        metrics = ['binary_crossentropy', 'binary_accuracy']
    model_object.model.compile(optimizer=sgd,
        loss=loss)
        #metrics=metrics)
    # moving metrics to my part
    model_object.define_metrics(metrics)
    #model_object.save_weights(path=model_object.init_path)
    #K.backend.get_session().run(tf.global_variables_initializer())
    return True

def train_model(model_object, n_epochs, x_train, y_train, x_vali, y_vali, 
        examine_gradients, label, batch_size=32, cadence=100):
    if examine_gradients:
        inspector = Inspector(model_object, x_train, y_train, label=label, cadence=cadence)
        #my_callback.initialise(model_object, x_train, y_train, label=label)
    else:
        inspector = Inspector(model_object, x_train, y_train, label=label, cadence=cadence, n_minibatches=0)
    #callbacks.append(my_callback)

    # breaking out of the fit method, back to old skool
    N = x_train.shape[0]
    if not N % batch_size == 0:
        print('[model utils] WARNING: Training set size is not multiple of batch size - some data will be missed every epoch!')
    n_batches = N // batch_size

    inspector.initialise_files()
    for e in range(n_epochs):
        if e % 100 == 0:
            print('epoch:', e)
        shuf = np.random.permutation(N)
        x_train = x_train[shuf]
        y_train = y_train[shuf]
        # at the very beginning!
        inspector.on_batch_end(x_vali, y_vali)
        for batch_idx in range(n_batches):
            x_batch = x_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
            y_batch = y_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
            _ = model_object.model.train_on_batch(x_batch, y_batch)
            inspector.on_batch_end(x_vali, y_vali)
        # at the end of the epoch, test on everything
        inspector.on_epoch_end()
    K.backend.clear_session()
    #tf.reset_default_graph()
    return True

def add_gaussian_noise(weights, sigma):
    n_weights = len(weights)
    noise = np.random.normal(loc=0, scale=sigma, size=n_weights)
    assert noise.shape == weights.shape
    noised_weights = weights + noise
    return noised_weights

### --- VESTIGIAL --- ###
def estimate_sensitivity(weights_with_different_data, reference_idx=0, return_all=False, identifiers=None):
    """
    """
    n_runs = weights_with_different_data.shape[0]
    print('Estimating sensitivity over', n_runs, 'different datasets')
    print('Comparing to reference, indexed at', reference_idx)
   
    norms = np.zeros(n_runs)
    for i in range(n_runs):
        if i == reference_idx:
            distance = np.nan
        elif (identifiers is not None) and (identifiers[i] == identifiers[reference_idx]):
            distance = np.nan
        else:
            distance = np.linalg.norm(weights_with_different_data[i] - weights_with_different_data[reference_idx])
            if np.abs(distance) < 1e-5:
                if identifiers is None:
                    print('WARNING: two identical sets of weights? ... run', i, 'and', reference_idx)
                else:
                    print('WARNING: two identical sets of weights? ... models', identifiers[i], 'and', identifiers[reference_idx])
        norms[i] = distance
    norms = norms[~np.isnan(norms)]
    if return_all:
        return norms
    else:
        sensitivity = np.max(norms)
        return sensitivity
