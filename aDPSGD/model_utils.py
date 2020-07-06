#!/usr/bin/env ipython

import abc
import numpy as np
import pandas as pd
import tensorflow as tf
import ipdb
from tensorflow import keras as K
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D


class Logger(object):
    """
    """
    def __init__(self, model: 'Model', path_stub: str, cadence: int,
                 X_train: np.ndarray, y_train: np.ndarray, X_vali: np.ndarray, y_vali: np.ndarray) -> None:
        self.model = model
        self.path_stub = path_stub
        self.logging_cadence = cadence
        self.logging_counter = tf.Variable(initial_value=0, name='logging_counter', trainable=False, dtype=tf.int32)
        self.X_train = X_train
        self.X_vali = X_vali
        self.y_train = y_train
        self.y_vali = y_vali
        self.metric_names = self.model.metric_names

        # initialise some stuff now
        self.initialise_log_files()
        self.build_metrics()

    def build_metrics(self) -> None:
        """
        These are defined in the logger and not the model because
        if I define them in the model, they somehow get added to what keras
        thinks are the 'weights' of the model.
        """
        self.metric_functions = define_metric_functions(self.metric_names)

    def initialise_log_files(self) -> None:
        # define paths (we will need this for tensorflow's print function later)
        self.weights_file_path = f'{self.path_stub}.weights.csv'
        self.grads_file_path = f'{self.path_stub}.grads.csv'
        self.loss_file_path = f'{self.path_stub}.loss.csv'
        print(f'[logging] Saving information with identifier {self.path_stub}')
        # open them up
        self.weights_file = open(self.weights_file_path, 'w')
        self.grads_file = open(self.grads_file_path, 'w')
        self.loss_file = open(self.loss_file_path, 'w')
        # list of files makes flushing easier
        self.log_files = [self.weights_file, self.grads_file, self.loss_file]
        # store headers (TODO could do this while tidying the files up later)
        n_parameters = len(self.model.get_weights(flat=True))
        print(f'[logging] There are {n_parameters} weights in the model!')
        self.weights_file.write('t,' + ','.join(['#' + str(x) for x in range(n_parameters)]) + '\n')
        self.grads_file.write('t,minibatch_id,' + ','.join(['#' + str(x) for x in range(n_parameters)]) + '\n')
        self.loss_file.write('t,minibatch_id,' + ','.join(self.metric_names) + '\n')
        for f in self.log_files:
            f.flush()

    def finalise_log_files(self) -> None:
        """
        tf.print uglies up the files so we need to undo that at the end of training
        """
        for f in self.log_files:
            f.flush()
            f.close()
        # loss is easy because there should be commas
        loss = pd.read_csv(self.loss_file_path)
        loss.replace('[\[\] ]', '', regex=True, inplace=True)
        loss.to_csv(self.loss_file_path, index=False)
        del loss
        # weights has both commas and (single) spaces as delimeters
        weights = pd.read_csv(self.weights_file_path, sep='[, ]', engine='python')
        weights.replace('[\[\] ]', '', regex=True, inplace=True)
        weights.to_csv(self.weights_file_path, index=False)
        del weights
        # gradients
        # TODO should be the same as weights though

    @tf.function
    def log_model(self, X: np.ndarray, y: np.ndarray, minibatch_id: str,
                  save_weights: bool = False, save_gradients: bool = False) -> None:
        # --- metrics --- #
        metric_results = self.model.compute_metrics(X, y, metric_functions=self.metric_functions)
        tf.print(self.logging_counter, output_stream='file:///' + self.loss_file_path, end=',')
        tf.print(minibatch_id, output_stream='file:///' + self.loss_file_path, end=',')
        tf.print(metric_results, output_stream='file:///' + self.loss_file_path, summarize=-1, end='\n')
        # --- weights --- #
        if save_weights:
            weights = self.model.get_weights(flat=True)
            tf.print(self.logging_counter, output_stream='file:///' + self.weights_file_path, end=',')
            tf.print(weights, output_stream='file:///' + self.weights_file_path, summarize=-1, end='\n')
        # --- gradients --- #
        if save_gradients:
            raise NotImplementedError
            # TODO get this to work
            # gradients = self.model.compute_gradients(X, y)
            # self.grads_file.write(str(self.counter) + ',' + minibatch_id)
            # for g in gradients:
            #    self.grads_file.write(',')
            #    self.grads_file.write(','.join(map(str, g.flatten())))
            # self.grads_file.write('\n')

    def on_batch_end(self) -> None:
        if self.logging_counter % self.logging_cadence == 0:
            self.log_model(X=self.X_train, y=self.y_train, minibatch_id='ALL', save_weights=True)
            self.log_model(X=self.X_vali, y=self.y_vali, minibatch_id='VALI', save_weights=False)
            # N = self.X_train.shape[0]
            # now over the minibatches
            # TODO do this
#            for s in range(self.n_minibatches):
#                minibatch_idx = np.random.choice(N, self.minibatch_size, replace=False)
#                X_batch = self.X[minibatch_idx]
#                y_batch = self.y[minibatch_idx]
#                self.log_model(X=X_batch, y=y_batch, minibatch_id=str(s), save_weights=False)
        self.logging_counter.assign_add(1)

    def on_epoch_end(self) -> None:
        for f in self.log_files:
            f.flush()

    def on_training_end(self) -> None:
        self.finalise_log_files()
        print('[logging] Log files finalised')


def build_model(architecture: str, input_size: int, output_size: int,
                task_type: str, hidden_size: int, init_path, t=None, **kwargs) -> 'Model':
    """
    Wrapper around defining the model architecture
    """
    if architecture == 'mlp':
        model = Feedforward(input_size, output_size, task_type, init_path, hidden_size, t)
    elif architecture == 'linear':
        model = Linear(input_size, init_path, t)
    elif architecture == 'logistic':
        model = Logistic(input_size, init_path, t)
    elif architecture == 'cnn':
        model = CNN(input_size=input_size, output_size=output_size,
                    task_type=task_type, init_path=init_path,
                    hidden_size=hidden_size, t=t)
    else:
        raise ValueError(architecture)
    return model


class Model(K.Sequential):
    """
    """
    def __init__(self, input_size: int, init_path: str, t: int) -> None:
        super(Model, self).__init__()
        self.init_t = t
        self.input_size = input_size
        self.init_path = init_path
        self.grads = None
        self.hessian = None

    def build(self) -> None:
        self.define_layers()
        if self.init_path is None:
            print('[model_utils] WARNING: No init path provided, not loading weights!')
        else:
            try:
                self.load_weights(self.init_path, self.init_t)
            except FileNotFoundError:
                print(f'WARNING: Could not load weights from {self.init_path}')

    @abc.abstractmethod
    def define_layers(self) -> None:
        pass

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, batch_size: int,
            epochs: int, logger: 'Logger' = None) -> None:
        N = x_train.shape[0]
        if not N % batch_size == 0:
            print('WARNING: Training set size is not multiple of batch size - some data will be missed every epoch!')
        n_batches = N // batch_size
        if logger is not None:
            # just once at the start
            logger.on_batch_end()
        for e in range(epochs):
            if e % 100 == 0:
                print(f'epoch: {e}')
            shuf = np.random.permutation(N)
            x_train = x_train[shuf]
            y_train = y_train[shuf]
            for batch_idx in range(n_batches):
                x_batch = x_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
                y_batch = y_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
                _ = self.train_on_batch(x_batch, y_batch)
                if logger is not None:
                    logger.on_batch_end()
            if logger is not None:
                logger.on_epoch_end()
        if logger is not None:
            logger.on_training_end()

    def load_weights(self, path: str, t: int = None) -> None:
        print(f'Loading weights from {path}')
        if not path.exists():
            print(f'WARNING: Weights path  {path} does not exist, can\'t load weights!')
            return
        if path.suffix == '.csv':
            if t is None:
                t = 0
            self.load_and_set_weights_from_flat(path, t)
        else:
            assert path.suffix == '.h5'
            assert t is None
            super(Model, self).load_weights(path.as_posix())

    def save_weights(self, path: str) -> None:
        print(f'Saving weights to {path}')
        super(Model, self).save_weights(path.as_posix())

    @tf.function
    def get_weights(self, flat: bool = False) -> tf.Tensor:
        weights = self.weights
        if flat:
            weights = tf.squeeze(tf.concat([tf.reshape(w, [-1, 1]) for w in weights], axis=0))
        return weights

    def unflatten_weights(self, vector):
        shapes = self.get_shape_of_weights()
        assert np.sum([np.product(x) for x in shapes]) == len(vector)
        list_of_weights = []
        indicator = 0           # where we are in the vector
        for shape_size in shapes:
            weight_size = np.product(shape_size)
            weight_values = vector[indicator:(indicator + weight_size)]
            this_weight = tf.reshape(weight_values, shape=shape_size)
            list_of_weights.append(this_weight)
            indicator = indicator + weight_size
        return list_of_weights

    def get_shape_of_weights(self):
        weights = self.get_weights(flat=False)
        shapes = [w.shape for w in weights]
        return shapes

    def load_and_set_weights_from_flat(self, path, t):
        print(f'[model_utils] Loading flattened weights from {path} at time {t}')
        weights = pd.read_csv(path)
        if t not in weights['t'].unique():
            print(f'ERROR: Timepoint {t} is not available in file {path} (largest t is {weights["t"].max()}')
            raise ValueError(t)
        weights_at_t = weights.loc[weights['t'] == t, :].values[0, 1:]
        list_of_weights = self.unflatten_weights(weights_at_t)
        self.set_weights(list_of_weights)

    @tf.function
    def compute_metrics(self, X, y, metric_functions):
        predictions = self(X)
        results = []
        for metric in metric_functions:
            results.append(metric(y, predictions))
        return results

    def compute_hessian(self, X, y):
        """
        you REALLY do not want to compute this for a large model!!!
        """
        raise NotImplementedError
        if self.hessian is None:
            self.hessian = tf.hessians(ys=self.model.total_loss, xs=self.model.weights)
        feed_dict = {self.model.input: X, self.model._targets[0]: y.reshape(-1, 1)}
        hessian = K.backend.get_session().run([self.hessian], feed_dict=feed_dict)
        return hessian

    def compute_gradients(self, X, y):
        """
        """
        raise NotImplementedError
        if self.grads is None:
            # the loss only exists after the model has been compiled!
            self.grads = tf.gradients(ys=self.model.total_loss, xs=self.model.weights)
        feed_dict = {self.model.input: X, self.model._targets[0]: y.reshape(-1, 1)}
        # feed_dict.update(temp_weights_dict)
        gradients = K.backend.get_session().run([self.grads], feed_dict=feed_dict)[0]
        return gradients


class Linear(Model):
    """
    Massive overkill doing this in Keras
    """
    def __init__(self, input_size, init_path, t=0):
        super(Linear, self).__init__(input_size=input_size, init_path=init_path, t=t)
        self.build()

    def define_layers(self):
        self.add(Dense(1, activation='linear', input_shape=(self.input_size, )))


class Logistic(Model):
    """
    """
    def __init__(self, input_size, init_path, t=0):
        super(Logistic, self).__init__(input_size=input_size, init_path=init_path, t=t)
        self.build()

    def define_layers(self):
        self.add(Dense(1, activation='sigmoid', input_shape=(self.input_size,)))


class Feedforward(Model):
    """
    This model was taken from the Tensorflow MNIST tutorial!
    """
    def __init__(self, input_size, output_size, task_type, init_path, hidden_size, t):
        super(Feedforward, self).__init__(input_size=input_size, init_path=init_path, t=t)
        self.output_size = output_size
        self.task_type = task_type
        self.hidden_size = hidden_size
        self.build()

    def define_layers(self):
        if type(self.input_size) == int:
            # no flatten required if input is a vector
            self.add(Dense(self.hidden_size, input_dim=self.input_size, activation='relu'))
        else:
            self.add(Flatten(input_shape=self.input_size))
            self.add(Dense(self.hidden_size, activation='relu'))
        # shared piece
        self.add(Dropout(rate=0.2))
        # output-size-dependent piece
        if self.task_type == 'classification':
            activation = 'softmax'
        elif self.task_type == 'binary':
            activation = 'sigmoid'
        elif self.task_type == 'regression':
            activation = 'linear'
        else:
            raise ValueError(self.task_type)
        self.add(Dense(self.output_size, activation=activation))


class CNN(Model):
    """
    Trying to replicate the cuda-convnet model referenced in the Hardt paper
    "three convolutional layers each followed by a pooling operation"
    no dropout
    no mention of any other HPs in that paper from what I can tell
    """
    def __init__(self, input_size, output_size, task_type, init_path, hidden_size, t):
        super(CNN, self).__init__(input_size=input_size, init_path=init_path, t=t)
        self.output_size = output_size
        self.task_type = task_type
        self.hidden_size = hidden_size

        # input validation
        if len(self.input_size) < 1:
            print('ERROR: CNN is not designed to take flat inputs!')
            raise ValueError(self.input_size)
        elif len(self.input_size) == 2:
            print('WARNING: Assuming a single channel provided')
            self.input_size = (self.input_size[0], self.input_size[1], 1)
        elif len(self.input_size) == 3:
            pass
        else:
            raise ValueError(self.input_size)
        self.build()

    def define_layers(self):
        self.add(Conv2D(filters=8, kernel_size=3, padding='same',
                        input_shape=self.input_size, activation='relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Conv2D(filters=8, kernel_size=(2, 2), activation='relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Conv2D(filters=8, kernel_size=(2, 2), padding='same', activation='relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Flatten())
        self.add(Dense(self.hidden_size, activation='relu'))
        if self.task_type == 'classification':
            activation = 'softmax'
        elif self.task_type == 'binary':
            activation = 'sigmoid'
        elif self.task_type == 'regression':
            activation = 'linear'
        else:
            raise ValueError(self.task_type)
        self.add(Dense(self.output_size, activation=activation))


def prep_for_training(model: 'Model', seed: int, optimizer_settings: dict, task_type: str) -> None:
    # set seeds
    tf.random.set_seed(seed)
    np.random.seed(seed)
    # set up the optimizer
    if optimizer_settings['name'] == 'SGD':
        lr = optimizer_settings['learning_rate']
        sgd = K.optimizers.SGD(lr=lr, decay=0, momentum=0, nesterov=False)
    else:
        raise NotImplementedError('Only SGD is implemented currently')
    # set up the loss
    if task_type == 'classification':
        loss = 'sparse_categorical_crossentropy'
        metric_names = ['ce', 'accuracy']
    elif task_type == 'regression':
        loss = 'mse'
        metric_names = ['mse']
    elif task_type == 'binary':
        loss = 'binary_crossentropy'
        metric_names = ['binary_crossentropy', 'binary_accuracy']
    else:
        raise ValueError(task_type)
    model.compile(optimizer=sgd,
                  loss=loss)
    model.metric_names = metric_names

    if model.init_path is None:
        print('Not saving weights as no init path given')
    else:
        if model.init_path.exists():
            print('Not saving weights as path already exists')
        else:
            model.save_weights(path=model.init_path)
            print(f'Saved weights to {model.init_path}')
    return


def train_model(model: 'Model', training_cfg: dict, logging_cfg: dict,
                x_train: np.ndarray, y_train: np.ndarray,
                x_vali: np.ndarray, y_vali: np.ndarray,
                path_stub: str) -> None:

    experiment_logger = Logger(model, path_stub,
                               cadence=logging_cfg['cadence'],
                               X_train=x_train, y_train=y_train,
                               X_vali=x_vali, y_vali=y_vali)

    model.fit(x_train=x_train, y_train=y_train,
              batch_size=training_cfg['batch_size'],
              epochs=training_cfg['n_epochs'],
              logger=experiment_logger)
    return


def define_metric_functions(metric_names):
    metric_functions = [0]*len(metric_names)
    for i, metric in enumerate(metric_names):
        if metric == 'mse':
            metric_functions[i] = K.metrics.MeanSquareError()
        elif metric == 'accuracy':
            metric_functions[i] = K.metrics.SparseCategoricalAccuracy()
        elif metric == 'ce':
            metric_functions[i] = K.metrics.SparseCategoricalCrossentropy()
        elif metric == 'binary_crossentropy':
            metric_functions[i] = K.metrics.BinaryCrossentropy()
        elif metric == 'binary_accuracy':
            metric_functions[i] = K.metrics.BinaryAccuracy(threshold=0.5)
        else:
            raise ValueError(metric)
    return metric_functions
