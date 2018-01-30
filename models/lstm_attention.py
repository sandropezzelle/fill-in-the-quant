# --------------------------------------------------------
# --------------------------------------------------------
# code by Shane Steinert-Threlkeld (ILLC, UvA)
# and Sandro Pezzelle (CIMeC, Unitn)
# except where otherwise noted
# January, 2018
# --------------------------------------------------------
# --------------------------------------------------------

from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM
# , Bidirectional, Activation, Input, Flatten
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.engine.topology import Layer
from keras import constraints
from keras import initializers
from keras import regularizers

import util


# TODO: command line arguments for path to vectors, path to text data
def run_trial(params, fixed_params, data_path, embedding_file):

    Xs, Ys, word_index = util.generate_datasets(data_path, params['setting'],
                                                ['train', 'test', 'val'],
                                                fixed_params['quantifiers'],
                                                fixed_params['max_num_words'],
                                                params['max_seq_len'],
                                                params['punct'])

    # embedding_matrix = util.get_embedding_matrix(embedding_file, word_index,
                                                 # fixed_params['embedding_dim'])
    embedding_matrix = np.zeros((len(word_index) + 1,
                                 params['embedding_dim']))

    model = build_model(params, embedding_matrix)

    print('Model built. Time to train!')

    checkpoint = ModelCheckpoint(fixed_params['output_path'],
                                 monitor='val_loss',
                                 verbose=1, save_best_only=True, mode='min')
    callback_list = [checkpoint]
    # we save the weights of the model for which the validation loss is lowest!

    model.fit(Xs['train'], Ys['train'],
              batch_size=params['batch_size'],
              epochs=params['num_epochs'],
              validation_data=[Xs['val'], Ys['val']],
              callbacks=callback_list)


def build_model(params, embedding_matrix):

    embedding_layer = Embedding(len(embedding_matrix),
                                params['embedding_dim'],
                                weights=[embedding_matrix],
                                input_length=params['max_seq_len'],
                                mask_zero=True,  # MASK the padding!
                                trainable=False)  # should we try True?

    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(params['hidden_units'], return_sequences=True))
    model.add(Dropout(params['dropout']))
    if params['context']:
        model.add(AttentionWithContext())
    else:
        model.add(Attention())
    model.add(Dense(params['num_classes'], activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])

    return model


# ------------------------
# The code for Attention and AttentionWithContext layers comes from @cbaziotis
# See, respectively: 
# - https://gist.github.com/cbaziotis/6428df359af27d58078ca5ed9792bd6d
# - https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2
# ------------------------

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class Attention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.


        Note: The layer has been tested with Keras 2.0.6

        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...

        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        eij = dot_product(x, self.W)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training
        # the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small
        # positive number epsilon to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.

    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.

    Note: The layer has been tested with Keras 2.0.6

    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training
        # the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small
        # positive number epsilon to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


if __name__ == '__main__':

    params = {
        'punct': True,
        'setting': util.SettingsValues.starget,
        'max_seq_len': 50,
        'hidden_units': 128,
        'num_classes': 9,
        'batch_size': 64,
        'num_epochs': 30,
        'dropout': 1.0,
        'optimizer': 'nadam',
        'embedding_dim': 300,
        'context': True
    }

    fixed_params = {
        'model_name': 'lstm-attention{}{}'.format(
            '-context' if params['context'] else '',
            '-punct' if params['punct'] else ''),
        'max_num_words': 50000,
        'quantifiers': ['none of ', 'a few of ', 'few of ', 'some of ',
                        'many of ', 'most of ', 'more than half of ',
                        'almost all of ', 'all of '],
        'output_path': '/tmp/keras'
    }

    run_trial(params, fixed_params, '../data/',
              '/Users/shanest/Documents/GoogleNews-vectors-negative300.txt')
