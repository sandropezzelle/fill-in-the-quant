# --------------------------------------------------------
# --------------------------------------------------------
# code by Shane Steinert-Threlkeld (ILLC, UvA)
# and Sandro Pezzelle (CIMeC, Unitn)
# except where otherwise noted
# January, 2018
# --------------------------------------------------------
# --------------------------------------------------------

from __future__ import print_function
import argparse

import numpy as np
from keras.models import Sequential
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint

import util
from lstm_attention import build_attention_model, build_attention_context_model
from ntm import build_ntm_model

""" Example usage:

    python lstm_attention.py \
            --model ntm \
            --data /path/to/data/ \
            --vectors /path/to/vectors.txt \
            --out_path /path/to/output

    If '--context' is omitted, Attention() layer will be used instead of
    AttentionWithContext.
"""


# TODO: command line arguments for path to vectors, path to text data
def run_trial(model_fn, params, fixed_params, data_path, embedding_file):

    print(params['setting'])
    Xs, Ys, word_index = util.generate_datasets(data_path, params['setting'],
                                                ['train', 'test', 'val'],
                                                fixed_params['quantifiers'],
                                                fixed_params['max_num_words'],
                                                params['max_seq_len'],
                                                params['punct'])

    embedding_matrix = util.get_embedding_matrix(embedding_file, word_index,
                                                 params['embedding_dim'])
    embedding_model = Sequential()
    embedding_model.add(Embedding(len(embedding_matrix),
                                  params['embedding_dim'],
                                  weights=[embedding_matrix],
                                  input_length=params['max_seq_len'],
                                  mask_zero=True,
                                  trainable=False))
    embedding_model.compile(params['optimizer'], 'binary_crossentropy')

    X_vectors = {k: embedding_model.predict(Xs[k],
                                            batch_size=len(Xs[k])) for k in Xs}

    model = model_fn(params)

    print('Model built. Time to train!')

    checkpoint = ModelCheckpoint(fixed_params['output_path'],
                                 monitor='val_loss',
                                 verbose=1, save_best_only=True, mode='min')
    callback_list = [checkpoint]
    # we save the weights of the model for which the validation loss is lowest!

    model.fit(X_vectors['train'],  # Xs['train'],
              Ys['train'],
              batch_size=params['batch_size'],
              epochs=params['num_epochs'],
              validation_data=[X_vectors['val'], Ys['val']],
              callbacks=callback_list)


if __name__ == '__main__':

    # TODO: check for correct arguments, e.g. for attention vs ntm

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='name of model function',
                        type=str, required=True)
    parser.add_argument('--vectors', help='file with vector embedding',
                        type=str, required=True)
    parser.add_argument('--data', help='path to data files', type=str,
                        required=True)
    parser.add_argument('--out_path', help='path to output', type=str,
                        default='/tmp/')
    parser.add_argument('--context',
                        help='whether to use AttentionWithContext',
                        action='store_true', default=False)
    parser.add_argument('--setting',
                        help='setting for data type',
                        type=str, default='starget')
    args = parser.parse_args()

    # TODO: improve the division of labor between args and params/fixed_params
    # e.g. make everything an arg, with default values?
    shared_params = {
        'punct': True,
        'setting': args.setting,
        'max_seq_len': 50,
        'num_classes': 9,
        'batch_size': 64,
        'num_epochs': 30,
        'optimizer': 'nadam',
        'embedding_dim': 300
    }

    attention_params = {
        'name': 'lstm-attention',
        'hidden_units': 128,
        'dropout': 0.25,
    }

    attention_context_params = {
        'name': 'lstm-attention-context',
        'hidden_units': 128,
        'dropout': 0.25,
    }

    ntm_params = {
        'name': 'ntm',
        'n_slots': 128,
        'm_depth': 20,
        'read_heads': 1,
        'write_heads': 1,
        'shift_range': 3,
        'activation': 'softmax',  # for classification
        'controller': 'ffnn'
    }

    if args.model == 'att' or args.model == 'att_con':
        params = util.merge_dicts(shared_params, attention_params)
    elif args.model == 'ntm':
        params = util.merge_dicts(shared_params, ntm_params)
    else:
        raise ValueError("--model must be one of att, att_con, ntm")

    fixed_params = {
        'model_name': '{}{}'.format(
            params['name'],
            '-punct' if params['punct'] else ''),
        'max_num_words': 50000,
        'quantifiers': ['none of ', 'a few of ', 'few of ', 'some of ',
                        'many of ', 'most of ', 'more than half of ',
                        'almost all of ', 'all of '],
        'output_path': args.out_path
    }

    model_fn_dict = {
        'att': build_attention_model,
        'att_con': build_attention_context_model,
        'ntm': build_ntm_model
    }
    model_fn = model_fn_dict[args.model]

    run_trial(model_fn, params, fixed_params, args.data, args.vectors)
