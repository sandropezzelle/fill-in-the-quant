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
import sklearn
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint

import util
from lstm_attention import build_attention_model, build_attention_context_model
from ntm import build_ntm_model

""" Example usage:

    python run_experiment.py \
            --model att \
            --data /path/to/data/ \
            --vectors /path/to/vectors.txt \
            --out_path /path/to/output
"""


# TODO: command line arguments for path to vectors, path to text data
def run_trial(model_fn, params, fixed_params, data_path, embedding_file,
              train=True):

    print(params['setting'])
    Xs, Ys, word_index = util.generate_datasets(data_path, params['setting'],
                                                ['train', 'test', 'val'],
                                                fixed_params['quantifiers'],
                                                fixed_params['max_num_words'],
                                                params['max_seq_len'],
                                                params['punct'])

    embedding_matrix = util.get_embedding_matrix(embedding_file, word_index,
                                                 params['embedding_dim'])
    # embedding_matrix = np.zeros((len(word_index)+1, params['embedding_dim']))
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

    # NOTE: this is a hack.  The implementation of NTM depends on batch_size;
    # if that does not evenly divide the size of one epoch, the last batch will
    # be a different size and break everything.  For now, we make sure that
    # batch_size evenly divides the size of the training data _and_ the size of
    # the validation data.  Eventually, we should fix the NTM.
    if params['name'] == 'ntm':
        train_data_size = len(X_vectors['train'])
        val_data_size = len(X_vectors['val'])
        batch_size = params['batch_size']
        for n in range(batch_size):
            new = batch_size - n
            if train_data_size % new == 0 and val_data_size % new == 0:
                params['batch_size'] = new
                print('NTM mode, found batch size {}'.format(new))
                break

    model = model_fn(params)

    print('Model built.')

    if train:
        print('Time to train!')

        checkpoint = ModelCheckpoint(fixed_params['output_path'],
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='min')
        callback_list = [checkpoint]
        # we save the weights of the model for which the validation loss is lowest!

        model.fit(X_vectors['train'],  # Xs['train'],
                  Ys['train'],
                  shuffle=True,
                  batch_size=params['batch_size'],
                  epochs=params['num_epochs'],
                  validation_data=[X_vectors['val'], Ys['val']],
                  callbacks=callback_list)
        print('Training done.')

    print('Evaluating best model.')
    model.load_weights(fixed_params['output_path'])
    # need to feed proper batch size even for evaluation of NTM
    for dataset in ['test', 'val']:
        print(dataset)
        evaluation = model.evaluate(X_vectors[dataset], Ys[dataset],
                                    batch_size=params['batch_size'])
        print(evaluation)
        print('Confusion matrix.')
        y_pred = model.predict_classes(X_vectors[dataset])
        y_true = np.argmax(Ys[dataset], axis=1)
        print(confusion_matrix(y_pred, y_true))


if __name__ == '__main__':

    # TODO: check for correct arguments, e.g. for attention vs ntm
    # TODO: optimizer as arg?

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='name of model function',
                        type=str, required=True)
    parser.add_argument('--vectors', help='file with vector embedding',
                        type=str, required=True)
    parser.add_argument('--data', help='path to data files', type=str,
                        required=True)
    parser.add_argument('--out_path', help='path to output', type=str,
                        default='/tmp/')
    parser.add_argument('--setting',
                        help='setting for data type',
                        type=str, default='starget')
    parser.add_argument('--batch_size', help='size of each batch',
                        type=int, default=64)
    parser.add_argument('--hidden_units', help='size of hidden layer in LSTM',
                        type=int, default=128)
    parser.add_argument('--dropout', help='amount of dropout',
                        type=float, default=0.25)
    parser.add_argument('--optimizer', help='which optimizer to use',
                        type=str, default='nadam')
    parser.add_argument('--eval', help='evaluate a model instead of train',
                        default=False, action='store_true')
    args = parser.parse_args()

    # TODO: improve the division of labor between args and params/fixed_params
    # e.g. make everything an arg, with default values?
    shared_params = {
        'punct': True,
        'setting': args.setting,
        'max_seq_len': (50 if args.setting == 'starget' else 150),
        'num_classes': 9,
        'batch_size': args.batch_size,
        'num_epochs': 30,
        'optimizer': args.optimizer,
        'embedding_dim': 300
    }

    attention_params = {
        'name': 'lstm-attention',
        'hidden_units': args.hidden_units,
        'dropout': args.dropout,
    }

    attention_context_params = {
        'name': 'lstm-attention-context',
        'hidden_units': args.hidden_units,
        'dropout': args.dropout,
    }

    ntm_params = {
        'name': 'ntm',
        'n_slots': 128,
        'm_depth': 20,
        'read_heads': 1,
        'write_heads': 1,
        'shift_range': 3,
        'activation': 'softmax',  # for classification
        'hidden_units': args.hidden_units,
        'controller': 'ffnn'
    }

    if args.model == 'att':
        params = util.merge_dicts(shared_params, attention_params)
    elif args.model == 'att_con':
        params = util.merge_dicts(shared_params, attention_context_params)
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

    run_trial(model_fn, params, fixed_params, args.data, args.vectors,
              not args.eval)
