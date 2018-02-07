# --------------------------------------------------------
# --------------------------------------------------------
# code by Shane Steinert-Threlkeld (ILLC, UvA)
# and Sandro Pezzelle (CIMec, Unitn)
# unless otherwise noted
# January 2018
# --------------------------------------------------------
# --------------------------------------------------------

from __future__ import print_function
from itertools import chain
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import keras.utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class SettingsValues(object):

    starget = 'starget'
    left = 'left'
    right = 'right'
    s3 = 's3'


def read_data_from_txt(path, setting, sets, label_list):
    """Reads input data from text files. It's assumed that the data comes in
    files of the form path/setting-X.txt where the X comes from the sets arg.
    Each file is assumed to contain a tab-separated pair of a text followed by
    a label, each coming from the provided argument.

    Args:
        path: path to text files
        setting: one of the SettingsValues
        sets: a list of strings
        label_list: a list of labels

    Returns:
        a tuple of dictionaries (texts, labels). Each dictionary has keys taken
        from sets, and has lists as values (strings for texts, integer indices
        for labels)
    """
    texts = {k: [] for k in sets}
    labels = {k: [] for k in sets}

    root = path + setting

    for dataset in sets:

        file_name = '{}-{}.txt'.format(root, dataset)
        with open(file_name, 'r') as f:
            data = f.readlines()

        for line in data:
            text = line.rstrip('\n')
            text, label = text.split('\t')
            texts[dataset].append(text)
            labels[dataset].append(label_list.index(label))

        print('{} data read'.format(dataset))

    return texts, labels


def generate_datasets(path, setting, sets, label_list, max_num_words,
                      max_seq_len, punct=True):
    """Generates datasets that can be used in Keras model.

    Args:
        path: as in read_data_from_txt
        setting: as in read_data_from_txt
        sets: as in read_data_from_txt
        label_list: as in read_data_from_txt
        max_num_words: maximum number of words to consider in data
        max_seq_len: maximum sequence length for data
        punct (Optional, default True): whether to include puncutation or not

    Returns:
        a pair of dictionaries (texts, labels), with keys from sets. Values in
        texts are lists of sequences that can be fed into Keras.  Values in
        labels are lists of labels.
    """

    texts, labels = read_data_from_txt(path, setting, sets, label_list)

    # convert labels to one_hot
    labels = {k: keras.utils.to_categorical(
        labels[k], num_classes=len(label_list))
        for k in labels}

    tokenizer = (Tokenizer(num_words=max_num_words, filters='\t\n') if punct
                 else Tokenizer(num_words=max_num_words))

    # note: texts.values() is a list of lists of texts; use * to pass it as
    # args to itertools.chain
    all_texts = list(chain(*texts.values()))

    tokenizer.fit_on_texts(all_texts)
    word_index = tokenizer.word_index
    print('Found {} unique tokens.'.format(len(word_index)))

    for dataset in texts:
        sequences = tokenizer.texts_to_sequences(texts[dataset])
        texts[dataset] = pad_sequences(sequences,
                                       maxlen=max_seq_len, padding='post')

    return texts, labels, word_index


# TODO: document!
def get_embedding_matrix(embedding_file, word_index, embedding_dim):

    print('loading embeddings...')

    # READ VECTORS FROM TEXT
    embeddings_index = {}
    with open(embedding_file, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector

    # GENERATE EMBEDDING MATRIX
    count = 0
    # unfound words will be all zero; random instead?
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        # found in our vectors
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            count += 1

    print('{} tokens not found in vector space.'.format(count))
    return embedding_matrix


def merge_dicts(*dicts):
    """Merges any number of dictionaries into one.

    Args:
        *dicts: some dictionaries

    Returns:
        a dictionary, containing all the k, v pairs from *dicts
    """
    merged = {}
    for dictionary in dicts:
        merged.update(dictionary)
    return merged

# see https://stackoverflow.com/a/33183634
def word2vec_bin_to_txt(in_file, out_file):
    model = KeyedVectors.load_word2vec_format(in_file, binary=True)
    model.save_word2vec_format(out_file, binary=False)


""" Example usage:
path = '../data/'
setting = SettingsValues.starget
sets = ['train', 'test', 'val']
texts, labels = generate_datasets(path, setting, sets,
                                  ['none of ', 'a few of ', 'few of ',
                                   'some of ', 'many of ', 'most of ',
                                   'more than half of ', 'almost all of ',
                                   'all of '],
                                  50000, 50)
print(texts['train'])
"""
