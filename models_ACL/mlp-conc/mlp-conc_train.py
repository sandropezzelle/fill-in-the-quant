# --------------------------------------------------------
# Sandro Pezzelle, University of Trento
# 2018
# --------------------------------------------------------

from __future__ import print_function
import numpy as np
import sklearn
import argparse
from sklearn.metrics import confusion_matrix
import keras.utils
from keras import utils as np_utils
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Activation, Input, Flatten
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint

"""
read arguments,
define preliminary settings

"""
dataset_path = '../../data/'
embeddings_filename ='../../data/GoogleNews-vectors-negative300.txt'
weights_path = '../../models_weights/'


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default=dataset_path)
parser.add_argument("--embeddings_filename", type=str, default=embeddings_filename)
parser.add_argument("--weights_path", type=str, default=weights_path)
parser.add_argument("--setting", type=str, required=True, choices=["1-Sent","3-Sent"])
parser.add_argument("--punctuation", type=str, default="yes", choices=["yes","no"])
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--MAX_NB_WORDS", type=int, default=50000)
parser.add_argument("--EMBEDDING_DIM", type=int, default=300)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--hidden_units", type=int, default=512)
parser.add_argument("--optimizer", type=str, default="adagrad")
args = parser.parse_args()

"""
set preliminary settings,
define paths

"""

if args.punctuation == "no":
  tokenizer = Tokenizer(num_words=args.MAX_NB_WORDS)
else:
    tokenizer = Tokenizer(num_words=args.MAX_NB_WORDS,filters='\t\n')

if args.setting == "1-Sent":
  n_tokens = 50
  data_path = dataset_path+str(args.setting)+'/'
else:
    n_tokens = 150
    data_path = dataset_path+str(args.setting)+'/'

mod_name = "mlp-conc"
weights_filename = weights_path+str(mod_name)+'/'+str(args.setting)+'_my-best-model.hdf5'


quantifiers = ['none of ', 'a few of ', 'few of ', 'some of ', 'many of ', 'most of ', 'more than half of ', 'almost all of ', 'all of ']
MAX_SEQUENCE_LENGTH = n_tokens
dim_sentence = int((args.EMBEDDING_DIM)*(MAX_SEQUENCE_LENGTH))
empty_vec = np.zeros((1,300))

texts,labels = [],[]
embeddings_index = {}

"""
read partitions of dataset

"""

train = open(data_path+'train.txt', 'r')
val = open(data_path+'val.txt', 'r')
test = open(data_path+'test.txt', 'r')


def read_dataset(split):
   s = split.readlines()
   num_s = len(s)
   s_txt,s_lab = [],[]
   for i in range(num_s):
      f = s[i]
      f = f.rstrip('\n')
      t = f.split('\t')[0]
      s_txt.append(t)
      l = f.split('\t')[1]
      idx = quantifiers.index(l)
      s_lab.append(idx)
   split.close()
   return s_txt,s_lab


def build_embedding_matrix(path_to_vectors, word_dictionary):
   print('loading embeddings...')
   f = open(path_to_vectors)
   for line in f:
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs
   f.close()
   print('Found %s word vectors.' % len(embeddings_index))

   count = 0
   embedding_matrix = np.zeros((len(word_dictionary) + 1, args.EMBEDDING_DIM))
   for word, i in word_dictionary.items():
       embedding_vector = embeddings_index.get(word)
       if embedding_vector is not None:
           embedding_matrix[i] = embedding_vector
       if (embedding_matrix[i] == empty_vec).all():
           count += 1
   print('%s tokens not found in space.' % count)     
   return embedding_matrix


def build_matrix(partition):
   partition_matrix = np.zeros((len(partition), dim_sentence))
   for i,dtp in enumerate(partition):
       conc = []
       for n in dtp:
           vector = embedding_matrix[n]
           conc = np.append(conc,vector)
       conc.flatten()
       partition_matrix[i] = conc
   return partition_matrix



if __name__ == '__main__':
    """
    it initializes the hyperparameters,
    preprocesses the input,
    trains the model

    """
    tr_txt, tr_lab = read_dataset(train)
    #print(len(tr_txt))
    texts.append(tr_txt)
    labels.append(tr_lab)
    val_txt, val_lab = read_dataset(val)
    texts.append(val_txt)
    labels.append(val_lab)
    tst_txt, tst_lab = read_dataset(test)
    texts.append(tst_txt)
    labels.append(tst_lab)

    texts = [itt for subtxt in texts for itt in subtxt]
    labels = [itl for sublab in labels for itl in sublab]

    print(len(texts))

    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH,truncating='post')

    a = int(len(tr_txt))
    b = int(len(tr_txt))+int(len(val_txt))
    
    x_train = data[:a]
    y_train = labels[:a]
    x_val = data[a:b]
    y_val = labels[a:b]
    x_test = data[b:]
    y_test = labels[b:]

    embedding_matrix = build_embedding_matrix(embeddings_filename,word_index)

    x_tr = build_matrix(x_train)
    x_v = build_matrix(x_val)
    x_t = build_matrix(x_test)

    y_train_multi = keras.utils.to_categorical(y_train, num_classes=9)
    y_val_multi = keras.utils.to_categorical(y_val, num_classes=9)
    y_test_multi = keras.utils.to_categorical(y_test, num_classes=9)

    print('Building model...')
    model = Sequential()
    model.add(Dense(args.hidden_units, input_shape=(dim_sentence,)))
    model.add(Activation('relu'))
    model.add(Dropout(args.dropout))
    model.add(Dense(9, activation = 'softmax'))

    model.compile(loss='categorical_crossentropy',
              optimizer=args.optimizer,
              metrics=['accuracy'])

    checkpoint = ModelCheckpoint(weights_filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    print('Train the model...')
    model.fit(x_tr, y_train_multi,
              batch_size=args.batch_size,
              epochs=args.num_epochs,
              validation_data=[x_v, y_val_multi],
              callbacks=callbacks_list)

