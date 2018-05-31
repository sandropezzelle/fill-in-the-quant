# --------------------------------------------------------
# --------------------------------------------------------
# code by Sandro Pezzelle (CIMeC, Unitn)
# January, 2018
# inspired by/adapted from:
# https://raw.githubusercontent.com/fchollet/keras/master/examples/imdb_bidirectional_lstm.py
# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
# --------------------------------------------------------
# --------------------------------------------------------

from __future__ import print_function
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
import keras.utils
from keras import utils as np_utils
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Activation, Input, Flatten
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint

#---------------------------------------------------------
### PARAMETERS TO BE SPECIFIED
#---------------------------------------------------------

punct = "no" #whether punctuation is taken into account or not. Options are: "yes","no"
setting = "starget" #context setting. Options are: "starget","left","right","s3"
tokens = 50 #maximum length of sequences. Options are: 50 (starget), 100 (right), 100 (left), 150 (s3)

batch_size = 64
hidden_units = 128
drop = 0.5
opt = "nadam"
n_epochs = 30

#---------------------------------------------------------
### FIXED PARAMETERS
#---------------------------------------------------------

model_name = "lstm-"+str(punct)+"-punct"
MAX_NB_WORDS = 50000 #50000 is actually bigger than our actual vocabulary, so we consider all the tokens
EMBEDDING_DIM = 300 #dimensionality of google vectors
classes = 9
quantifiers = ['none of ', 'a few of ', 'few of ', 'some of ', 'many of ', 'most of ', 'more than half of ', 'almost all of ', 'all of ']

path_vectors = '../../corpus-and-vectors/GoogleNews-vectors-negative300.txt' #where the google vectors are

train = open('../data/ACL18/'+str(setting)+'/'+str(setting)+'-train.txt', 'r')
valid = open('../data/ACL18/'+str(setting)+'/'+str(setting)+'-val.txt', 'r')
test = open('../data/ACL18/'+str(setting)+'/'+str(setting)+'-test.txt', 'r')

#---------------------------------------------------------
### LOAD DATA
#---------------------------------------------------------

x= train.readlines()
y = valid.readlines()
z = test.readlines()

num_train = len(x)
num_val = len(y)
num_test = len(z)

texts = []
for i in range(len(x)):
   f = x[i]
   f = f.rstrip('\n')
   f = f.split('\t')[0]
   texts.append(f)
print('train sentences appended')
for i in range(len(y)):
   f = y[i]
   f = f.rstrip('\n')
   f = f.split('\t')[0]
   texts.append(f)
print('val sentences appended')
for i in range(len(z)):
   f = z[i]
   f = f.rstrip('\n')
   f = f.split('\t')[0]
   texts.append(f)
print('test sentences appended')
print('Found %s texts.' % len(texts))

#---------------------------------------------------------
### LOAD PREDICTIONS
#---------------------------------------------------------

labels = []
for i in range(len(x)):
   l = x[i]
   l = l.rstrip('\n')
   l = l.split('\t')[1]
   idx=quantifiers.index(l)
   labels.append(idx)
train.close()
for i in range(len(y)):
   l = y[i]
   l = l.rstrip('\n')
   l = l.split('\t')[1]
   idx=quantifiers.index(l)
   labels.append(idx)
valid.close()
for i in range(len(z)):
   l = z[i]
   l = l.rstrip('\n')
   l = l.split('\t')[1]
   idx=quantifiers.index(l)
   labels.append(idx)
test.close()
print('Found %s labels.' % len(labels))

#---------------------------------------------------------
### TOKENIZATION AND SEQUENCE PADDING
#---------------------------------------------------------

MAX_SEQUENCE_LENGTH = tokens

if punct == "no":
  tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
else:
    #yes! I do not filter out punctuation
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS,filters='\t\n')

tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences,maxlen=MAX_SEQUENCE_LENGTH,padding="post")
#here we pad sequences by inserting 0s to complete the sequence up to max length
#by specifying "post" in both options, we are filling the cells at the right, e.g. [12,43,987,3,23,0,0,0,0]

point = num_train+num_val
x_train = data[:num_train]
print(len(x_train))
y_train = labels[:num_train]
x_val = data[num_train:point]
print(len(x_val))
y_val = labels[num_train:point]
x_test = data[point:]
y_test = labels[point:]
print(len(x_test))

#---------------------------------------------------------
### LOAD PRETRAINED EMBEDDINGS
#---------------------------------------------------------
#comment this and the following block if you want the model to learn embeddings from scratch
#instead of loading pretrained embeddings

print('loading embeddings...')

embeddings_index = {}
f = open(path_vectors)
#print('embeddings loaded')
for line in f:
#   print(line)
   values = line.split()
   word = values[0]
   coefs = np.asarray(values[1:], dtype='float32')
   embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

#---------------------------------------------------------
### BUILD EMBEDDING MATRIX (WEIGHTS)
#---------------------------------------------------------

gt = np.zeros((1, 300))
count = 0
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        #words not found in embedding index will be all-zeros. Random vectors might be put instead ?
        embedding_matrix[i] = embedding_vector
    if (embedding_matrix[i] == gt).all():
        count += 1
print(len(embedding_matrix))
print('%s tokens not found in space.' % count)

#---------------------------------------------------------
### DEFINE EMBEDDING LAYER (MODEL)
#---------------------------------------------------------

embedding_layer = Embedding(len(word_index) + 1, 
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            mask_zero=True, ###IMPORTANT: by setting "True", the model will ignore the cells filled with 0 !!
                            trainable=False) 

#in case we do not use pretrained embedding, comment the block above and comment out this line
#embedding_layer = Embedding(len(word_index) + 1,EMBEDDING_DIM,input_length=MAX_SEQUENCE_LENGTH)


#convert labels to categorical one-hot encoding for 'categorical crossentropy'
y_train_multi = keras.utils.to_categorical(y_train, num_classes=classes)
y_val_multi = keras.utils.to_categorical(y_val, num_classes=classes)

#---------------------------------------------------------
### DEFINE THE MODEL
#---------------------------------------------------------

model = Sequential()
model.add(embedding_layer)
model.add(LSTM(hidden_units))
model.add(Dropout(drop))
model.add(Dense(classes, activation = 'softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

#---------------------------------------------------------
### SAVE THE BEST WEIGHTS
#---------------------------------------------------------

filepath= "../data/ACL18/"+str(setting)+"/models/"+str(classes)+"-"+str(model_name)+"-"+str(MAX_SEQUENCE_LENGTH)+"word-"+str(hidden_units)+"hidden-"+str(opt)+"-"+str(drop)+"dropout-best-weights-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
#we save the weights of the model for which the validation loss is lowest!

print('Train the model...')
model.fit(x_train, y_train_multi,
          batch_size=batch_size,
          epochs=n_epochs,
          validation_data=[x_val, y_val_multi],
          callbacks=callbacks_list)

