#!/usr/bin/env python3

import os
from setup import *
import numpy as np
import pandas as pd
import gender_guesser.detector as gender
from keras import optimizers
from keras.layers import Activation, Embedding, Flatten, Dense, Dropout, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import matplotlib.pyplot as plt


############################ Parameters #######################################
max_len = 500                         
train_samples = 1800
test_samples = 250 
valid_samples = 250
max_words = 103000
embedding_dim = 100
acc_plot_filename = 'accuracy.png'
loss_plot_filename = 'loss.png'
weights_filename = 'glove_model_01.h5'

data_filepath = '../../data/kaggle-data/'
filename = 'merged_data.csv'

acc_plot_filename = 'lstm_accuracy_01.png'
loss_plot_filename = 'lstm_loss_01.png'
weights_filename = 'lstm_glove_model_01.h5'
###############################################################################

# Load data
data = np.load('data')
labels = np.load('labels')

# Split the data
X_train = data[:train_samples]
y_train = labels[:train_samples]
X_test = data[train_samples:(train_samples + test_samples)]
y_test = labels[train_samples:(train_samples + test_samples)]
print("Data shapes are:\nX_train = {}\ny_train = {}\nX_test = {}\ny_test = {}"
        .format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

# Get embeddings
#embeddings_index = {}
#f = open(os.path.join(data_filepath, 'glove.6B.100d.txt'))
#for line in f:
#    values = line.split()
#    word = values[0]
#    coefs = np.asarray(values[1:], dtype='float32')
#    embeddings_index[word] = coefs
#f.close()

#print('Found {} word vectors'.format(len(embeddings_index)))

# Make embeddings matrix
#embedding_matrix = np.zeros((max_words, embedding_dim))
#for word, i in word_index.items():
#    if i < max_words:
#        embedding_vector = embeddings_index.get(word)
#        if embedding_vector is not None:
#            embedding_matrix[i] = embedding_vector

# Define the model
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_len))
#model.add(Flatten())
model.add(LSTM(100, activation='tanh'))
model.add(Dropout(0.3))
model.add(LSTM(100, activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# Load the glove embeddings
#model.layers[0].set_weights([embedding_matrix])
#model.layers[0].trainable = False

# Training
adam = optimizers.Adam()
model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['acc'])
history = model.fit(X_train, y_train,
                    epochs=50,
