#!/usr/bin/env python3

import os
from forward import *
import numpy as np
import pandas as pd
import gender_guesser.detector as gender
from keras import optimizers
from keras.layers import Activation, Embedding, Flatten, Dense, Dropout, LSTM, Conv1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


############################ Parameters #######################################
max_len = 500                         
train_samples = 1900
test_samples = 250 
valid_samples = 250
acc_plot_filename = 'accuracy.png'
loss_plot_filename = 'loss.png'
weights_filename = 'glove_model_01.h5'

data_filepath = '../../data/kaggle-data/'
filename = 'merged_data.csv'

acc_plot_filename = 'lstm_accuracy_02.png'
loss_plot_filename = 'lstm_loss_02.png'
weights_filename = 'lstm_glove_model_02.h5'
###############################################################################

# Load data
data = np.load('data.npy')
labels = np.load('labels.npy')

# Split the data
X_train = data[:train_samples]
y_train = labels[:train_samples]
X_test = data[train_samples:(train_samples + test_samples)]
y_test = labels[train_samples:(train_samples + test_samples)]
print("Data shapes are:\nX_train = {}\ny_train = {}\nX_test = {}\ny_test = {}"
        .format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

#embedding_layer = Embedding(len word index + 1,
#                            1024,
#                            weights = data,

# Define the mode
model = Sequential()

#model.add(Dense(512, activation='relu', input_dim=1024))
model.add.Conv1D()
model.add(Dropout(0.7))
#model.add(Dense(512, activation='relu'))
model.addConv1D()
model.add(Dropout(0.7))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# Training
adam = optimizers.Adam()
model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['acc'])
history = model.fit(X_train, y_train,
			epochs=60,
			batch_size=80,
			validation_data=(X_test, y_test))
model.save_weights(weights_filename)

# Save Plot
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig(acc_plot_filename)

plt.plot(epochs, loss, 'ro', label = 'Training Loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig(loss_plot_filename)
