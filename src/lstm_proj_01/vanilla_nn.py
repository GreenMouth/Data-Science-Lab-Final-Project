#!/usr/bin/env python3

import os
from forward import *
import numpy as np
import pandas as pd
import gender_guesser.detector as gender
from keras import optimizers
from keras.layers import Activation, Embedding, Flatten, Dense, Dropout, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


############################ Parameters #######################################
max_words = 1000
max_len = 50
train_samples = 1900
test_samples = 250 
#valid_samples = 250

data_filepath = '../../data/kaggle-data/'
filename = 'merged_data.csv'

acc_plot_filename = 'lstm_accuracy_06.png'
loss_plot_filename = 'lstm_loss_06.png'
weights_filename = 'lstm_glove_model_06.h5'
###############################################################################
'''
df = pd.read_csv(data_filepath + filename)
gen_detector = gender.Detector()
speaker_names = df.main_speaker.tolist()

first_names = []
genders = []
for name in speaker_names:
    first_last = name.split(' ')
    first = first_last[0]
    first_names.append(first)
    genders.append(gen_detector.get_gender(first))

for i, gender in enumerate(genders):
    if gender == 'mostly_nale':
        genders[i] = 'male'
    if gender == 'mostly_female':
        genders[i] = 'female'
    if gender == 'andy':
        genders[i] = 'unknown'

df['gender'] = genders
df = df[df.gender != 'unknown']
df.replace('male', 0, inplace=True)
df.replace('female', 1, inplace=True)

transcripts = [row.clean_transcripts for row in df.itertuples()]
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(transcripts)
sequences = tokenizer.texts_to_sequences(transcripts)
word_index = tokenizer.word_index
'''
data = np.load('data.npy')#pad_sequences(sequences, maxlen=max_len)
labels = np.load('labels.npy')

# Split the data
X_train = data[:train_samples]
y_train = labels[:train_samples]
X_test = data[train_samples:(train_samples + test_samples)]
y_test = labels[train_samples:(train_samples + test_samples)]
print("Data shapes are:\nX_train = {}\ny_train = {}\nX_test = {}\ny_test = {}"
        .format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

# Define the model
model = Sequential()
#model.add(Embedding(2152, 1024, input_length=max_len))
#model.add(Flatten())
model.add(Dense(512, activation='relu', use_bias=True, input_shape=(1024,)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu', use_bias=True))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu', use_bias=True))
model.add(Dropout(0.5))
#model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.7))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#model.layers[0].set_weights([np.load('data.npy')])
#model.layers[0].trainable = False

# Training
#adam = optimizers.Adam()
sgd = optimizers.SGD()
model.compile(optimizer='sgd',
            loss='binary_crossentropy',
            metrics=['acc'])
history = model.fit(X_train, y_train,
			epochs=300,
			batch_size=600,
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
#plt.savefig(acc_plot_filename)

plt.plot(epochs, loss, 'ro', label = 'Training Cross Entropy Loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation Cross Entropy Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss or Accuracy')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig(loss_plot_filename)
