#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import gender_guesser.detector as gender
from keras import optimizers
from keras.layers import Activation, Embedding, Flatten, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import matplotlib.pyplot as plt


############################ Parameters #######################################
max_len = 10000                         
train_samples = 1800
test_samples = 250 
valid_samples = 250
max_words = 103000
embedding_dim = 100
acc_plot_filename = 'accuracy.png'
loss_plot_filename = 'loss.png'
weights_filename = 'glove_model_01.h5'

data_filepath = '../data/kaggle-data/'
filename = 'merged_data.csv'

###############################################################################

# Load the data
df = pd.read_csv(data_filepath + filename)
genDetector = gender.Detector()
speaker_names = df.main_speaker.tolist()

print ("Gathering gender data and adding it to data set...")

# Get gender data
first_names = []
genders = []
for name in speaker_names:
    first_last = name.split(' ')
    first = first_last[0]
    first_names.append(first)
    genders.append(genDetector.get_gender(first))

for i, gender in enumerate(genders):
    if gender == 'mostly_male':
        genders[i] = 'male'
    if gender == 'mostly_female':
        genders[i] = 'female'
    if gender == 'andy':
        genders[i] = 'unknown'

print("Current number of data points is {}".format(df.shape[0]))
print("Dropping unknown genders from data set...")
df['gender'] = genders
df = df[df.gender != 'unknown']
print("After dropping, there are {} data points".format(df.shape[0]))

# Convert to 0 and 1
#pd.Series(np.where(df.gender.values == 'male', 1, 0), df.index)
df.replace('male', 0, inplace=True)
df.replace('female', 1, inplace=True)
print("Labels converted to {}".format(type(df.gender.values)))

# Get a list of strings
transcripts = [row.clean_transcripts for row in df.itertuples()]

# Tokenize
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(transcripts)
sequences = tokenizer.texts_to_sequences(transcripts)
word_index = tokenizer.word_index
print("Found {} unique tokens".format(len(word_index)))

# Extract features and labels
data = pad_sequences(sequences, maxlen=max_len)
labels = df.gender.values
print("Shape of features: {}".format(data.shape))
print("Shape of labels:   {}".format(labels.shape))

# Get training and validation sets
X_train = data[:train_samples]
y_train = labels[:train_samples]
X_test = data[train_samples:(train_samples + test_samples)]
y_test = labels[train_samples:(train_samples + test_samples)]
print("Data shapes are:\nX_train = {}\ny_train = {}\nX_test = {}\ny_test = {}"
        .format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

# Get embeddings
embeddings_index = {}
f = open(os.path.join(data_filepath, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found {} word vectors'.format(len(embeddings_index)))

# Make embeddings matrix
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Define the model
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_len))
model.add(Flatten())
model.add(Dense(80, activation='tanh'))
model.add(Dropout(0.5))
#model.add(Dense(80, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# Load the glove embeddings
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

# Training
adam = optimizers.Adam()
model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['acc'])
history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=50,
                    validation_data=(X_test, y_test))
model.save_weights(weights_filename)

# Save plot
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

acc_plot_filename = 'accuracy.png'
loss_plot_filename = 'loss.png'
weights_filename = 'glove_model_01.h5'
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig(acc_plot_filename)

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig(loss_plot_filename)

