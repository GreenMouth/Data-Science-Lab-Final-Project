#!/usr/bin/env python3

import numpy as np
import pandas as pd
from keras import optimizers
from keras.layers import Embedding, Flatten, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import matplotlib.pyplot as plt


############################ Parameters #######################################
max_len = 10000                         
train_samples = 100
test_samples = 100
valid_samples = 100
max_words = 103000

embedding_dim = 1000

acc_plot_filename = 'accuracy.png'
loss_plot_filename = 'loss.png'
weights_filename = 'glove_model_01.h5'

data_filepath = '../data/kaggle-data/'
filename = 'merged_data.csv'

###############################################################################

# Load the data
df = pd.read_csv(data_filepath + filename)

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
labels = df.views.values
print("Shape of features: {}".format(data.shape))
print("Shape of labels:   {}".format(labels.shape))

# Get training and validation sets
X_train = data[:train_samples]
y_train = data[:train_samples]
X_test = data[train_samples:(train_samples + test_samples)]
y_test = data[train_samples:(train_samples + test_samples)]

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
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='relu'))
model.summary()

# Load the glove embeddings
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

# Training
adam = keras.optimizers.Adam()
model.compile(optimizer='adam',
            loss='mean_squared_error',
            metrics=['acc'])
history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=32,
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

