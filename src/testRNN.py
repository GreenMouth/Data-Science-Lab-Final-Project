#!/usr/bin/env python3

import numpy as np
import pandas as pd
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


max_len = 10000
train_samples = 100
test_samples = 100
valid_samples = 100
max_words = 103000

# Load the data
filepath = '../data/kaggle-data/'
filename = 'merged_data.csv'
df = pd.read_csv(filepath + filename)

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


