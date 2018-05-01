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

# Truncate transcripts
trunc_transcripts = [elem[:1000] for elem in transcripts]

print("Running the LSTM ...")
new_data = []
for idx, transcript in enumerate(trunc_transcripts):
    new_data.extend(forward(transcript))
    if (idx % 2) == 0:
        print("Working on transcript number {}".format(idx))

# Extract features and labels
data = np.array(new_data)
labels = df.gender.values
print("Shape of features: {}".format(data.shape))
print("Shape of labels:   {}".format(labels.shape))
dest_file1 = 'data'
dest_file2 = 'labels'
np.save(dest_file1, data)
np.save(dest_file2, labels)
print("Wrote data and labels to {} and {}".format(dest_file1, dest_file2)) 
