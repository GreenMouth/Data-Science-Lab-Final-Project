#!/usr/bin/env python3

import os
from forward import *
import numpy as np
import pandas as pd

data_filepath = '../../data/kaggle-data/'
filename = 'merged_data.csv'

df = pd.read_csv(data_filepath + filename)

transcripts = [row.clean_transcripts for row in df.itertuples()]

min_word = 0
mean_word = 0
max_word = 0
total = 0
for transcript in transcripts:
    transcript = transcript.replace("\n", " ")
    splitTranscript = transcript.split(" ")
    numWords = len(splitTranscript)
    if numWords > max_word:
        max_word = numWords
    elif numWords < min_word:
        min_word = numWords
    total += numWords

mean_word = total / float(len(transcripts))

min_char = 0
mean_char = 0
max_char = 0
total = 0
for transcript in transcripts:
    splitTranscript = list(transcript)
    numChars = len(splitTranscript)
    if numChars > max_char:
        max_char = numChars
    elif numChars < min_char:
        min_char = numChars

    total += numChars

mean_char = total / float(len(transcripts))

print("Mean words {}".format(mean_word))
print("Min words {}".format(min_word))
print("Max words {}".format(max_word))
print("Mean char {}".format(mean_char))
print("Min char {}".format(min_char))
print("Max char {}".format(max_char))
