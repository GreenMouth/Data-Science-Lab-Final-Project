#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import text

def getVectors(dataFrame):
    scripts = dataFrame['transcript'].tolist()
    print(len(scripts))
    print(dataFrame['transcript'])
    tfidfGenerator = text.TfidfVectorizer(input= scripts, stop_words= "english")
    matrix = tfidfGenerator.fit_transform(scripts)
    return matrix

def writeFile(fileName, vectors):
    with open(fileName[:-4] + "_vectors.csv", 'w+') as file:
        for row in vectors:
            line = ''
            for num in row:
                line += str(num) + ','
            line = line[:-1]
            line = line + '\n'
            file.write(line)

if __name__ == "__main__":
    #fileName = sys.argv[0]
    fileName = "clean_transcripts.csv"
    print("Generating TFID vectors...")
    dataFrame = pd.read_csv(fileName)
    vectors = getVectors(dataFrame)
    writeFile(fileName, vectors)
    print("Wrote TFIFD data to {}".format(fileName[:-4] + "_vectors.csv"))
