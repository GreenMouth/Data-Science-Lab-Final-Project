#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import text

def getVectors(dataFrame):
    scripts = dataFrame['transcript'].tolist()
    #print(len(scripts))
    #print(dataFrame['transcript'])
    #print(dataFrame.transcript.isnull().sum())
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

def main():
    #fileName = sys.argv[0]
    filepath = "../data/kaggle-data/"
    fileName = "merged_data.csv"
    print("Generating TFID vectors...")
    dataFrame = pd.read_csv(filepath + fileName)
    vectors = getVectors(dataFrame)
    writeFile(filepath + fileName, vectors)
    print("Wrote TFIFD data to {}".format(filepath + fileName[:-4] + "_vectors.csv"))

if __name__ == "__main__":
    main()
