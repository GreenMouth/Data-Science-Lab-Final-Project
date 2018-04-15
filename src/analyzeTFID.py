#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction import text

def getVectors(dataFrame):
    tfidfGenerator = text.TfidfVectorizer(input= scripts, stop_words= "english")
    matrix = tfidfGenerator.fit_transform(dataFrame['transcripts'])

    subjects = {'TFID': matrix}
    return pd.DataFrame(data= subjects)
    

if __name__ == "__main__":
    #fileName = sys.argv[0]
    fileName = mer
    dataFrame = pd.read_csv(fileName)
    vectorsDF = getVectors(dataFrame)
    merged = pd.concat([dataFrame, vectorsDF], axis= 1)
    merged.to_csv(fileName)
