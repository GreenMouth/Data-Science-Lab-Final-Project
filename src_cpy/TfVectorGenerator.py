#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity


transcripts = pd.read_csv("transcripts.csv")
transcripts['title']=transcripts['url'].map(lambda x:x.split("/")[-1])

def analyzeScripts():
    scripts = getScripts()
    vectorMatrix = generateTFIDMatrix(scripts)
    unigramMatrix = generateUnigramMatrix(vectorMatrix)
    return vectorMatrix, unigramMatrix

def getScripts():
    return transcripts['transcript'].tolist()

def generateTFIDMatrix(scripts):
    tfidfGenerator = text.TfidfVectorizer(input= scripts, stop_words= "english")
    matrix = tfidfGenerator.fit_transform(scripts)
    return matrix

def generateUnigram(tfidMatrix):
    return cosine_similarity(tfidMatrix)

def getSimilarArticles(articleText):
    allScripts = getScripts()
    allScripts.append(articleText)
    tfdiMatrix = generateTFIDMatrix(allScripts)
    unigram = generateUnigram(tfdiMatrix)
    return ",".join(transcripts['title'].loc[unigram[-1].argsort()[-5:-1]])


if __name__ == "__main__":
    allScripts = getScripts()
    testText = allScripts.pop(5)
    transcripts.drop(5, inplace=True)
    #tfidfGenerator = text.TfidfVectorizer(input= allScripts, stop_words= "english")
    #matrix = tfidfGenerator.fit_transform(allScripts)
    #print(matrix.shape)

    #print(generateTFIDMatrix(allScripts).shape)
    #print(generateUnigram(matrix).shape)
    print(getSimilarArticles(testText))
    
