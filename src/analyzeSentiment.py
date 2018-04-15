#!/usr/bin/env python3
import sys
import pandas as pd
import textblob

def getSentimentScores(dataFrame):
    polarities = []
    subjectivity = []
    for row in dataFrame.rows():
        scores = textblob.TextBlob(row['transcript'])
        polarities.append(scores.sentiment.polarity)
        subjectivity.append(scores.sentiment.subjectivity)

    subjects = {'polarity': polarities, 'subjectivity': subjectivity}
    return pd.DataFrame(data= subjects)
    

if __name__ == "__main__":
    fileName = sys.argv[0]
    dataFrame = pd.read_csv(fileName)
    scoresDF = getSentimentScores(dataFrame)
    merged = pd.concat([dataFrame, scoresDF], axis= 1)
    merged.to_csv(fileName)
