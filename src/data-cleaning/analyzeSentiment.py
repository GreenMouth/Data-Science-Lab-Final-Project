#!/usr/bin/env python3
import sys
import pandas as pd
import textblob

def getSentimentScores(dataFrame):
    polarities = []
    subjectivity = []
    for index, row in dataFrame.iterrows():
        try:
            scores = textblob.TextBlob(row['transcript'])
            polarities.append(scores.sentiment.polarity)
            subjectivity.append(scores.sentiment.subjectivity)
        except:
            pass
    subjects = {'polarity': polarities, 'subjectivity': subjectivity}
    return pd.DataFrame(data= subjects)
    
def main():
    #fileName = sys.argv[0]
    filepath = "../data/kaggle-data/"
    fileName = "merged_data.csv"
    print("Analyzing sentiment...")
    dataFrame = pd.read_csv(filepath + fileName)
    scoresDF = getSentimentScores(dataFrame)
    merged = pd.concat([dataFrame, scoresDF], axis= 1)
    merged[['polarity', 'subjectivity']].fillna(value = 0)
    merged.to_csv(filepath + fileName)
    print("Rewrote data back to {}".format(filepath + fileName))

if __name__ == "__main__":
    main()
