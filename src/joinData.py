#!/usr/bin/env python3

#Author: John Sigmon
# Last updated: April 11, 2018

import finalStrip
import analyzeTFID
import analyzeSentiment

def main():
    print("Joining data sets")
    import pandas as pd
    filepath = '../data/kaggle-data/'
    file_1 = 'clean_transcripts.csv'
    file_2 = 'ted_main.csv'
    dest_file = 'merged_data.csv'

    dfs = [pd.read_csv(filepath + file_1), pd.read_csv(filepath + file_2)]
    df = pd.concat(dfs, axis=1)
    df.drop(df.columns[0], axis=1, inplace=True)
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    vectors = analyzeTFID.getVectors(df)
    analyzeTFID.writeFile(filepath + file_1, vectors)
    scoresDF = getSentimentScores(df)
    df = pd.concat([df, scoresDF], axis= 1)
    df.to_csv(filepath + dest_file)
    print('Your file was written to {}{}'.format(filepath, dest_file))
    finalStrip.main()

if __name__ == "__main__":
    main()
