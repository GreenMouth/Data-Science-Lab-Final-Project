#!/usr/bin/env python3

# Author John Sigmon

import pandas as pd
import numpy as np

def main():
    filename = 'merged_data.csv'
    filepath = '../data/kaggle-data/'
    data = pd.read_csv(filepath + filename)
    data = data.dropna(subset=['clean_transcripts'])
    data = data.dropna(subset=['transcript'])
    
    # Drop junk transcripts
    garbage = ['10 ways the world could end', 
            'How to make a splash in social media',
            'A cyber-magic card trick like no other',
            'How to go to space, without having to go to space']
    for title in garbage:
        data = data[data.title != title]
 
    #print(data.transcript.isnull().sum())
    data['stripped_transcripts'] = [row.clean_transcripts.split() for row in data.itertuples()]    
    _ = data.to_csv(filepath + filename)
    print("File written to {}".format(filepath + filename))

if __name__ == "__main__":
    main()
