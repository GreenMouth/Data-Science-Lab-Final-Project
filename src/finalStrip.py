#!/usr/bin/env python3

# Author John Sigmon

import pandas as pd
import numpy as np

def main():
    filename = 'merged_data.csv'
    filepath = '../data/kaggle-data/'
    data = pd.read_csv(filepath + filename)
    data = data.dropna(subset=['clean_transcripts'])
    data['stripped_transcripts'] = [row.clean_transcripts.split() for row in data.itertuples()]    
    _ = data.to_csv(filepath + filename)
    print("File written to {}".format(filepath + filename))

if __name__ == "__main__":
    main()
