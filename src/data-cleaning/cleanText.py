#!/usr/bin/env python3
# Author: John Sigmon
# Last updated: April 14, 2018 by Kate Baumli

import pandas as pd

def main():
    print('Cleaning your data...')
    filename = 'transcripts.csv'
    target_filename = 'clean_transcripts.csv'
    filepath = '../data/kaggle-data/'

    df = pd.read_csv(filepath + filename)
    
    # Remove non-ascii first
    df.transcript.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
    patts = [".", "?", "!", "\'", "(Laughter)", "(Applause)", "(", ")", "\"", "\n ", "-", ";", ":"]
    repl = ["\n", "", "Laughter\n", "Applause\n"]

    # Replace everything
    df['clean_transcripts'] = df.transcript.str.replace(patts[0], repl[0])
    df['clean_transcripts'] = df.clean_transcripts.str.replace(patts[1], repl[0])
    df['clean_transcripts'] = df.clean_transcripts.str.replace(patts[2], repl[0])
    df['clean_transcripts'] = df.clean_transcripts.str.replace(patts[3], repl[1])
    df['clean_transcripts'] = df.clean_transcripts.str.replace(patts[4], repl[2])
    df['clean_transcripts'] = df.clean_transcripts.str.replace(patts[5], repl[3])
    df['clean_transcripts'] = df.clean_transcripts.str.replace(patts[6], repl[1])
    df['clean_transcripts'] = df.clean_transcripts.str.replace(patts[7], repl[1])
    df['clean_transcripts'] = df.clean_transcripts.str.replace(patts[8], repl[1])
    df['clean_transcripts'] = df.clean_transcripts.str.replace(patts[9], repl[1])
    df['clean_transcripts'] = df.clean_transcripts.str.replace(patts[10], repl[1])
    df['clean_transcripts'] = df.clean_transcripts.str.replace(patts[11], repl[1])
    
    # Make everything lower case
    df['clean_transcripts'] = df['clean_transcripts'].str.lower()

    df.to_csv(filepath + target_filename)
    print('Your new .csv has been written to {}'.format(filepath 
        + target_filename))

if __name__ == "__main__":
    main()
