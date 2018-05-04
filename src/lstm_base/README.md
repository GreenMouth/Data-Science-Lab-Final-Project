In this folder we have all our scripts for using LSTMs for generating text based off of previous TED talks.

lstmTesting
This is a library we made for our LSTM work that has a lot of useful functions for generating the model or prepping the data for text generation whether you're doing character distributions or word embeddings. 
Both trainChars and trainWords use functions from here extensively.

trainChars
This script breaks up the transcripts into segments of 100 sequential characters in the text and one target character which is the next character in the text. It then trains an LSTM on these sequences so that it learns how to predict the next character based off of the previous 100. Then, once trained, a seed of 100 characters from a random transcript is fed in and the LSTM freely begins to write it's own text.
To run this script just call it from the command line and it will build, train, then spit out the predicted text from the model into a text file called generated_chars.txt and the top weights in an hdf5 file.

trainWords
This script is the exact same as trainChars except it breaks the transcripts into sequences of five words at a time and has it guess the word that comes next. As well, generation is also done a word at a time.
Usage is same as the trainChars script except it outputs a text file called generated_words.txt