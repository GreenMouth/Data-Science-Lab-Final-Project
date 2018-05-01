#!/usr/bin/env python3

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

defaultFileName = "topWeights.hdf5"

def getData(filePath = '../../../data/kaggle-data/', fileName= 'clean_transcripts.csv'):
    dataFrame = pd.read_csv(filePath + fileName)
    allScripts = dataFrame['transcript'].tolist()
    allScripts = [script.lower() for script in allScripts]
    return allScripts

def prepSequences(rawText, encoding, sequenceLength = 100): 
    data = []
    targets = []
    for i in range(0, len(rawText) - sequenceLength, 1):
        sequence = rawText[i: i+sequenceLength]
        target = rawText[i + sequenceLength]
        data.append([encoding[char] for char in sequence]) #Here we are encoding the characters to their previous assigned values
        targets.append(encoding[target])                   #Same with the target answer

    return data, targets

def prepX(data, lengthOfSequence, numUniqueChars):
    data = np.reshape(data, (len(data), lengthOfSequence, 1))
    data = data / float(numUniqueChars)
    return data

def prepY(targets):
    targets = np_utils.to_categorical(targets)
    return targets

def generateModel(X, y, size= 256):
    model = Sequential()
    model.add(LSTM(size, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(size))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def trainModel(model, X, y, numEpochs= 20, batchSize= 128):
    filepath= defaultFileName #replace with lowest loss file
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.fit(X, y, epochs = numEpochs, batch_size= batchSize, callbacks=callbacks_list)
    return model


def loadModel(model, filename= defaultFileName):  #replace with best weights file for your training
    model.load_weights(filename)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

##Now for text generation
def generateSeedFromData(data):
    start = np.random.randint(0, len(data)-1)
    pattern = data[start]
    return pattern


def generateText(model, pattern, decoding, length= 1000, vocabSize= 47, delimeter= ''):
    text= delimeter.join([decoding[value] for value in pattern])
    for i in range(length):
        preppedPattern = prepPattern(pattern, vocabSize)
        prediction = model.predict(preppedPattern, verbose= 0)
        index = np.argmax(prediction)
        result = decoding[index]
        text += delimeter + result
        pattern.append(index)
        pattern = pattern[1:]
        
    return text

def prepPattern(pattern, vocabSize):
    pattern = np.reshape(pattern, (1, len(pattern), 1))
    pattern = pattern / float(vocabSize)
    return pattern

if __name__ == "__main__":
    allScripts = getData()
    script1 = allScripts[0]

    chars = sorted(list(set(script1)))
    charsToInt = dict((char, i) for i, char in enumerate(chars))
    uniqueChars = len(charsToInt)

    lengthOfSequence = 100
    data, targets = prepSequences(script1, charsToInt, sequenceLength = lengthOfSequence)
    preppedX = prepX(data, lengthOfSequence, uniqueChars)
    preppedY = prepY(targets)
    model = generateModel(preppedX, preppedY)
    model = trainModel(model, preppedX, preppedY, 1)

    ###generation through here
    #filename = "weights-improvement-19-1.9435.hdf5" #replace with best weights file
    #model.load_weights(filename)
    #model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    intToChar = dict((i, char) for i, char in enumerate(chars))  #creating a demapping of our original encoding
    seed = generateSeedFromData(data) #get a random starting point from our paper and let the network continue the writing
    numCharacters= 100   #length of each window the network will use to predict the output
    text = generateText(model, seed, intToChar, length= numCharacters, vocabSize= numUniqueChars)
    print(text)


    ###############Below here is building an LSTM to predict on words not chars#############

    ##Let's look at a list of all unique characters in our scripts, 
    ##we'll eventually need to one hot encode them to make training easier:
    uniqueWords = sorted(script1.split(' '))
    numUniqueWords = len(uniqueWords)

    ##Lets make a mapping of each character to a specific number, this will help our training since we need numerical data:
    stringToInt = dict((string, i) for i, string in enumerate(uniqueWords))

    lengthOfSequence = 25
    words = transcript.split(" ")
    data, targets = prepSequences(words, stringToInt, lengthOfSequence)
    preppedX = prepX(data, lengthOfSequence, numUniqueWords)
    preppedY = prepY(targets)

    model = generateModel(preppedX, preppedY, size= 512)
    model = trainModel(model, preppedX, preppedY, 5, 5)

    
    model = loadModel(model, "weights-improvement-05-6.2815.hdf5")
    intToString = dict((i, word) for i, word in enumerate(uniqueWords))  #creating a demapping of our original encoding
    seed = generateSeedFromData(data) #get a random starting point from our paper and let the network continue the writing
    numWords= 100   #length of each window the network will use to predict the output
    text = generateText(model, seed, intToString, length= numWords, vocabSize= numUniqueWords, delimeter= ' ')
    print(text)
