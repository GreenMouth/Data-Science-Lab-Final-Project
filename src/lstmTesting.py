import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

def getData():
    dataFrame = pd.read_csv('transcripts.csv')
    allScripts = dataFrame['transcript'].tolist()
    allScripts = [script.lower() for script in allScripts]
    return allScripts

def prepSequences(rawText, encoding, sequenceLength = 100):
    data = []
    targets = []
    for i in range(0, len(rawText) - sequenceLength, 1):
        sequence = rawText[i: i+sequenceLength]
        target = rawText[i + sequenceLength]
        data.append([encoding[char] for char in sequence])
        targets.append(encoding[target])

    return data, targets

def prepX(data, lengthOfSequence, uniqueChars):
    data = np.reshape(data, (len(data), lengthOfSequence, 1))
    data = data / float(uniqueChars)
    return data

def prepY(targets):
    targets = np_utils.to_categorical(targets)
    return targets

def generateModel(X, y):
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def trainModel(X, y):
    model = generateModel(X, y)

    filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5" #replace with lowest loss file
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)
    return model

def generateSeedFromData(data):
    start = np.random.randint(0, len(data)-1)
    pattern = data[start]
    return pattern

def generateText(model, seed, decoding, length= 1000, uniqueChars= 47):
    for i in range(length):
        seed = prepSeed(seed)
        prediction = model.predict(seed, verbose= 0)
        index = np.argmax(prediction)
        result = decoding[index]
        seed.append(index)
        
    text = [intToChar[value] for value in seed]
    return ''.join(text)
    

def prepSeed(seed, vocabSize)
    seed = np.reshape(seed, (1, len(seed), 1))
    seed = seed / float(uniqueChars)
    return seed

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
    model = trainModel(preppedX, preppedY)

    ###generation through here
    filename = "weights-improvement-19-1.9435.hdf5" #replace with best weights file
    model.load_weights(filename)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    intToChar = dict((i, char) for i, char in enumerate(chars))
    seed = generateSeedFromData(data)
    numCharacters= 100
    text = generateText(model, seed, intToChar, length= numCharacters, vocabSize= uniqueChars)
