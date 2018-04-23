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

def trainModel(model, X, y, epochNum= 20):
    filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5" 
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.fit(X, y, epochs= epochNum, batch_size=128, callbacks=callbacks_list)
    return model

def generateSeedFromData(data):
    start = np.random.randint(0, len(data)-1)
    pattern = data[start]
    print("Starting Seed: ", ''.join([intToChar[value] for value in pattern]), end= '\n\n\n')
    return pattern

def generateText(model, pattern, decoding, length= 1000, vocabSize= 47):
    text= ''
    for i in range(length):
        preppedPattern = prepSeed(pattern, vocabSize)
        prediction = model.predict(preppedPattern, verbose= 0)
        index = np.argmax(prediction)
        result = decoding[index]
        print('here: {0}\n{1}\n{2}'.format(prediction, index, result), end='\n\n\n')
        text += result
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
    filename = "weights-improvement-19-1.9435.hdf5" #replace with best weights file
    model.load_weights(filename)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    intToChar = dict((i, char) for i, char in enumerate(chars))  #creating a demapping of our original encoding
    seed = generateSeedFromData(data) #get a random starting point from our paper and let the network continue the writing
    numCharacters= 100   #length of each window the network will use to predict the output
    text = generateText(model, seed, intToChar, length= numCharacters, vocabSize= numUniqueChars)
    print(text)
