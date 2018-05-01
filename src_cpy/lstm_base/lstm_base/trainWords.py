#!/usr/bin/env python3

from lstmTesting import *

if __name__ == "__main__":
    allScripts = getData()
    transcript = ' '.join(allScripts[:5])
    #transcript = allScripts[0]
    uniqueWords = sorted(set(transcript.split(' ')))
    numUniqueWords = len(uniqueWords)
    stringToInt = dict((string, i) for i, string in enumerate(uniqueWords))

    lengthOfSequence = 10
    words = transcript.split(" ")
    data, targets = prepSequences(words, stringToInt, lengthOfSequence)
    preppedX = prepX(data, lengthOfSequence, numUniqueWords)
    preppedY = prepY(targets)

    model = generateModel(preppedX, preppedY, size= 512)
    model = trainModel(model, preppedX, preppedY, numEpochs= 2)
    model = loadModel(model)

    intToString = dict((i, word) for i, word in enumerate(uniqueWords))  #creating a demapping of our original encoding
    seed = generateSeedFromData(data) #get a random starting point from our paper and let the network continue the writing
    numWords= 300   #length of each window the network will use to predict the output
    text = generateText(model, seed, intToString, length= numWords, vocabSize= numUniqueWords, delimeter= ' ')
    with open("generated_words.txt", 'w+') as file:
        file.write(text)
