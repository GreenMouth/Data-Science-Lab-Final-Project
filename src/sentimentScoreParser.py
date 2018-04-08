import os
import textblob

class Logger():

    def __init__(self, datasetName, header= ".md"):
        self.datasetName = datasetName
        self.headerInfo = "Talk Title,Polarity,Subjectivity\n"
        self.header = header

    def createDataset(self):
        files = self.getAllDataFiles()
        allScores = self.collectScores(files)
        self.logToDataset(allScores)
        

    def getAllDataFiles(self):
        filesInCWD = [file for file in os.listdir('.') if os.path.isfile(file)]
        if self.datasetName not in filesInCWD:
            self.initDataCSV()

        dataFiles = [file for file in filesInCWD if self.isDataFile(file)]
        return dataFiles
                

    def initDataCSV(self):
        fileName = self.getFileName()

        with open(fileName, 'w+') as file:
            file.write(self.headerInfo)

    def getFileName(self):
        fileName = self.datasetName
        if '.csv' not in self.datasetName:
            fileName += '.csv'
        return fileName


    def isDataFile(self, fileName):
        if fileName[-len(self.header):] == self.header:
            return True
        else:
            return False
    
    def collectScores(self, files):
        scores = [self.collectScore(file) for file in files]
        return scores

    def collectScore(self, fileName):
        with open(fileName, 'r') as file:
            scores = textblob.TextBlob(file.read())
            polarity = str(round(scores.sentiment.polarity, 3))
            subjectivity = str(round(scores.sentiment.subjectivity, 3))
        return [fileName[:-len(self.header)], polarity, subjectivity]

    def logToDataset(self, scores):
        with open(self.getFileName(), 'a') as file:
            for fileScore in scores:
                file.write(','.join(fileScore) + "\n")

    
if __name__ == "__main__":
    outputName = "testDataSet.csv"
    dataLogger = Logger(outputName)
    dataLogger.createDataset()
