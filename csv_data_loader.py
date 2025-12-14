import csv
import os

class CSVDataLoader:
    def __init__(self, filePath: str, hasHeader=True):
        self.__filePath = filePath
        self.__hasHeader = hasHeader
        self.__filesize = 0
        self.__data = []
        self.__features = []
        self.__labels = []
        self.__namesOfFeatures = []
        self.__namesOfLabels = []
        self.__labelsMapping = {} 
    

    # ---------- SETTERS ----------
    def setFilePath(self, filePath: str):
        self.__filePath = filePath

    def setData(self, data):
        self.__data = data

    def setFileSize(self, filesize):
        self.__filesize = filesize

    def setFeatures(self, features):
        self.__features = features

    def setLabels(self, labels):
        self.__labels = labels

    def setNamesOfFeatures(self, names):
        self.__namesOfFeatures = names

    def setNamesOfLabels(self, names):
        self.__namesOfLabels = names

    # ---------- GETTERS ----------
    def getFilePath(self):
        return self.__filePath

    def getData(self):
        return self.__data

    def getFileSize(self):
        return self.__filesize

    def getFeatures(self):
        return self.__features
  
    def getLabels(self):
        return self.__labels

    def getNamesOfFeatures(self):
        return self.__namesOfFeatures

    def getNamesOfLabels(self):
        return self.__namesOfLabels
    
    def getLabelsMapping(self):
        return self.__labelsMapping
    
    def getNumberOfFeatures(self):
        return len(self.__namesOfFeatures) if self.__namesOfFeatures else 0

    # ---------- LABEL ENCODING ----------
    def changeLabelsTypesToNumeric(self):
        # Use sorted unique labels for stable, predictable mapping
        unique_labels = sorted(set(self.__labels))

        # Build mapping: {"cat":0, "dog":1...}
        self.__labelsMapping = {label: i for i, label in enumerate(unique_labels)}

        # Replace labels with numbers
        self.__labels = [self.__labelsMapping[label] for label in self.__labels]


    # ---------- MAIN METHOD ----------
    def loadData(self):
        data = []
        features = []
        labels = []

        self.setFileSize(os.path.getsize(self.__filePath))

        with open(self.__filePath, 'r') as f:
            reader = csv.reader(f)

            # Header extraction
            if self.__hasHeader:
                header = next(reader)
                self.setNamesOfFeatures(header[:-1])
                self.setNamesOfLabels([header[-1]])

            # Read rows
            for row in reader:
                *f_values, label = row

                try:
                    f_values = [float(x) for x in f_values]
                except ValueError:
                    print("valeur non numérique →", row)
                    continue
                
                data.append(row)
                features.append(f_values)
                labels.append(label)  # KEEP AS STRING UNTIL WE ENCODE

        self.setData(data)
        self.setFeatures(features)
        self.setLabels(labels)

        # IMPORTANT: encode labels after loading
        self.changeLabelsTypesToNumeric()

        return self.getFeatures(), self.getLabels()