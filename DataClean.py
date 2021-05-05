import pandas as pd

class DataClean:
    # Init dataclean
    def __init__(self, data_set):
        self.data_set = data_set

    # Combines dat a set from init with a parsed dataset - finds key itself
    def combineDataSets(self, mergeFile):
        self.data_set = pd.merge(self.data_set, mergeFile)
        return self.data_set

    # Removes column from dataset - takes list input pass array of strings
    def removeCols(self, Col):
        for i in Col:
            del self.data_set[i]
        return self.data_set

    #Drops all rows with N/A (Wont delete oneHotEncodes because we havent split by the time we use this function
    def dropNA(self):
        self.data_set = self.data_set.dropna()
        return self.data_set