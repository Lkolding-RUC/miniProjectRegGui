import pandas as pd

class DataClean:
    # Init dataclean
    def __init__(self, data_set):
        self.data_set = data_set

    # Removes column from dataset - takes list input pass array of strings
    def keep_columns(self, col):
        for i in self.data_set.head():
            if i not in col:
                del self.data_set[i]
        return self.data_set
