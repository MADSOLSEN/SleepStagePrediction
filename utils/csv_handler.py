import pandas as pd
import numpy as np
import os

class CSVHandler:

    def __init__(
            self,
            name,
            labels
    ):
        self.name = name
        self.labels = labels
        self.table = self.start_table()

    def start_table(self):
        new_table = np.empty((0, len(self.labels)))
        return new_table

    def add_to_table(self, table):
        self.table = np.append(self.table, table, axis=0)

    def save_to_csv(self, path, round_off=False):
        self.create_directory_tree(path)
        df = pd.DataFrame(self.table, columns=self.labels)
        df.to_csv('{}\\{}.csv'.format(path, self.name))

    def return_dataframe(self):
        return pd.DataFrame(self.table, columns=self.labels)

    def save_to_txt(self, path):
        self.create_directory_tree(path)
        np.savetxt('{}\\{}.txt'.format(path, self.name), (self.table).values, fmt='%d')

    @staticmethod
    def create_directory_tree(full_filename):
        full_filename_split = full_filename.split('\\')
        full_filename_reconstruction = ''
        for n in range(len(full_filename_split)):
            full_filename_reconstruction += full_filename_split[n] + '\\'
            if not os.path.isdir(full_filename_reconstruction):
                os.mkdir(full_filename_reconstruction)

