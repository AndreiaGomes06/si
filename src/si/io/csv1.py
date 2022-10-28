# -*- coding: utf-8 -*-
import sys

sys.path.insert(0, r'C:\Users\Asus\Desktop\Bioinformática\2º Ano\1º Semestre\Sistemas inteligentes\si\src\si')
print(sys.path)

import pandas as pd 
import numpy as np
from data.dataset import Dataset

def read_csv(filename: str, sep: str = ',', features: bool = False, label: bool = False) -> Dataset:
    '''
    Function reads a csv file and returns a Dataset
    Parameters
    ----------
    filename : str
        Path to the file
    sep : str, optional
        The separator used in the file by default is ','
    features : bool, optional
        Whether the file has a header by default is False
    label : bool, optional
        Whether the file has a label by default is False

    Returns
    -------
    Dataset
        The dataset object
    '''
    data = pd.read_csv(filename, sep=sep) 

    if features and label:
        features = data.columns[:-1]
        label = data.columns[-1] 
        X = data.iloc[:, :-1].to_numpy() 
        y = data.iloc[:, -1].to_numpy() 

    elif features and not label:
        features = data.columns
        X = data.to_numpy()
        y = None

    elif not features and label:
        X = data.iloc[:, :-1].to_numpy()
        y = data.iloc[:, -1].to_numpy()
        features = None
        label = None

    else: 
        X = data.to_numpy()
        y = None
        features = None
        label = None
    return Dataset(X, y, features = features, label = label) 


def write_csv(filename: str, dataset: Dataset, sep: str = ',', features: bool = False, label: bool = False) -> None:
    '''
    Writes a csv file from a dataset object
    Parameters
    ----------
    filename : str
        Path to the file
    dataset : Dataset
        The dataset object
    sep : str, optional
        The separator used in the file by default is ','
    features : bool, optional
        Whether the file has a header by default is False
    label : bool, optional
        Whether the file has a label by default is False
    '''
    data = pd.DataFrame(dataset.X)
    if features:
        data.columns = dataset.features
    if label:
        data[dataset.label] = dataset.y
    data.to_csv(filename, sep = sep, index = False)


if __name__ == "__main__":
    # TESTING
     file = r"C:\Users\Asus\Desktop\Bioinformática\2º Ano\1º Semestre\Sistemas inteligentes\si\datasets\iris.csv"
     a = read_csv(filename=file, sep = ",", features=True, label=4)
     print(a.print_dataframe())
     
    # print(a.summary())
    # write_csv(a, "csv_write_test1.csv", features=True, label=False)
    
    # TESTING MISSING VALUES METHODS ON DATASET CLASS
    #file = r"C:\Users\Asus\Desktop\Bioinformática\2º Ano\1º Semestre\Sistemas inteligentes\si\datasets\iris_missing_data.csv"
    #a = read_csv(filename=file, sep = ",", features=True, label=4)
    # print(a.dropna())
    # print(a.fillna(100))