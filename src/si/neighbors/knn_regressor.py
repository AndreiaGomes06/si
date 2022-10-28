# -*- coding: utf-8 -*-
from typing import Callable, Union
import sys

sys.path.insert(0, r'C:\Users\Asus\Desktop\Bioinformática\2º Ano\1º Semestre\Sistemas inteligentes\si\src\si')
sys.path.insert(0, r'C:\Users\Asus\Desktop\Bioinformática\2º Ano\1º Semestre\Sistemas inteligentes\si\src\si\io')
from data.dataset import Dataset
from statistic.euclidean_distance import euclidean_distance
from metrics.rmse import rmse

import numpy as np

class KNNRegressor:
    """
    KNN Regressor 
    k-Nearst Neighbors classifier is a machine learning model that approximates the association between independent variables 
    and the continuous outcome by averaging the observations in the same neighbourhood
    """
    def __init__(self, k: int = 1, distance: Callable = euclidean_distance):
        """
        Initialize the KNN Regressor
        Parameters
        ----------
        k: int
            The number of nearest neighbors to use
        distance: Callable
            The distance function to use
        """
        self.k = k
        self.distance = distance
        #atributes
        self.dataset = None

    def fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        It fits the model to the given dataset
        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        """
        self.dataset = dataset  
        return self

    def _get_closest_label(self, sample: np.ndarray) -> Union[int, str]:
        """
        It returns the closest label of the given sample
        Parameters
        ----------
        sample: np.ndarray
            The sample to get the closest label of
        """
        distances = self.distance(sample, self.dataset.X) 
        k_nearest_neighbors = np.argsort(distances)[:self.k] 
        k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors] 
                
        return np.mean(k_nearest_neighbors_labels)

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        It predicts the classes of the given dataset
        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the classes of
        """
        return np.apply_along_axis(self._get_closest_label, axis = 1, arr = dataset.X)

    def score(self, dataset: Dataset) -> float:
        """
        It returns the rmse of the model on the given dataset
        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on
        """
        predictions = self.predict(dataset)
        return rmse(dataset.y, predictions)


if __name__ == '__main__':
    # import dataset
    from data.dataset import Dataset
    from model_selection.split import train_test_split
    from csv1 import read_csv

    # load and split the dataset
    path = (r'C:\Users\Asus\Desktop\Bioinformática\2º Ano\1º Semestre\Sistemas inteligentes\si\datasets\cpu.csv')
    dataset_ = read_csv(path, sep = ",", features = True, label = True)
    dataset_train, dataset_test = train_test_split(dataset_, test_size = 0.2)

    # initialize the KNN classifier
    knn = KNNRegressor(k=3)

    # fit the model to the train dataset
    knn.fit(dataset_train)

    # evaluate the model on the test dataset
    score = knn.score(dataset_test)
    print(f'The accuracy of the model is: {score}')