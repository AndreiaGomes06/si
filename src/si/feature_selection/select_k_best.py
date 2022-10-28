
from typing import Callable

import sys

sys.path.insert(0, r'C:\Users\Asus\Desktop\Bioinformática\2º Ano\1º Semestre\Sistemas inteligentes\si\src\si')
from data.dataset import Dataset
from statistic.f_classification import f_classification
import numpy as np

class SelectKBest:
    """
    Select features according to the k highest scores.
        Feature ranking is performed by computing the scores of each feature using a scoring function:
            f_classification: ANOVA F-value between label/feature for classification tasks.
    """
    def __init__(self, score_func: Callable = f_classification, k: int = 10):
        """
        Select features according to the k highest scores.

        Parameters
        ----------
        score_func: callable
            Function taking dataset and returning a pair of arrays (scores, p_values)
        k: int, default=10
            Number of top features to select.
        Attributes
        ----------
        F: array, shape (n_features,)
            F scores of features.
        p: array, shape (n_features,)
            p-values of F-scores.
        """
        self.score_func = score_func
        self.k = k
        #atributes
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset) -> 'SelectKBest':
        """
        It fits SelectKBest to compute the F scores and p-values
        Parameters
        ----------
        dataset: Dataset
            A labeled dataset
        """
        self.F, self.p = self.score_func(dataset)
        return self
    
    def transform(self, dataset: Dataset) -> Dataset:
        """
        It transforms the dataset by selecting the k highest scoring features. 
        This function returns a new labeled Dataset with the k highest scoring features.
        Returns a new dataset object with the percentile highest score features.
        Parameters
        ----------
        dataset: Dataset
            A labeled dataset
        """
        idxs = np.argsort(self.F)[-self.k:]
        features = np.array(dataset.features)[idxs]
        return Dataset(X = dataset.X[:, idxs], y = dataset.y, features = list(features), label = dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        It fits SelectKBest and transforms the dataset by selecting the k highest scoring features.
        This funtion return a new Dataset object with the k highest scoring features.
        Parameters
        ----------
        dataset: Dataset
            A labeled dataset
        """
        self.fit(dataset)
        return self.transform(dataset)

if __name__ == "__main__":
    a = SelectKBest(score_func = f_classification, k=2)
    dataset = Dataset(X = np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y = np.array([0, 1, 0]),
                      features = ["f1", "f2", "f3", "f4"],
                      label = "y")

    a.fit(dataset)
    b = a.transform(dataset)
    print(b.features) 
