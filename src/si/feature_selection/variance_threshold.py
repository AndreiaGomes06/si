import sys


sys.path.insert(0, r'C:\Users\Asus\Desktop\Bioinformática\2º Ano\1º Semestre\Sistemas inteligentes\si\src\si\data')

import numpy as np
from dataset import Dataset

class VarianceThreshold:
    """
    Variance Threshold feature selection.
    Features with a training-set variance lower than this threshold will be removed from the dataset.
    """

    def __init__(self, threshold: float = 0.0):
        """
        Variance Threshold feature selection.
        Features with a training-set variance lower than this threshold will be removed from the dataset.
        Parameters
        ----------
        threshold: float
            The threshold value to use for feature selection. Features with a
            training-set variance lower than this threshold will be removed.
        Attributes
        ----------
        variance: array-like, shape (n_features,)
            The variance of each feature.
        """
        if threshold < 0:
            raise ValueError("Threshold must be non-negative")
        self.threshold = threshold
        #atributes
        self.variance = None

    def fit(self, dataset: Dataset) -> 'VarianceThreshold':
        """
        Fit the VarianceThreshold model according to the given training data.
        Parameters
        ----------
        dataset : Dataset
            The dataset to fit.
        """
        variance = np.var(dataset.X, axis = 0) 
        self.variance = variance 
        return self 

    def transform(self, dataset: Dataset) -> Dataset:
        """
        It removes all features whose variance does not meet the threshold. 
        This function returns a new Dataset object.
        Parameters
        ----------
        dataset: Dataset 
        """
        mask = self.variance > self.threshold
        X = dataset.X[:,mask] 
        features = np.array(dataset.features)[mask]
        return Dataset(X = X, y = dataset.y, features = list(features), label = dataset.label) 

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Fit to data, then transform it. 
        This function returns a new Dataset object.
        Parameters
        ----------
        dataset: Dataset 
        """
        self.fit(dataset)
        return self.transform(dataset)


if __name__ == '__main__':
    from dataset import Dataset
    dataset = Dataset(X = np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y = np.array([0, 1, 0]),
                      features = ["f1", "f2", "f3", "f4"],
                      label = "y")

    a = VarianceThreshold()
    a = a.fit(dataset)
    dataset = a.transform(dataset)
    print(dataset.features)

    #b = VarianceThreshold()
    #b = b.fit_transform(dataset)
    #print(b.features)