# -*- coding: utf-8 -*-

from typing import Tuple, Sequence
import numpy as np
import pandas as pd 

class Dataset:
    def __init__(self, X: np.ndarray, y: np.ndarray = None, features: list = None, label: str = None):
        """
        Dataset represents a machine learning tabular dataset.
        
        Paramaters
        ----------
        X: np.ndarray
            A matrix with the dataset features
        y: np.ndarray
            Label vector
        features: list of strings
            Names of the features
        label: str
            Name of the label
        """
        if X is None:
            raise ValueError("X can't be None")
        if features is None:
            features = [str(i) for i in range(X.shape[1])]
        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def shape(self) -> Tuple:
        """
        Returns tuple with the dataset dimensions 
        """
        return self.X.shape

    def has_label(self) -> bool:
        """
        Checks if the dataset contains a label
        """
        if self.y is not None:
            return True 
        else: return False
    
    def get_classes(self) -> np.ndarray:
        """
        Returns the unique classes in the dataset
        """
        if self.y is None:
            raise ValueError("Dataset does not have a label")
        else:
            return np.unique(self.y) 

    def get_mean(self) -> np.ndarray:
        """
        Returns a np.ndarray with the mean for each feature of the dataset
        """
        return np.nanmean(self.X, axis=0) 

    def get_variance(self) -> np.ndarray:
        """
        Returns a np.ndarray with the variance for each feature of the dataset
        """
        return np.nanvar(self.X, axis=0)

    def get_median(self) -> np.ndarray:
        """
        Returns a np.ndarray with the median for each feature of the dataset
        """
        return np.nanmedian(self.X, axis=0)
    
    def get_min(self) -> np.ndarray:
        """
        Returns a np.ndarray with the minimum for each feature of the dataset
        """
        return np.nanmin(self.X, axis=0)

    def get_max(self) -> np.ndarray:
        """
        Returns a np.ndarray with the maximum for each feature of the dataset
        """
        return np.nanmax(self.X, axis=0)

    def summary(self):
        """
        Returns a summary of the dataset
        """
        return pd.DataFrame(
            {"mean": self.get_mean(),
             "median": self.get_variance(),
             "variance": self.get_median(),
             "min": self.get_min(),
             "max": self.get_max()}
        )

    def print_dataframe(self):
        """
        Prints dataframe in pandas DataFrame format
        """
        if self.y is None:
            return pd.DataFrame(self.X, columns=self.features)
        else:
            return pd.DataFrame(self.X, columns=self.features, index=self.y)

    def dropna(self):
        """
        Remove all samples that contain at least one null value (NaN)
        """
        mask_na = np.isnan(self.X).any(axis = 1) 
        if self.has_label():
            self.y = self.y[~mask_na] 
        self.X = self.X[~mask_na] 

    def fillna(self, value: int):
        """
        Replaces all null values with a given value
        Paramaters
        ----------
        value: int
            Given value to replace the NaN values with
        """
        return np.nan_to_num(self.X, nan = value, copy = False) #substitui todos os valores que tinham na com o value por nós dado

    
    @classmethod
    def from_random(cls,
                    n_samples: int,
                    n_features: int,
                    n_classes: int = 2,
                    features: Sequence[str] = None,
                    label: str = None):
        """
        Creates a Dataset object from random data
        Parameters
        ----------
        n_samples: int
            The number of samples
        n_features: int
            The number of features
        n_classes: int
            The number of classes
        features: list of str
            The feature names
        label: str
            The label name
        Returns
        -------
        Dataset
        """
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        return cls(X, y, features=features, label=label)


if __name__ == '__main__':
    x = np.array([[1,2,3], [1,2,3]])  # matriz
    y = np.array([1,2,2])  # vetor
    features = ['A', 'B', 'C']
    label = 'y'  # nome do vetor
    dataset = Dataset(X = x, y = y, features = features, label = label)

    print('Shape:',dataset.shape())
    print('Label:',dataset.has_label())
    print('Classes:',dataset.get_classes())
    print('Summary:',dataset.summary())

    print('----Testing the drop Nan----')
    x1 = np.array([[1,2,3], [1,np.nan,np.nan], [4,7,np.nan]]) 
    y1 = np.array([1,3,2]) 
    dataset2 = Dataset(X = x1, y = y1, features = features, label = label)
    print(dataset2.print_dataframe())  # before removing the Nan
    dataset2.dropna()
    print(dataset2.print_dataframe())  # after removing the Nan

    print('----Testing the fill Nan----')
    dataset3 = Dataset(X = x1, y = y1, features = features, label = label)
    print(dataset3.print_dataframe())  
    dataset3.fillna(5)
    print(dataset3.print_dataframe())  