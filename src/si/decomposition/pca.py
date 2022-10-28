import sys

sys.path.insert(0, r'C:\Users\Asus\Desktop\Bioinformática\2º Ano\1º Semestre\Sistemas inteligentes\si\src\si')
from data.dataset import Dataset

import numpy as np

class PCA:
    """
    It performs the Principal Component Analysis (PCA) on a givrn dataset, using the Singular Value Decomposition method.
    """
    def __init__(self, n_components: int) -> None:
        """
        PCA algorithm
        Parameters
        ----------
            n_components (int): Number of components to be considered and returned from the analysis.
        """
        self.n_components = n_components
        #atributes
        self.mean = None
        self.components = None
        self.explained_variance = None
    
    def fit(self, dataset: Dataset) -> tuple:
        """
        It fits the data and stores the mean values of each sample, the principial components and the explained variance.
        Parameters
        ----------
            dataset (Dataset): Dataset object.
        """
        self.mean = np.mean(dataset.X, axis = 0)
        self.cent_data = np.subtract(dataset.X, self.mean)

        U, S, Vt = np.linalg.svd(self.cent_data, full_matrices = False) 

        self.components = Vt[:self.n_components]

        n = len(dataset.X[:, 0])
        EV = S**2/(n-1)
        self.explained_variance = EV[:self.n_components]
        return self

    def transform(self, dataset: Dataset) -> tuple:
        """
        Returns the calculated reduced Singular Value Decomposition (SVD)
        Parameters
        ----------
            dataset (Dataset): Dataset object
        """
        V = self.components.T #transposta
        X_reduced = np.dot(self.cent_data, V) #numpy.dot(X, V)
        return X_reduced

    def fit_transform(self, dataset: Dataset) -> tuple:
        """
        It fit and transform the dataset
        Parameters
        ----------
            dataset (Dataset): Dataset object
        """
        self.fit(dataset)
        return self.transform(dataset)

if __name__ == "__main__":
    dataset = Dataset(X = np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y = np.array([0, 1, 0]),
                      features = ["f1", "f2", "f3", "f4"],
                      label = "y")

    a = PCA(n_components = 2)
    print(a.fit_transform(dataset))