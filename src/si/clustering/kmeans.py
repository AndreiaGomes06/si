import sys

sys.path.insert(0, r'C:\Users\Asus\Desktop\Bioinformática\2º Ano\1º Semestre\Sistemas inteligentes\si\src\si')
from statistic.euclidean_distance import euclidean_distance
from data.dataset import Dataset

import numpy as np
from typing import Callable

class KMeans:
    """
    It performs k-means clustering on the dataset.
    It groups samples into k clusters by trying to minimize the distance between samples and their closest centroid.
    It returns the centroids and the indexes of the closest centroid for each point.
    """
    def __init__(self, k: int, max_iter: int = 1000, distance: Callable = euclidean_distance):
        """
        K-means clustering algorithm.
        Parameters
        ----------
        k: int
            Number of clusters.
        max_iter: int
            Maximum number of iterations.
        distance: Callable
            Distance function.
        """
        self.k = k
        self.max_iter = max_iter
        self.distance = distance
        #parametros estimados
        self.centriods = None
        self.labels = None

    def _init_centriods(self, dataset: Dataset):
        """
        It generates initial k centroids.
        Parameters
        ----------
        dataset: Dataset
            Dataset object.
        """
        seeds = np.random.permutation(dataset.X.shape[0])[:self.k]
        self.centriods = dataset.X[seeds] 

    def _get_closest_centroid (self,  sample: np.ndarray) -> np.ndarray:
        """
        Get the closest centroid to each data point.
        Parameters
        ----------
        sample: np.ndarray, shape=(n_features,)
            A sample.
        """
        centroids_distances = self.distance(sample, self.centriods) 
        closest_centroid_index = np.argmin(centroids_distances, axis=0) 
        return closest_centroid_index

    def fit(self, dataset: Dataset) -> 'KMeans':
        """
        It fits k-means clustering on the dataset.
        The k-means algorithm initializes the centroids and then iteratively updates them until convergence or max_iter.
        Convergence is reached when the centroids do not change anymore.
        Parameters
        ----------
        dataset: Dataset
            Dataset object.
        """
        
        self._init_centriods(dataset) # Usa a 1ª função para inicializa k centroids

        # fitting the k-means
        convergence = False # variable to check differences between
        i = 0 # variable to check max_iter
        labels = np.zeros(dataset.shape()[0]) 

        while not convergence and i < self.max_iter: 

            # get closest centroid - index 
            new_labels = np.apply_along_axis(self._get_closest_centroid, axis=1, arr=dataset.X)

            # compute the new centroids
            centroids = []
            for j in range(self.k):
                centroid = dataset.X[new_labels == j]
                centroid_mean = np.mean(centroid, axis=0) #colunas
                centroids.append(centroid_mean)

            self.centroids = np.array(centroids)

            # check if the centroids have changed, Convergence is reached when the centroids do not change anymore.
            convergence = np.any(new_labels != labels) 

            # replace labels
            labels = new_labels

            # increment counting
            i += 1

        self.labels = labels
        return self

    def _get_distances(self, sample: np.ndarray) -> np.ndarray:
        """
        It computes the distance between each sample and the closest centroid.
        Parameters
        ----------
        sample : np.ndarray, shape=(n_features,)
            A sample.
        """
        return self.distance(sample, self.centroids)

    def transform(self, dataset: Dataset) -> np.ndarray:
        """
        It transforms the dataset.
        It computes the distance between each sample and the closest centroid.
        Parameters
        ----------
        dataset: Dataset
            Dataset object.
        """
        return np.apply_along_axis(self._get_distances, axis = 1, arr = dataset.X)

    def fit_transform(self, dataset: Dataset) -> np.ndarray:
        """
        It fits and transforms the dataset.
        Parameters
        ----------
        dataset: Dataset
            Dataset object.
        """
        self.fit(dataset)
        return self.transform(dataset)

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        It predicts the labels of the dataset.
        Parameters
        ----------
        dataset: Dataset
            Dataset object.
        """
        return np.apply_along_axis(self._get_closest_centroid, axis = 1, arr = dataset.X)

    def fit_predict(self, dataset: Dataset) -> np.ndarray:
        """
        It fits and predicts the labels of the dataset.
        Parameters
        ----------
        dataset: Dataset
            Dataset object.
        """
        self.fit(dataset)
        return self.predict(dataset)

if __name__ == '__main__':
    dataset = Dataset(X = np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y = np.array([0, 1, 0]),
                      features = ["f1", "f2", "f3", "f4"],
                      label = "y")

    k_ = 3
    kmeans = KMeans(k_)
    res = kmeans.fit_transform(dataset)
    predictions = kmeans.predict(dataset)
    print(res.shape)
    print(predictions.shape)