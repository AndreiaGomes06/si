from typing import Tuple
import sys

sys.path.insert(0, r'C:\Users\Asus\Desktop\Bioinformática\2º Ano\1º Semestre\Sistemas inteligentes\si\src\si')
from data.dataset import Dataset

import numpy as np

def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets
    Returns a tuple with the train dataset and the test dataset
    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator
    """
    np.random.seed(random_state) 

    len_samples = dataset.shape()[0] 

    n_test = int(len_samples * test_size) 

    permutations = np.random.permutation(len_samples) 
    
    test_idxs = permutations[:n_test] 

    train_idxs = permutations[n_test:] 

    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features = dataset.features, label = dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features = dataset.features, label = dataset.label)
    return train, test