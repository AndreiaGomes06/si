# -*- coding: utf-8 -*-

import itertools
import numpy as np
import sys

sys.path.insert(0, r'C:\Users\Asus\Desktop\Bioinformática\2º Ano\1º Semestre\Sistemas inteligentes\si\src\si')
from data.dataset import Dataset
from model_selection.split import train_test_split

class KMer:
    """
    A sequence descriptor that returns the k-mer composition of the sequence.
    Parameters
    ----------
    k : int
        The k-mer length.
    Attributes
    ----------
    k_mers : list of str
        The k-mers.
    """
    def __init__(self, k: int = 2,  alphabet: str = 'DNA'):
        """
        Parameters
        ----------
        k : int
            The k-mer length.
        alphabet: str
            Biological sequence alphabet
        """
        # parameters
        self.k = k
        self.alphabet = alphabet.upper()

        if self.alphabet == 'DNA':
            self.alphabet = 'ACTG'
        elif self.alphabet == 'PROT':
            self.alphabet = 'ACDEFGHIKLMNPQRSTVWY'
        else:
            raise TypeError('The sequence must be peptidic or from DNA.')
        
        self.k_mers = None

    def fit(self, dataset: Dataset) -> 'KMer': 
        """
        Fits the descriptor to the dataset. Returns self.
        Parameters
        ----------
        dataset : Dataset
            The dataset to fit the descriptor to.
        """
        # generate the k-mers
        self.k_mers = [''.join(k_mer) for k_mer in itertools.product(self.alphabet, repeat = self.k)]  
        return self 

    def _get_sequence_k_mer_composition(self, sequence: str) -> np.ndarray:
        """
        Calculates the k-mer composition of the sequence.
        Parameters
        ----------
        sequence : str
            The sequence to calculate the k-mer composition for.
        """
        # calculate the k-mer composition
        counts = {k_mer: 0 for k_mer in self.k_mers} 

        for i in range(len(sequence) - self.k + 1):
            k_mer = sequence[i: i + self.k] 
            counts[k_mer] += 1 

        # normalize the counts 
        return np.array([counts[k_mer] / len(sequence) for k_mer in self.k_mers])

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Transforms the dataset. Returns a new dataset object.
        Parameters
        ----------
        dataset : Dataset
            The dataset to transform.
        """
        # calculate the k-mer composition
        sequences_k_mer_composition = [self._get_sequence_k_mer_composition(sequence) for sequence in dataset.X[:, 0]] 
        sequences_k_mer_composition = np.array(sequences_k_mer_composition) 

        # create a new dataset
        return Dataset(X = sequences_k_mer_composition, y = dataset.y, features = self.k_mers, label = dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Fits the descriptor to the dataset and transforms the dataset.
        Parameters
        ----------
        dataset : Dataset
            The dataset to fit the descriptor to and transform.
        """
        self.fit(dataset)
        return self.transform(dataset)


if __name__ == '__main__':
    print("Example 1")
    from sklearn.preprocessing import StandardScaler
    from linear_model.logistic_regression import LogisticRegression
    sys.path.insert(0, r'C:\Users\Asus\Desktop\Bioinformática\2º Ano\1º Semestre\Sistemas inteligentes\si\src\si\io')
    from csv1 import read_csv
    path = r'C:\Users\Asus\Desktop\Bioinformática\2º Ano\1º Semestre\Sistemas inteligentes\si\datasets\tfbs.csv'
    tfbs = read_csv(path , sep=",", features = True, label = True)
    kmer = KMer(3, alphabet="DNA")
    kmer_dataset = kmer.fit_transform(tfbs)

    kmer_dataset.X = StandardScaler().fit_transform(kmer_dataset.X)
    train, test = train_test_split(kmer_dataset, test_size = 0.3, random_state = 2)

    log_reg = LogisticRegression(use_adaptive_alpha = False)
    log_reg.fit(train)
    print(f"Predictions: {log_reg.predict(test)}")
    print(f"Score: {log_reg.score(test)}")

    print("Example 2")
    path2 = r'C:\Users\Asus\Desktop\Bioinformática\2º Ano\1º Semestre\Sistemas inteligentes\si\datasets\transporters.csv'
    transporters = read_csv(path, sep = ",", features = True, label = True)
    kmers = KMer(2, alphabet="PROT")
    transporter_dataset = kmers.fit_transform(transporters)

    transporter_dataset.X = StandardScaler().fit_transform(transporter_dataset.X)
    train, test = train_test_split(transporter_dataset, test_size = 0.3, random_state = 2)

    log = LogisticRegression(use_adaptive_alpha = False)
    log.fit(train)
    print(f"Predictions: {log.predict(test)}")
    print(f"Score: {log.score(test)}")