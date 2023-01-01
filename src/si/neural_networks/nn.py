# -*- coding: utf-8 -*-

from typing import Callable
import numpy as np
import sys
sys.path.insert(0, r'C:\Users\Asus\Desktop\Bioinformática\2º Ano\1º Semestre\Sistemas inteligentes\si\src\si')
from data.dataset import Dataset
from metrics.mse import mse, mse_derivative
from metrics.accuracy import accuracy


class NN:
    """
    The NN is the Neural Network model.
    It comprehends the model topology including several neural network layers.
    The algorithm for fitting the model is based on backpropagation.
    """
    def __init__(self, layers: list , epochs: int = 1000, learning_rate: float = 0.01, loss: Callable = mse, loss_derivative: Callable = mse_derivative, verbose: bool = False):
        """
        Initialize the neural network model.
        Parameters
        ----------
        layers: list
            List of layers in the neural network.
        epochs: int
            Number of epochs to train the model.
        learning_rate: float
            The learning rate of the model.
        loss: Callable
            The loss function to use.
        loss_derivative: Callable
            The derivative of the loss function to use.
        verbose: bool
            Whether to print the loss at each epoch.
        """
        # parameters
        self.layers = layers
        self.epochs = int(epochs)
        self.learning_rate = learning_rate
        self.loss = loss
        self.loss_derivative = loss_derivative
        self.verbose = verbose #fazer copia para vermos o que está a acontecer

        # attributes
        self.history = {}


    def fit(self, dataset: Dataset) -> 'NN': 
        """
        It fits the model to the given dataset.
        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        """

        for epoch in range(1, self.epochs + 1): 
            # Extract the input data and the target data

            y_pred = dataset.X.copy()
            y_true = np.reshape(dataset.y, (-1, 1))  # reshape the target data to a column vector

            # forward propagation  
            for layer in self.layers:
                y_pred = layer.forward(y_pred)

            # backward propagation
            error = self.loss_derivative(y_true, y_pred) #metrica de erro derivado

            for layer in self.layers[::-1]: 
                error = layer.backward(error, self.learning_rate)

            # save history
            cost = self.loss(y_pred, y_true) 
            self.history[epoch] = cost

            # print loss
            if self.verbose: 
                print(f'Epoch {epoch}/{self.epochs} - cost: {cost}')

        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the output of the dataset
        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of
        """ 
        X = dataset.X.copy()
        # forward propagation
        for layer in self.layers:
            X = layer.forward(X)

        return X

    def cost(self, dataset: Dataset) -> float:
        """
        It computes the cost of the model on the given dataset.
        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the cost on
        """
        y_pred = self.predict(dataset)
        return self.loss(dataset.y, y_pred)

    def score(self, dataset: Dataset, scoring_func: Callable = accuracy) -> float:
        """
        It computes the score of the model on the given dataset.
        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the score on
        scoring_func: Callable
            The scoring function to use
        """
        y_pred = self.predict(dataset)
        return scoring_func(dataset.y, y_pred)



if __name__ == '__main__':
    from neural_networks.layers import Dense, Sigmoid_Activation, SoftMaxActivation, ReLUActivation

    X = np.array([[0,0],
                [0,1],
                [1,0],
                [1,2]])

    Y = np.array([1, 0, 0, 1])

    dataset = Dataset(X, Y, ['x1', 'x2'], 'x1 XNOR x2')
    print(dataset.print_dataframe())


    w1 = np.array([[20, -20],
                    [20, -20]])

    b1 = np.array([[-30, 10]])


    l1 = Dense(input_size = 2, output_size=2)
    l1.weights = w1
    l1.bias = b1


    w2 = np.array([[20],
                    [20]])

    b2 = np.array([[-10]])

    l2 = Dense(input_size = 2, output_size=1)
    l2.weights = w2
    l2.bias = b2

    l1_sa = Sigmoid_Activation()
    l2_sa = Sigmoid_Activation()

    nn_model_sa = NN(layers=[l1, l1_sa, l2, l2_sa])
    nn_model_sa.fit(dataset=dataset)


    print(nn_model_sa.predict(dataset))