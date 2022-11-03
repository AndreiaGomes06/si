# -*- coding: utf-8 -*-  

import numpy as np
import sys
sys.path.insert(0, r'C:\Users\Asus\Desktop\Bioinformática\2º Ano\1º Semestre\Sistemas inteligentes\si\src\si')

from data.dataset import Dataset
from metrics.accuracy import accuracy
from statistic.sigmoid_function import sigmoid_function

class LogisticRegression:
    """
    The LogisticRegression is a logistic model using the L2 regularization.
    This model solves the logistic regression problem using an adapted Gradient Descent technique
    """
    def __init__(self, use_adaptive_alpha: bool, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 2000):
        """
        Initializes the LogisticRegression object.
        Parameters
        ----------
        use_adaptive_alpha: bool
            If the adaptive alpha is used or not in the gradient descent, which implies the use of different fit methods
        l2_penalty: float
            The L2 regularization parameter
        alpha: float
            The learning rate
        max_iter: int
            The maximum number of iterations
        Attributes
        ----------
        theta: np.array
            The model parameters, namely the coefficients of the linear model.
            For example, x0 * theta[0] + x1 * theta[1] + ...
        theta_zero: float
            The model parameter, namely the intercept of the linear model.
            For example, theta_zero * 1
        cost_history: dict
            The key is the descent gradient iteration number and the value is the cost in that iteration (using the cost function).
        """
        # parameters
        self.use_adaptive_alpha = use_adaptive_alpha
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter

        # attributes
        self.theta = None
        self.theta_zero = None
        self.cost_history = {}

    def gradient_descent(self, dataset: Dataset, m: int) -> None:  
        """
        Implements the gradient descent algorithm 
        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        m: int
            Number of examples in the dataset
        """
        # predicted y
        y_pred = np.dot(dataset.X, self.theta) + self.theta_zero

        # apply sigmoid function 
        y_pred = sigmoid_function(y_pred)

        # compute the gradient using the learning rate
        gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, dataset.X)

        # compute the penalty
        penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

        # update the model parameters
        self.theta = self.theta - gradient - penalization_term
        self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)


    def _regular_fit(self, dataset: Dataset) -> 'LogisticRegression':
        """
        Executes the gradient descent algorithm, that must stop when the value of the cost function doesn't change.
        When the difference between the cost of the previous and the current iteration is less than 0.0001, the Gradient Descent must stop.
        Fits the model without updating the learning rate (alpha).
        Returns self.
        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        """
        m, n = dataset.shape()

        # initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0

        for i in range(self.max_iter):
            self.gradient_descent(dataset, m) #gradient descent
            # creats the dictionary keys
            self.cost_history[i] = self.cost(dataset) 

            if i != 0 and self.cost_history[i-1] - self.cost_history[i] < 0.0001: 
                break
        return self


    def _adaptive_fit(self, dataset: Dataset) -> 'LogisticRegression':
        """
        Executes the gradient descent algorithm, that must decrease the alpha value when the value of the cost function doesn't change.
        When the difference between the cost of the previous and the current iteration is less than 0.0001.
        Fits the model updating the learning rate (alpha).
        Returns self.
        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        """
        m, n = dataset.shape()

        # initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0

        for i in range(self.max_iter):
            self.gradient_descent(dataset, m) #gradient descent
            # creats the dictionary keys
            self.cost_history[i] = self.cost(dataset) 

            if i != 0 and self.cost_history[i-1] - self.cost_history[i] < 0.0001: 
                self.alpha = self.alpha/2 
            
        return self


    def fit(self, dataset: Dataset) -> 'LogisticRegression':
        """
        Fit the model to the dataset
        If the use_adaptive_alfa is True, fits the model updating the alpha using the metod _adaptive_fit
        If the use_adaptive_alfa is False, fits the model not updating the alpha using the metod _regular_fit
        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        """
        if self.use_adaptive_alpha:
            return self._adaptive_fit(dataset)
        else:
            return self._regular_fit(dataset)

    def predict(self, dataset: Dataset) -> np.array:
        """
        Predict the output of the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of

        Returns
        -------
        predictions: np.array
            The predictions of the dataset
        """
        predictions = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)

        # convert the predictions to 0 or 1 (binarization)
        mask = predictions >= 0.5
        predictions[mask] = 1 
        predictions[~mask] = 0 
        return predictions 

    def score(self, dataset: Dataset) -> float:
        """
        Compute the Mean Square Error of the model on the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the MSE on

        Returns
        -------
        mse: float
            The Mean Square Error of the model
        """
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred) 

    def cost(self, dataset: Dataset) -> float:
        """
        Compute the cost function (J function) of the model on the dataset using L2 regularization

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the cost function on

        Returns
        -------
        cost: float
            The cost function of the model
        """
        predictions = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)
        cost = (-dataset.y * np.log(predictions)) - ((1 - dataset.y) * np.log(1 - predictions)) 
        cost = np.sum(cost) / dataset.shape()[0] 
        cost = cost + (self.l2_penalty * np.sum(self.theta ** 2) / (2 * dataset.shape()[0])) 
        return cost


if __name__ == '__main__':
    # import dataset
    from data.dataset import Dataset
    from model_selection.split import train_test_split

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # fit the model
    model = LogisticRegression(use_adaptive_alpha = True, l2_penalty=1, alpha=0.001, max_iter=1000)
    model.fit(dataset_train)

    # compute the score
    score = model.score(dataset_test)
    print(f"Score: {score}")
