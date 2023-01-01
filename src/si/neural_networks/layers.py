# -*- coding: utf-8 -*-
import numpy as np
import sys
sys.path.insert(0, r'C:\Users\Asus\Desktop\Bioinformática\2º Ano\1º Semestre\Sistemas inteligentes\si\src\si')
from statistic.sigmoid_function import sigmoid_function

class Dense:
    """
    A dense layer is a layer where each neuron is connected to all neurons in the previous layer.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        Initialize the dense layer.
        Parameters
        ----------
        input_size: int
            The number of inputs the layer will receive.
        output_size: int
            The number of outputs the layer will produce.
        Attributes
        ----------
        weights: np.ndarray
            The weights of the layer.
        bias: np.ndarray
            The bias of the layer.
        """
        # parameters
        self.input_size = input_size
        self.output_size = output_size

        # attributes 
        self.weights = np.random.randn(input_size, output_size) * 0.01  
        self.bias = np.zeros((1, output_size)) 
        self.X = None

    def forward(self, input_data: np.ndarray) -> np.ndarray: 
        """
        Performs a forward pass of the layer using the given input.
        Returns a 2d numpy array with shape (1, output_size). Returns the output of the layer.
        Parameters
        ----------
        input_data: np.ndarray
            The input to the layer.   #o input data é uma matriz de n exemplo por n features
        """
        self.X = input_data
        return np.dot(input_data, self.weights) + self.bias 

    def backward(self, error:np.ndarray, learning_rate:float = 0.01) -> np.ndarray:
        """
        Computes the backward pass of the layer
        Parameters
        ----------
        error: np.ndarray
            Error function.
        learning_rate: float
            Learning rate.
        """
        error_to_propagate = np.dot(error, self.weights.T) 

        # updates the weights and bias 
        self.weights = self.weights - learning_rate*np.dot(self.X.T, error)  # x.T is used to multiply the error by

        self.bias = self.bias - learning_rate * np.sum(error, axis = 0)  

        return error_to_propagate 
        
class Sigmoid_Activation:
    """
    A sigmoid activation layer.
    """

    def __init__(self):
        """
        Initialize the sigmoid activation layer.
        """
        self.X = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:  
        """
        Performs a forward pass of the layer using the given input.
        Returns a 2d numpy array with shape (1, output_size).
        Parameters
        ----------
        input_data: np.ndarray
            The input to the layer.
        """
        self.X = input_data
        return sigmoid_function(input_data)  

    def backward(self, error: np.ndarray, learning_rate:float) -> np.ndarray: 
        """
        Computes the backward pass of the layer
        Parameters
        ----------
        error: np.ndarray
            Error function.
        """
        # multiplication of each element by the derivative and not by the entire matrix
        deriv_sig =  sigmoid_function(self.X) * (1-sigmoid_function(self.X)) 
        #backward pass:
        error_to_propagate = error * deriv_sig 
        return error_to_propagate

class SoftMaxActivation: 
    """
    A SoftMax Activation layer. This layer is applied to multiclass problems.
    """

    def __init__(self): 
        """
        Initialize the SoftMax Activation layer.
        """
        self.X = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the layer using the given input.
        Softmax algorithm: e**(input_data - max(input_data)) / np.sum(e**(input_data - max(input_data)))
        Parameters
        ----------
        input_data: np.ndarray
            The input to the layer.
        """
        self.X = input_data
        e = np.exp(input_data - np.max(input_data))
        return e / np.sum(e, axis = 1, keepdims = True) 

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Computes the backward pass of the layer
        Parameters
        ----------
        error: np.ndarray
            Error function.
        learning_rate: float
            Learning rate.
        """
        return error

class ReLUActivation: 
    """
    A ReLUActivation layer. This layer considers only positive values.
    """

    def __init__(self):
        """
        Initialize the ReLUActivation layer.
        """
        self.X = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the layer using the given input.
        Parameters
        ----------
        input_data: np.ndarray
            The input to the layer.
        """
        self.X = input_data
        return np.maximum(0, input_data)

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Computes the backward pass of the layer
        Parameters
        ----------
        error: np.ndarray
            Error function.
        """
        relu_derivative = np.where(self.X > 0, 1, 0) 
        error_to_propagate = error * relu_derivative 

        return error_to_propagate


class LinearActivation:
    """
    A LinearActivation layer. 
    """
    def __init__(self) -> None:
        """
        Initialize the LinearActivation layer.
        """
        self.X = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Calculates the linear activation algorithm.
        Parameters
        ----------
        input_data: np.ndarray
            The input values to be activated
        """
        self.X = input_data
        return input_data

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Computes the backwards pass of layer.
        Parameters
        ----------
        error: np.ndarray
            The error propagated to the layer
        alpha: float
            The learning rate of the model
        """
        return error * np.ones_like(self.X) 